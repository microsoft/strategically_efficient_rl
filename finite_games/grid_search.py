'''
Implements grid-search over training configurations.
'''

from collections import namedtuple
from copy import deepcopy
import pandas as pd

Parameter = namedtuple("Parameter", ["key", "value"])


def get_parameters(dictionary, base_key=[]):

    if "grid_search" in dictionary:
        assert len(dictionary) == 1, "'grid_search' entries must be unique"
        assert isinstance(dictionary["grid_search"], list), "'grid_search' value must be a list of parameters"
        
        return [Parameter(base_key, dictionary["grid_search"])]
    else:
        parameters = []

        for key, value in dictionary.items():
            if isinstance(value, dict):
                parameters += get_parameters(value, base_key + [key])
        
        return parameters


def set_recursive(dictionary, key, value):
    for idx in range(len(key) - 1):
        dictionary = dictionary[key[idx]]

    dictionary[key[-1]] = value


def get_variations(base_name, base_config, parameters, fixed_parameters=[]):

    if len(parameters) == 0:
        name = []

        for param in fixed_parameters:
            if isinstance(param.value, dict):
                for k, v in param.value.items():
                    name.append('='.join((k, str(v))))
            else:
                name.append('='.join((param.key[-1], str(param.value))))

        name = ','.join(name)
        name = '_'.join((base_name, name))

        config = deepcopy(base_config)

        for p in fixed_parameters:
            set_recursive(config, p.key, p.value)

        return {name: config}
    else:
        variations = {}

        for value in parameters[0].value:
            parameter = Parameter(parameters[0].key, value)
            variations.update(get_variations(base_name, 
                                             base_config, 
                                             parameters=parameters[1:],
                                             fixed_parameters=fixed_parameters + [parameter]))

        return variations


def grid_search(name, config):
    parameters = get_parameters(config)

    if len(parameters) == 0:
        return None
    else:
        return get_variations(name, config, parameters)
