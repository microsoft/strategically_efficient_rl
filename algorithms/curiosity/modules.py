from algorithms.curiosity.random_network_distillation import RandomNetworkDistillation
# from algorithms.curiosity.intrinsic_curiosity_module import IntrinsicCuriosityModule
# from algorithms.curiosity.opponent_value_prediction import OpponentValuePrediction
from algorithms.curiosity.shaping import PotentialShaping
from algorithms.curiosity.hashing import Hashing

MODULES = {
    "RND": RandomNetworkDistillation,
    # "ICM": IntrinsicCuriosityModule,  # NOTE: Why did we remove these - seems to be a switch between Ray 0.8.3 and 1.0.0
    # "OVP": OpponentValuePrediction,
    "shaping": PotentialShaping,
    "hashing": Hashing,
}

def get_module_class(name):
    if name not in MODULES:
        raise ValueError(f"No curiosity module named '{name}' is defined")

    return MODULES[name]
