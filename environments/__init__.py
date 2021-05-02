from ray.tune.registry import register_env


def create_particle_env(config):
    from environments.particle_envs import ParticleEnv
    return ParticleEnv(config)


def create_spiel_env(config):
    from environments.openspiel import OpenSpielGame
    return OpenSpielGame(config)


def create_deep_sea(config):
    from environments.discrete_wrapper import DiscreteWrapper
    from environments.deep_sea import DeepSea
    return DiscreteWrapper(DeepSea(config))


def create_deep_bandit(config):
    from environments.discrete_wrapper import DiscreteWrapper
    from environments.deep_bandit import DeepBandit
    return DiscreteWrapper(DeepBandit(config))


def create_random_tree(config):
    from environments.discrete_wrapper import DiscreteWrapper
    from environments.random_tree import RandomTree
    return DiscreteWrapper(RandomTree(config))


def create_contextual_bandit(config):
    from environments.contextual_bandit import ContextualBandit
    return ContextualBandit(config)


def create_roshambo(config):
    from environments.roshambo import ContextualRoShamBo
    return ContextualRoShamBo(config)


def create_gym_atari(config):
    from environments.gym_atari import GymAtari
    return GymAtari(config)


def create_pettingzoo_atari(config):
    from environments.pettingzoo_atari import PettingZooAtari
    return PettingZooAtari(config)


def create_pettingzoo_mpe(config):
    from environments.pettingzoo_mpe import PettingZooMPE
    return PettingZooMPE(config)


register_env("mpe", create_particle_env)
register_env("openspiel", create_spiel_env)
register_env("deep_sea", create_deep_sea)
register_env("deep_bandit", create_deep_bandit)
register_env("contextual_bandit", create_contextual_bandit)
register_env("roshambo", create_roshambo)
register_env("random_tree", create_random_tree)
register_env("gym_atari", create_gym_atari)
register_env("pettingzoo_atari", create_pettingzoo_atari)
register_env("pettingzoo_mpe", create_pettingzoo_mpe)
