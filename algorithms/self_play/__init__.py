from ray.tune.registry import register_trainable

from algorithms.self_play.evaluation import load_checkpoints
from algorithms.self_play.simultaneous import Simultaneous
from algorithms.self_play.self_play import SelfPlay

register_trainable("SIMULTANEOUS_PLAY", Simultaneous)
register_trainable("SELF_PLAY", SelfPlay)
