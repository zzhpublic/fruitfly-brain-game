"""Game environments for fruit fly brain."""
from .pong import PongEnv
from .maze import MazeEnv
from .odor import OdorNavigationEnv
from .looming import LoomingEscapeEnv
from .pinball import PinballEnv

__all__ = ["PongEnv", "MazeEnv", "OdorNavigationEnv", "LoomingEscapeEnv", "PinballEnv"]
