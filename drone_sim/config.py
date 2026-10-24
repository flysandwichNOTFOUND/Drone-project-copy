"""Configuration records; detailed parameters will follow later."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SimulationConfig:
    scene: Literal["indoor", "outdoor"] = "indoor"
    gui: bool = False
    timestep: float = 1.0 / 240.0
    gravity: float = -9.81


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    frames_per_second: int = 15


@dataclass
class DeepSeekConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = ""
    timeout_seconds: float = 10.0


@dataclass
class AppConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
