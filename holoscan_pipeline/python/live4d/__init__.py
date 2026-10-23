"""Live 4D-STEM acquisition and processing."""

__all__ = ["PipelineApp", "PipelineConfig"]


def __getattr__(name):
    if name == "PipelineConfig":
        from .config import PipelineConfig
        return PipelineConfig
    if name == "PipelineApp":
        from .app import PipelineApp
        return PipelineApp
    raise AttributeError(name)
