"""DECTRIS transport selection and shared Holoscan 4D-STEM processing."""

__all__ = ["PipelineApp", "PipelineConfig"]


def __getattr__(name):
    if name == "PipelineConfig":
        from .config import PipelineConfig
        return PipelineConfig
    if name == "PipelineApp":
        from .app import PipelineApp
        return PipelineApp
    raise AttributeError(name)
