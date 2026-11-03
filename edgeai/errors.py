"""Expected errors exposed by the image- and video-detection package."""


class ImageInputError(ValueError):
    """Raised when an image path or file cannot be used for inference."""


class VideoInputError(ValueError):
    """Raised when a video path or stream cannot be used for inference."""


class ModelLoadError(RuntimeError):
    """Raised when the configured pretrained model cannot be loaded."""


class DeviceUnavailableError(RuntimeError):
    """Raised when a specifically requested inference device is unavailable."""


class InferenceError(RuntimeError):
    """Raised when model prediction or annotated output generation fails."""


class VideoProcessingError(RuntimeError):
    """Raised when video inference or output encoding fails."""


class WebcamInputError(RuntimeError):
    """Raised when a webcam cannot be opened or its first frame cannot be read."""
