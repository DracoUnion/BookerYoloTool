"""Domain exception hierarchy for the YOLO tool (zero Qt)."""


class YoloToolError(Exception):
    """Base class for all tool errors."""


class ModelError(YoloToolError):
    """Raised when a model cannot be loaded or configured."""


class InferenceError(YoloToolError):
    """Raised when inference on an image fails."""


class TrainError(YoloToolError):
    """Raised when model training fails."""