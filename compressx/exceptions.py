class CompressXError(Exception):
    """Base exception for the project."""


class ConfigurationError(CompressXError):
    """Raised when user configuration is invalid."""


class StageExecutionError(CompressXError):
    """Raised when a pipeline stage fails."""


class DiskSpaceError(CompressXError):
    """Raised when there is not enough disk space to continue."""
