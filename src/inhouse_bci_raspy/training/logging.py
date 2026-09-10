"""Select TensorBoard logging for formal training or a no-op writer for smoke checks."""

from pathlib import Path


class SmokeWriter:
    """Implement the writer interface required by the unchanged training engine."""

    def add_scalar(self, *args, **kwargs) -> None:
        """Discard scalar events during a smoke check."""

    def close(self) -> None:
        """Finish the no-op writer without creating log files."""


def create_writer(directory: Path, smoke: bool):
    """Return a writer without requiring TensorBoard for the smoke-only stage."""
    if smoke:
        return SmokeWriter()
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=str(directory))
