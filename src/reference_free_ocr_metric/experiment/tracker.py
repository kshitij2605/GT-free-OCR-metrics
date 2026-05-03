"""Stub ExperimentTracker — no-op MLflow replacement for remote GPU host."""

import contextlib


class ExperimentTracker:
    """No-op experiment tracker -- stub replacement for MLflow on remote GPU hosts."""
    def __init__(self, *args, **kwargs):
        """Accept and discard all initialization arguments."""
        pass

    @contextlib.contextmanager
    def start_run(self, *args, **kwargs):
        """Context manager stub: yield self without starting any tracking run."""
        yield self

    def log_params(self, *args, **kwargs):
        """No-op parameter logging stub."""
        pass

    def log_metrics(self, *args, **kwargs):
        """No-op metrics logging stub."""
        pass

    def log_artifact(self, *args, **kwargs):
        """No-op artifact logging stub."""
        pass
