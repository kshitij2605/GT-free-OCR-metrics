"""Stub ExperimentTracker — no-op MLflow replacement for remote GPU host."""

import contextlib


class ExperimentTracker:
    def __init__(self, *args, **kwargs):
        pass

    @contextlib.contextmanager
    def start_run(self, *args, **kwargs):
        yield self

    def log_params(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass
