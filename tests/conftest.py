import pytest

from kmer_ord.utils.logging_utils import close_run_log


@pytest.fixture(autouse=True)
def _close_run_log_after_test():
    """A command test must not leave the tee pointed at a deleted tmp file."""
    yield
    close_run_log()
