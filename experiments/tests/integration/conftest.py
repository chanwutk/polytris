"""Integration tests run only with POLYIS_INTEGRATION=1: they need a GPU, a Gurobi license, and the detector weights.

    POLYIS_INTEGRATION=1 pytest experiments/tests/integration -v -s
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line('markers', 'integration: scripts-vs-pipelines tests (GPU, Gurobi, detector weights)')


def pytest_collection_modifyitems(config, items):
    # Without POLYIS_INTEGRATION=1 every integration test is skipped, so a plain `pytest` stays fast.
    if os.environ.get('POLYIS_INTEGRATION') == '1':
        return
    skip = pytest.mark.skip(reason='integration test: set POLYIS_INTEGRATION=1 to run it')
    for item in items:
        if 'integration' in item.keywords:
            item.add_marker(skip)
