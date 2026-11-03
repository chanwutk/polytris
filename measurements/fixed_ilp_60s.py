"""Run the fixed-test ILP benchmark with a 60-second limit per clip."""
from pathlib import Path

from measurements import fixed_ilp_unlimited as benchmark


benchmark.OUTPUT = Path('/polyis/measurements-results/fixed_ilp_60s')
benchmark.TIME_LIMIT_SECONDS = 60


if __name__ == '__main__':
    benchmark.main()
