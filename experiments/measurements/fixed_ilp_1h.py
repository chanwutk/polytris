"""Rerun, with a 1-hour limit per clip, the fixed-test ILPs whose 60-second run hit its cap.

Records Gurobi's status and proven lower bound with each solution, so a clip that hits the hour still bounds how far
its selection is from optimal. Uses the container's default cache, which reproduces every clip's recorded ILP size.
"""
import json
from pathlib import Path

from experiments.measurements import fixed_ilp_unlimited as benchmark


SIXTY_SECOND_RESULTS = Path('/polyis/measurements-results/fixed_ilp_60s/results.jsonl')
# Number of clips whose 60-second run reached the cap, as reported in the paper.
CAPPED_CLIPS = 48


def capped_clips(results_path: Path, cap_ms: float = 60_000) -> set[tuple[str, str]]:
    """(dataset, video) of every run in `results_path` whose solve reached `cap_ms`."""
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    return {(row['dataset'], row['video']) for row in rows if row['solve_ms'] >= cap_ms}


def configure():
    """Point the benchmark at the 1-hour rerun of the capped clips."""
    clips = capped_clips(SIXTY_SECOND_RESULTS)
    assert len(clips) == CAPPED_CLIPS, f'expected {CAPPED_CLIPS} capped clips, found {len(clips)}'
    benchmark.OUTPUT = Path('/polyis/measurements-results/fixed_ilp_1h')
    benchmark.TIME_LIMIT_SECONDS = 3600
    benchmark.CACHE_DIR = None
    benchmark.CLIPS = clips


if __name__ == '__main__':
    configure()
    benchmark.main()
