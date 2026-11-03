"""Inspect saved fixed-configuration ILP inputs and canvases without solving or rendering."""
import json
from pathlib import Path

import numpy as np
import duckdb

from measurements.common import decode_grids
from measurements import r9o3_ilp_instance_sizes as sizes
from polyis.io import cache


def main():
    root = Path('/polyis/measurements-results/fixed_test_s1_square_m02')
    inputs = json.loads((root / 'inputs.json').read_text())
    # Export counted operations in bulk; aggregate analysis stays in DuckDB.
    con = duckdb.connect(config={'threads': 4})
    con.execute("""COPY (SELECT regexp_extract(filename,'/([^/]+)\\.json$',1) AS dataset,
        stage,op,sum(time) AS seconds,count(*) AS entries
        FROM read_json_auto(?,filename=true) GROUP BY ALL)
        TO ? (FORMAT JSON,ARRAY true)""", [str(root / 'timings/*.json'), str(root / 'operator_op_times.json')])
    records = []
    for dataset, info in inputs.items():
        # Existing instance-counting helpers use the current dataset's isolated read-only cache.
        cache.CACHE_DIR = f'/runs/{dataset}/polyis-cache'
        distance = sizes.max_distance(dataset, 'sortcython', 60, .8, 1)
        for video in info['videos']:
            score = cache.exec(dataset, 'relevancy', video, 'ShuffleNet05_60_1', 'score/score.jsonl')
            grids, indices = decode_grids([json.loads(line) for line in score.read_text().splitlines()])
            tile_ids, lengths = sizes.group((grids >= np.floor(.5 * 255)).astype(np.uint8))
            structure = sizes.ilp_structure(tile_ids, lengths, distance)
            directory = cache.exec(dataset, 'pruned-polyominoes', video, 'ShuffleNet05_60_1_080_r050_sortcython', 'score')
            pruned, kept_indices = decode_grids([json.loads(line) for line in (directory / 'score.jsonl').read_text().splitlines()])
            assert indices == kept_indices
            check = sizes.check_solution(structure, tile_ids, pruned)
            assert check['tiles_outside_input'] == 0 and check['split_polyominoes'] == 0, (dataset, video, check)
            timing = json.loads((directory / 'runtime.jsonl').read_text())
            config = 'ShuffleNet05_60_1_080_r050_square_s100_sortcython'
            canvases = len(list(cache.exec(dataset, 'comp-frames', video, config, 'images').glob('*.jpg')))
            # Keep the existing model-size reconstruction; no Gurobi call is made.
            records.append(dict(dataset=dataset, video=video, frames=len(indices), canvases=canvases,
                                **sizes.instance_size(structure), runtime=timing, solution_check=check))
        print(dataset, 'inspected', len(info['videos']), 'cached clips', flush=True)
    assert len(records) == 200
    (root / 'operator_details.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
