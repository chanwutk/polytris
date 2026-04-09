# Outstanding Items

## 1. Preload mode
`pipeline.py:264` has a `pass` where the preload buffering logic should go.
The comment says it is delegated to `stage_decode.py` — verify that the
preload path is fully implemented there and that the `pass` is intentional.

## 2. Pixel-exact tile rendering test
The plan (`PLAN.md`) explicitly flags GPU tile rendering as a known risk:
> "Edge-repeat in GPU rendering... Needs pixel-exact testing."

No test exists yet. Write a test that compares the PyTorch GPU tile-copy
output against the original CPU/numpy implementation pixel-by-pixel,
covering boundary tiles where source and destination sizes differ.

## 3. Integration smoke test
No test runs the pipeline end-to-end. Add a smoke test that feeds a small
synthetic video through the full pipeline (Decode → Classify → Compress →
Detect → Track) and asserts a result is produced without errors.

## 4. First real run
The pipeline has not been executed yet. Run it and confirm it produces
correct output:

```bash
python execution/main.py --test
```
