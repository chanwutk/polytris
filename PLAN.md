# Groundtruth Detection Update Plan & Execution Summary

## 1) Conversation Summary

This thread focused on redesigning groundtruth-detection generation in preprocess stage `p003`.

### Original objective
- Rework `preprocess/p003_preprocess_groundtruth_detection.py` so groundtruth detections are generated in two different ways:
  1. **CalDOT / AMS family**: use existing groundtruth source data, but remove detections that fall into masked regions.
  2. **JNC family** (`jnc0`, `jnc2`, `jnc6`, `jnc7`): re-run detection from **original source videos at 30fps**, not from preprocessed 15fps split videos.

### Clarifications and decisions captured in-thread
- AMS source root for OTIF-style data: `/otif-dataset/dataset/amsterdam`.
- JNC preprocessing behavior before detection: keep **crop + domain mask + vertical rotation** from `p000` and remove resize in that stage.
- JNC 4-corner inference behavior: detect on 4 overlapping corner crops, then map offsets back, then run NMS.
- JNC output cadence: write detections for **all 30fps frames** in the relevant segment.
- JNC postprocessing update from user comments: after merge/NMS, **scale boxes to 1080x720** (same target frame size in `p000`).
- CalDOT/AMS source location update: stop using `SOURCE_DIR/{dataset}` for this path; use OTIF dataset root and add `OTIF_DATASET` in global config.
- Mask logic update for CalDOT/AMS:
  - First apply **include** mask: keep only overlapping detections and clip to include region.
  - Then apply **exclude** mask: remove detections overlapping exclusion polygon.
- AMS behavior correction: AMS has no GT detection JSON source to copy; it must run detector inference (and user requested forcing AMS yolov3 behavior).

---

## 2) Final Agreed Requirements (Consolidated)

### A. JNC datasets (`jnc0`, `jnc2`, `jnc6`, `jnc7`)
1. Read from original source video (30fps).
2. Use segment-range logic equivalent to `p000` line range where segment start/end are computed from `num_segments`.
3. Apply B3D preprocessing geometry before detection:
   - crop by annotation box,
   - apply domain mask,
   - rotate if vertical (same logic as `p000`).
4. Perform 4-corner detection on each frame:
   - TL, TR, BL, BR
   - each crop is `2/3` width by `2/3` height.
5. Remap detections to frame coordinates via offsets.
6. Merge with NMS.
7. Scale final boxes to target `1080x720`.
8. Output JSONL with one line per 30fps frame in selected segment.

### B. CalDOT and AMS datasets
1. Use OTIF-rooted data source pathing (config driven).
2. Apply mask filtering in this order:
   - include-mask clipping,
   - exclude-mask removal.
3. CalDOT: copy from existing GT JSON annotation source (OTIF path).
4. AMS: generate detections via detector inference (forced canonical AMS yolov3 behavior in p003 flow).

### C. Configuration
1. Add `DATA.OTIF_DATASET` in global config.
2. Keep behavior data-driven where possible.

---

## 3) Proposed Plan (that was drafted before coding)

1. Add OTIF data-root config (`OTIF_DATASET`) to global config.
2. Update `p000` caldot/ams source-input path to OTIF dataset root.
3. Refactor `p003` into explicit dataset-family pipelines:
   - CalDOT copy+mask,
   - AMS detect+mask,
   - JNC source-30fps detect.
4. Implement include/exclude mask helpers and clipping.
5. Implement JNC segment mapping + crop/mask/rotate + 4-corner inference + offset remap + NMS + scale-to-1080x720.
6. Validate syntax and basic integrity.

---

## 4) Execution Performed

### 4.1 Configuration updates
- Updated [configs/global.yaml](configs/global.yaml)
  - Added `DATA.OTIF_DATASET: /otif-dataset/dataset`.

### 4.2 `p000` source-path update
- Updated [preprocess/p000_preprocess_dataset.py](preprocess/p000_preprocess_dataset.py)
  - Added config read for `OTIF_DATASET`.
  - Switched caldot/ams preprocessing input root in `process_caldot()` from `SOURCE_DIR` to `OTIF_DATASET`.

### 4.3 Full `p003` implementation replacement
- Replaced [preprocess/p003_preprocess_groundtruth_detection.py](preprocess/p003_preprocess_groundtruth_detection.py) with a new implementation that includes:
  - Dataset routing by family (`caldot*`, `ams*`, `jnc*`).
  - OTIF dataset name normalization helper (`ams -> amsterdam` mapping).
  - Include-mask loader from `data/masks/include/{key}.xml` (box-based include clipping).
  - Exclude-mask loader from `data/masks/exclude/{key}.xml` with fallback to legacy config map.
  - Shared include+exclude application helper (`apply_include_exclude_masks`).

#### Implemented CalDOT flow
- `copy_detection_caldot(...)`:
  - Reads GT JSON from OTIF path: `{OTIF_DATASET}/{dataset_root}/{split}/yolov3-704x480/{id}.json`.
  - Converts to required JSONL output format.
  - Applies include clipping, then exclude filtering.
  - Writes to `003_groundtruth/detection.jsonl`.

#### Implemented AMS flow
- `run_detection_ams(...)`:
  - Runs detector inference per frame from dataset videos.
  - Forces detector loading via canonical `'ams'` in `polyis.models.detector.get_detector('ams', ...)`.
  - Applies include clipping and exclude filtering.
  - Writes to `003_groundtruth/detection.jsonl`.

#### Implemented JNC flow
- `run_detection_jnc(...)`:
  - Resolves original source video under `SOURCE_DIR/{dataset}` (asserts single source video).
  - Computes segment start/end with `num_segments` logic aligned to `p000`.
  - Builds B3D crop/mask/rotation state from `/polyis-data/sources/b3d/annotations.xml`.
  - Per frame in segment (30fps):
    - preprocesses with crop+mask+optional rotation,
    - runs 4 corner crops (`2/3` dimensions),
    - applies offsets,
    - merges via NMS (`polyis.b3d.nms.nms`),
    - scales final boxes to `1080x720`,
    - writes frame-level JSONL.

### 4.4 Validation completed
- Ran syntax compilation check:
  - `python3 -m py_compile preprocess/p003_preprocess_groundtruth_detection.py preprocess/p000_preprocess_dataset.py`
  - Result: passed (no syntax errors).
- Editor import-resolution diagnostics (`cv2`, `numpy`, `torch`) remain environment-indexing issues and not syntax defects.

---

## 5) Important Notes / Assumptions in Current Implementation

1. JNC source mapping currently assumes exactly one source `.mp4` exists in `SOURCE_DIR/{dataset}` and uses split filename index (`teXX`, `trXX`, `vaXX`) as segment index.
2. Include-mask clipping is rectangle-based (from CVAT box in include XML).
3. Exclude-mask filtering uses polygon intersection helper in existing utility.
4. AMS inference in `p003` intentionally uses canonical `ams` detector key to enforce yolov3 behavior regardless of variant suffix.

---

## 6) Remaining Optional Follow-ups

1. Run end-to-end smoke tests for one dataset each (`caldot1-y05`, `ams-y05`, `jnc0`) and verify frame counts / output paths.
2. If needed, add configurable NMS threshold for JNC via config.
3. If source-video mapping for JNC is more complex in production, replace the single-video assertion with explicit manifest mapping.

---

## 7) Files Changed in This Thread

- [configs/global.yaml](configs/global.yaml)
- [preprocess/p000_preprocess_dataset.py](preprocess/p000_preprocess_dataset.py)
- [preprocess/p003_preprocess_groundtruth_detection.py](preprocess/p003_preprocess_groundtruth_detection.py)
