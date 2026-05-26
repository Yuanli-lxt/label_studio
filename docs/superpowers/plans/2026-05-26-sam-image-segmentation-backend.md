# SAM Image Segmentation Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deterministic image segmentation placeholder path with an optional real SAM2/MobileSAM image backend while keeping the existing Label Studio BrushLabels contract testable.

**Architecture:** Keep the current Label Studio prediction response shape unchanged. Add a small backend selector in `services/ml-backend/app.py`: `placeholder` remains the default, while `sam2`, `mobilesam`, or `sam` load optional runtime libraries and convert predicted binary masks to Label Studio Brush RLE.

**Tech Stack:** Python stdlib, NumPy, Pillow, optional `sam2`, optional `segment_anything`/MobileSAM, `label-studio-converter`.

---

### Task 1: Backend Selection Tests

**Files:**
- Modify: `tests/test_image_segmentation_prediction.py`

- [ ] Add a failing test that sets `IMAGE_SEG_BACKEND=mobilesam`, injects fake `segment_anything` modules, and asserts prediction source is `mobilesam-image-segmentation`.
- [ ] Add a failing test that sets `IMAGE_SEG_BACKEND=sam2` without fake modules and asserts the code falls back with explicit `backend_error` metadata instead of pretending placeholder is real.
- [ ] Run: `python3 -m unittest tests/test_image_segmentation_prediction.py -v`

### Task 2: ML Backend Adapter

**Files:**
- Modify: `services/ml-backend/app.py`

- [ ] Add env vars: `IMAGE_SEG_BACKEND`, `IMAGE_SEG_MODEL_TYPE`, `IMAGE_SEG_CHECKPOINT`, `IMAGE_SEG_MODEL_ID`, and `IMAGE_SEG_DEVICE`.
- [ ] Add cached loader helpers for `sam2` and `mobilesam`/`sam`.
- [ ] Add a shared center-box prompt and mask-to-RLE encoder.
- [ ] Update `_image_segmentation()` to use the selected real backend when configured; keep placeholder as default and as explicit fallback.
- [ ] Run: `python3 -m unittest tests/test_image_segmentation_prediction.py -v`

### Task 3: Docs And Verification

**Files:**
- Modify: `README.md`

- [ ] Document `IMAGE_SEG_BACKEND=mobilesam` and `IMAGE_SEG_BACKEND=sam2` runtime configuration.
- [ ] Run focused segmentation tests and Python compile check.
- [ ] Do a targeted diff review for only changed files.
