# PurrSight Phase 1 Code Review & Architecture Diagnostic Report

**Date:** 2026-01-24
**Reviewer:** AI System Architect (IQ 150)
**Scope:** Preprocessing -> Encoder -> Alignment -> Loss

## 1. Executive Summary

The Phase 1 implementation (Multimodal Alignment) is functionally complete but contained critical numerical stability issues and performance bottlenecks that would hinder scaling to Phase 2 (LLM Integration). Specifically, the `ContrastiveLoss` had a flaw in handling masked samples that diluted gradients, and the preprocessing pipeline was inefficient for large-scale data.

## 2. Critical Findings (🔴 High Severity)

### 2.1 Numerical Instability in Contrastive Loss
*   **Location**: `purrsight/alignment/contrastive_loss.py` -> `infonce_loss`
*   **Issue**: The InfoNCE implementation calculated `logits = query @ key.T`. When keys were invalid (zero vectors due to missing modalities), their dot product with queries was 0. In the Softmax denominator $\sum \exp(logits)$, these zero entries contributed $\exp(0) = 1$.
*   **Impact**: For a batch size of $B$, if $K$ samples are missing a modality, the denominator is artificially inflated by $+K$. This systematically lowers the probability assigned to positive pairs, effectively "diluting" the gradient and slowing down convergence.
*   **Fix**: Applied a mask to the logits matrix: `logits[:, ~valid_mask] = -1e9`. This ensures $\exp(-1e9) \approx 0$, correctly removing invalid samples from the denominator.

## 3. Performance Findings (🟡 Medium Severity)

### 3.1 Blocking I/O in Preprocessing
*   **Location**: `purrsight/preprocess/prepre.py` & `purrsight/preprocess/__init__.py`
*   **Issue**: The `prepre.py` script accepted a `--num_workers` argument but executed a sequential loop. Furthermore, `Preprocessor` used blocking `subprocess.run` calls for FFmpeg.
*   **Impact**: GPU utilization during training would be bottlenecked by single-core CPU data preparation.
*   **Fix**: Refactored `prepre.py` to use `ProcessPoolExecutor` for true parallel processing.

### 3.2 Computational Redundancy in Inference
*   **Location**: `purrsight/preprocess/__init__.py` -> `_process_batch`
*   **Issue**: The pipeline forced single-frame images to be replicated 16 times to match the video tensor shape `(16, 3, 224, 224)`.
*   **Impact**: During inference with static images, this increased memory usage and compute by 16x.
*   **Fix**: Introduced `inference_mode` flag. When enabled, the preprocessor skips this expansion for static images (warning: requires model to handle variable batch shapes or pure-image batches).

## 4. Phase 2 Readiness (🔵 Forward Looking)

### 4.1 Modality Collapse & Zero Vectors
*   **Issue**: LLMs are sensitive to "all-zero" embedding inputs, which can lead to undefined behavior or hallucination.
*   **Solution**: Designed a `MultimodalProjector` that detects zero vectors and maps them to a learnable `<MISSING>` token embedding.

### 4.2 Feature Granularity
*   **Issue**: A single 512-dim vector is insufficient to capture the temporal nuances of cat behaviors (e.g., "ears flattening" vs "ears flat").
*   **Solution**: Proposed "Soft Prompting" in the Projector, mapping each modality feature to $N$ tokens to preserve more information.

## 5. Refactoring Summary

| Module | Change | Status |
| :--- | :--- | :--- |
| `contrastive_loss.py` | Added `-inf` masking for invalid keys in InfoNCE | ✅ Fixed |
| `prepre.py` | Implemented `ProcessPoolExecutor` for parallelism | ✅ Optimized |
| `preprocess/__init__.py` | Added `inference_mode` to skip 16x frame copy | ✅ Optimized |
| `LLM/projectors.py` | Created `MultimodalProjector` with Zero-Vector Fail-Safe | ✅ Implemented |
| `README.md` | Updated with Phase 2 Architecture & Fixes | ✅ Updated |

---

**Next Steps:**
Proceed with Phase 2: Initialize `MatFormer-OLMo-0.5B` and begin training the `MultimodalProjector` using the fixed alignment backbone.
