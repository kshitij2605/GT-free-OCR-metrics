# Position-Invariant Document Image Similarity: Research Synthesis

> **Context:** Comparing an original document page image against a deterministically reconstructed image (re-rendered from OCR bounding boxes + text). Problem: text elements shift slightly due to font/size differences, characters may wrap to the next line, and global metrics (SSIM, LPIPS, CLIP) punish any spatial mismatch even when text content is correct.
>
> **Goal:** Find metrics that are tolerant of small positional shifts while remaining sensitive to character-level OCR errors.

---

## The Core Tension

The fundamental challenge is that the two desiderata point in opposite directions:

- **Shift tolerance** requires the metric to be *insensitive* to local spatial displacements.
- **Character sensitivity** requires the metric to be *sensitive* to subtle content differences, even when those differences occupy a tiny spatial region.

No single metric satisfies both perfectly. The practical answer is a hybrid: use a shift-tolerant visual metric alongside a content-aware text metric, and combine them at the score level.

---

## Current Baseline (This Project)

| Metric | Mechanism | Shift Sensitivity | Character Sensitivity |
|--------|-----------|-------------------|----------------------|
| SSIM (global) | Structural correlation in sliding Gaussian window | High (any shift degrades score) | Low (text < 1% of pixels) |
| LPIPS (global) | Deep feature distance on full image | Moderate-high | Low-moderate |
| CLIP cosine (global) | Single CLS-token embedding cosine | Low-moderate | Low (semantic, not lexical) |

Known weakness: CLIP drops dramatically on images with complex layout even when OCR text is correct.

---

## Method 1: Patch-Based CLIP Matching with Flexible Pooling

### How It Works

Instead of passing the full image through CLIP and comparing the single CLS token, divide each image into a grid of overlapping patches (e.g., 64×64 or 128×128 pixels), extract the CLIP ViT patch tokens for each patch, and then match patches across the two images with a flexible matching strategy:

- **Best-match pooling**: For each patch in the reconstructed image, find the most similar patch (highest cosine similarity) in the original image, regardless of position. Take the mean of all best-match scores.
- **Hungarian matching**: Solve the optimal bipartite assignment between patch sets (scipy.optimize.linear_sum_assignment). Each patch is matched to exactly one partner; minimizes total dissimilarity.
- **Sliding-window comparison**: For each patch position, compare against a small neighborhood of positions (e.g., ±2 patches = ±32 pixels at 16px stride). Take the max similarity within the neighborhood.

CLIP's ViT-B/32 already divides images into 14×14 non-overlapping patches of 32×32 pixels internally. These intermediate patch embeddings (before global pooling) can be extracted directly from the model's forward hook.

### Why It Handles Positional Shifts

Best-match pooling and Hungarian matching are inherently translation-tolerant: a patch that shifts 20 pixels left still finds its best match at the adjacent patch location. The matching is content-driven, not position-driven.

### Character-Level Sensitivity

CLIP patch tokens carry semantic and visual texture information. At the 32×32 pixel granularity of ViT-B/32 patches, a single character typically spans 1–3 patches. A wrong character will produce a different visual texture in those patches, so the best-match score for those patches will be lower than 1.0. However, CLIP's semantic embedding space compresses some visual detail, so very similar characters (e.g., '0' vs 'O') may not be well-distinguished.

**Limitation:** CLIP was trained on natural images and captions, not on document character images. Its patch features are more sensitive to semantic regions than to glyph-level detail.

### Practical Implementation

```python
# Example: extract patch tokens from CLIP ViT
import torch, open_clip
from scipy.optimize import linear_sum_assignment

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

def get_patch_features(img_tensor):
    """Returns (N_patches, D) patch token matrix from CLIP ViT."""
    with torch.no_grad():
        out = model.visual.trunk.patch_embed(img_tensor)  # (1, N, D)
    return out.squeeze(0)  # (N, D)

def patch_clip_similarity(img_a, img_b, strategy='best_match'):
    feats_a = get_patch_features(preprocess(img_a).unsqueeze(0))  # (196, 512)
    feats_b = get_patch_features(preprocess(img_b).unsqueeze(0))
    # Cosine similarity matrix (196 x 196)
    sim = (feats_a @ feats_b.T) / (feats_a.norm(dim=1, keepdim=True) * feats_b.norm(dim=1).unsqueeze(0))
    if strategy == 'best_match':
        return sim.max(dim=1).values.mean().item()
    elif strategy == 'hungarian':
        cost = (1 - sim).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        return 1 - cost[row_ind, col_ind].mean()
```

**Libraries:** `open_clip`, `scipy` (already project dependencies or easy to add).

**Complexity:** O(N²) where N = number of patches (196 for ViT-B/32 at 224×224). For higher-resolution document images (1024×1024), the image must be tiled first. Hungarian adds O(N³) overhead — use best-match for speed.

### Tradeoffs vs Current Metrics

| Aspect | CLIP (global) | Patch-CLIP (best-match) |
|--------|---------------|-------------------------|
| Shift tolerance | Poor | Good |
| Character sensitivity | Poor | Moderate |
| Semantic understanding | High | Moderate |
| Computational cost | Low | Moderate (O(N²)) |
| Implementation effort | Already done | Moderate |

**Verdict:** A practical, incremental upgrade over global CLIP. Recommended as the first additional metric to implement.

---

## Method 2: Earth Mover's Distance / Optimal Transport on Visual Features

### How It Works

Treat each image as a **distribution of local visual features** embedded in a metric space. The Earth Mover's Distance (EMD), also known as the Wasserstein-1 distance, measures the minimum "work" needed to transform one distribution into the other — where work = mass × distance in feature space.

**Procedure:**
1. Divide each image into a grid of non-overlapping patches (e.g., 32×32 pixels).
2. Compute a feature vector for each patch (DINOv2 tokens, CLIP patch tokens, HOG, or simple CNN features).
3. Treat each patch's feature vector as a "supply" unit in source distribution (original image) and "demand" in target distribution (reconstructed image).
4. Compute the EMD between the two distributions using the POT (Python Optimal Transport) library.

The EMD naturally finds the cheapest way to match every patch in image A to some patch in image B. Patches that shift slightly will incur a small transport cost proportional to the shift distance in feature space (which correlates with spatial distance when features are position-aware). This is fundamentally shift-tolerant.

**Formal definition:**

```
EMD(P, Q) = min_{T ≥ 0} Σ_{i,j} T_ij · d(f_i, f_j)
            subject to: Σ_j T_ij = p_i  (supply)
                        Σ_i T_ij = q_j  (demand)
```

where f_i, f_j are feature vectors of patches, and d(·,·) is Euclidean or cosine distance.

### Why It Handles Positional Shifts

Shifts are inherently handled because EMD allows "transporting" patch features across positions. A patch that moves 20 pixels right in the reconstruction will simply be matched to the adjacent patch in the original at a small transport cost. If the text content is identical, the feature vectors will be similar and the transport cost low even after a shift.

The Rubner et al. (2000) IJCV paper specifically demonstrated that EMD on visual feature distributions captures perceptual similarity better than histogram matching, precisely because it accounts for the geometry of the feature space.

### Character-Level Sensitivity

Character errors affect the feature vectors of the patches containing that character. These patches will be costlier to transport to any matching patch in the other image, because no patch with those features exists there. The metric is sensitive to this if the features are fine-grained enough (DINOv2 or high-resolution HOG features work better than coarse CNN features).

### Practical Implementation

```python
import ot  # POT library: pip install POT
import numpy as np

def emd_image_similarity(img_a, img_b, patch_size=32, feature_fn=None):
    """
    Compute 1 - normalized EMD between feature distributions of two images.
    feature_fn: callable (patch_img) -> 1D numpy array. Default: flattened HOG.
    """
    patches_a = extract_patches(img_a, patch_size)  # list of PIL images
    patches_b = extract_patches(img_b, patch_size)

    feats_a = np.stack([feature_fn(p) for p in patches_a])  # (N, D)
    feats_b = np.stack([feature_fn(p) for p in patches_b])  # (M, D)

    # Uniform weights (all patches equally important)
    a = np.ones(len(feats_a)) / len(feats_a)
    b = np.ones(len(feats_b)) / len(feats_b)

    # Cost matrix: pairwise Euclidean distance in feature space
    M = ot.dist(feats_a, feats_b, metric='euclidean')
    M /= M.max()  # normalize

    emd_val = ot.emd2(a, b, M)  # scalar transport cost
    return 1.0 - emd_val  # higher = more similar
```

**Libraries:** `POT` (pip install POT), requires C++ compiler for EMD solver (or use Sinkhorn approximation for pure Python).

**Complexity:** O(N³ log N) exact EMD where N = number of patches. For a 1024×1024 image with 32×32 patches: N = 1024, making exact EMD computationally expensive. Use Sinkhorn regularization (`ot.sinkhorn2`) for faster approximation: O(N²/ε²).

**For production use:** With 64×64 patches, N = 256, which is tractable (~1–5 seconds for exact EMD). With Sinkhorn (reg=0.05), sub-second.

### Variants Worth Noting

- **Sliced Wasserstein Distance (SWD):** Projects high-dimensional features onto 1D lines, computes 1D EMD (trivially solved by sorting), and averages over many random projections. Scales as O(N·K) where K = projections. Available in POT as `ot.sliced_wasserstein_distance()`. Excellent speed-accuracy tradeoff.
- **WMD (Word Mover's Distance):** Same concept applied to text tokens with word2vec embeddings — directly applicable to character-level comparison of OCR output text.

### Tradeoffs vs Current Metrics

| Aspect | Global SSIM | EMD on Visual Features |
|--------|-------------|------------------------|
| Shift tolerance | Very poor | Excellent |
| Character sensitivity | Poor | Good (with DINOv2 features) |
| Theoretical grounding | Moderate | Strong (optimal transport theory) |
| Computational cost | Low | High (exact), Moderate (Sinkhorn) |
| Implementation effort | Done | Moderate-high |

**Verdict:** Theoretically the most principled approach for this problem. The Sliced Wasserstein variant is particularly attractive for its speed. Recommended as the second metric to investigate.

---

## Method 3: DINOv2 Dense Patch Features for Local Matching

### How It Works

DINOv2 (Oquab et al., 2023) is a self-supervised Vision Transformer trained with DINO + iBOT distillation on a curated dataset of 142M images. Unlike CLIP, DINOv2 was not trained with text supervision, which means its features capture fine-grained visual structure rather than abstract semantics.

Key property: DINOv2 produces **one embedding per 14×14-pixel patch** of the input image. These dense patch tokens can be used directly for local similarity matching.

**Procedure:**
1. Resize both images to the same resolution (e.g., 1120×1120 for ViT-L/14: 80×80 patch grid = 6400 patches).
2. Extract patch token embeddings from DINOv2 (shape: (H/14, W/14, D)).
3. For each patch in the reconstruction, find its nearest neighbor in the original image's patch grid by cosine similarity.
4. Aggregate matching scores: mean, or weighted by patch "importance" (e.g., higher weight for patches with more ink).

**Ink-weighted aggregation** is especially useful for documents: background patches (white) are unimportant and should not dominate the score. Weight patches by the fraction of dark pixels (ink density) in the original patch.

### Why It Handles Positional Shifts

The nearest-neighbor matching step is inherently shift-tolerant: a patch displaced by one patch-width (14 pixels) will still find its best match at the adjacent position. The matching radius can be explicitly bounded (e.g., search within ±3 patches) to avoid false matches across distant regions.

DINOv2's attention mechanism also tends to produce features that are somewhat invariant to small local deformations — empirically observed in the paper's feature matching demonstrations across pose variations and style changes.

### Character-Level Sensitivity

DINOv2 features have demonstrated strong performance at both instance-level and category-level retrieval (DINOv2 paper, TMLR 2024: +34% mAP on Oxford-Hard vs. prior SSL methods). At 14-pixel patch granularity, individual characters (typically 8–30 pixels tall in document images at typical DPI) span 1–3 patches. A wrong character creates a different ink pattern, producing different DINOv2 features.

**Key advantage over CLIP:** DINOv2 features are lower-level and more texture/structure sensitive. The absence of text supervision means 'p' and 'q' will produce more distinct features than they would in CLIP (where both are "letters" semantically).

**Caveat:** DINOv2 is still a ViT with learned attention patterns. Very fine-grained character differences (e.g., '1' vs 'l', 'O' vs '0') may still be poorly distinguished at 14-pixel resolution.

### Practical Implementation

```python
from transformers import AutoImageProcessor, AutoModel
import torch
import torch.nn.functional as F

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')

def get_dinov2_patches(pil_img):
    """Returns (H_patches, W_patches, D) patch token grid."""
    inputs = processor(images=pil_img, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # last_hidden_state: (1, num_patches+1, D) — first token is CLS
    patch_tokens = outputs.last_hidden_state[0, 1:, :]  # (N, D)
    h = w = int(patch_tokens.shape[0] ** 0.5)
    return patch_tokens.reshape(h, w, -1)  # (H, W, D)

def dino_patch_similarity(img_a, img_b, search_radius=2):
    """
    For each patch in img_b (reconstruction), find best-match in img_a (original).
    Returns mean cosine similarity across patches.
    """
    feats_a = get_dinov2_patches(img_a)  # (H, W, D)
    feats_b = get_dinov2_patches(img_b)
    H, W, D = feats_a.shape

    scores = []
    for i in range(H):
        for j in range(W):
            patch_b = feats_b[i, j]  # (D,)
            # Search neighborhood in img_a
            i_lo, i_hi = max(0, i - search_radius), min(H, i + search_radius + 1)
            j_lo, j_hi = max(0, j - search_radius), min(W, j + search_radius + 1)
            neighbors = feats_a[i_lo:i_hi, j_lo:j_hi].reshape(-1, D)  # (K, D)
            cosines = F.cosine_similarity(patch_b.unsqueeze(0), neighbors)
            scores.append(cosines.max().item())

    return float(torch.tensor(scores).mean())
```

**Libraries:** `transformers`, `torch`. DINOv2-large: ~300M parameters, ~1.2GB on GPU. DINOv2-base: ~86M, much faster.

**Complexity:** O(H·W·(2r+1)²·D) per image pair, where r = search_radius. For 560×560 input with ViT-B/14: 40×40 patches, D=768, r=2: fully tractable in < 1 second on GPU.

### Tradeoffs vs Current Metrics

| Aspect | CLIP cosine | DINOv2 patch matching |
|--------|-------------|----------------------|
| Shift tolerance | Poor | Good (with search radius) |
| Character sensitivity | Low | Moderate-good |
| Feature quality (documents) | Moderate | Good |
| Model size | 87M (ViT-B/32) | 307M (ViT-L/14) |
| GPU memory | ~350MB | ~1.2GB |
| Implementation effort | Done | Moderate |

**Verdict:** Highest feature quality among purely visual methods. Recommended alongside Patch-CLIP and EMD for comprehensive evaluation.

---

## Method 4: Sliding-Window SSIM and Multi-Scale SSIM (MS-SSIM)

### How It Works

**Sliding-window SSIM with shift tolerance:**

Standard SSIM already uses a sliding 11×11 Gaussian window internally, but the map is computed at pixel-precise alignment. The shift-tolerant variant takes the SSIM quality map and, for each position, takes the maximum SSIM score within a small spatial neighborhood (e.g., ±5 pixels). This is sometimes called "shift-max SSIM."

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np
from scipy.ndimage import maximum_filter

def shift_tolerant_ssim(img_a, img_b, shift_pixels=5):
    score_map = ssim(img_a, img_b, full=True)[1]  # Returns (score, map)
    # Max-pool the map over a neighborhood of size (2*shift_pixels+1)
    pooled_map = maximum_filter(score_map, size=2 * shift_pixels + 1)
    return float(pooled_map.mean())
```

**MS-SSIM (Multi-Scale SSIM):**

Wang et al. (2003) extend SSIM by computing it at multiple downsampled scales (typically 5 levels, each downsampled by 2×). Luminance comparison is used only at the finest scale; contrast and structure are combined across all scales. This makes MS-SSIM more robust to fine-vs-coarse content trade-offs.

The scikit-image implementation (`skimage.metrics.structural_similarity` with `gaussian_weights=True`) and the PIQ library (`piq.ms_ssim`) both provide MS-SSIM.

**Ghildyal & Liu (ECCV 2022) — ST-LPIPS:**

The "Shift-Tolerant LPIPS" (ST-LPIPS) paper is the most directly relevant reference for this problem. The authors found that standard LPIPS scores degrade significantly for images with as little as 1-pixel misalignment (imperceptible to humans). They systematically studied anti-aliasing filters, pooling strategies, and padding choices in the LPIPS backbone (VGG/AlexNet), finding that:

- **Anti-aliasing pooling** (blur before striding) dramatically reduces shift sensitivity.
- **Blurpool** (Zhang et al., 2019) — a specific anti-aliasing layer — is the most effective modification.

The ST-LPIPS metric is available at: https://github.com/abhijay9/ShiftTolerant-LPIPS
```bash
pip install git+https://github.com/abhijay9/ShiftTolerant-LPIPS.git
```

For document reconstruction, shifts of 5–50 pixels are common (font size variations, text wrapping). ST-LPIPS was designed for 1-pixel tolerance but the anti-aliasing principle generalizes to larger shifts when combined with larger pooling regions.

### Why It Handles Positional Shifts

Max-pooling in SSIM directly absorbs shifts up to the neighborhood size. MS-SSIM's multi-scale pyramid makes the metric robust to content at different spatial frequencies — coarse shifts are captured at coarser scales where the local structural correlation is less position-sensitive.

### Character-Level Sensitivity

SSIM computes local structure correlation in 11×11 windows. At 96 DPI with 12pt font, characters are roughly 16×16 pixels, so a single character substitution affects 2–4 SSIM windows. However, because SSIM aggregates over the entire image, and text occupies <10% of a typical document image, the impact on the global score is small.

**Key limitation:** Even shift-tolerant SSIM remains a pixel-level metric. It cannot distinguish *what* changed — visual noise vs. a character error. For character sensitivity, a text-based component (Method 4) is necessary.

### Practical Implementation

```python
# MS-SSIM via PIQ library (pip install piq)
import piq
import torch

def ms_ssim_similarity(img_tensor_a, img_tensor_b):
    """img tensors: (1, C, H, W) in [0, 1]"""
    return piq.ms_ssim(img_tensor_a, img_tensor_b).item()

# ST-LPIPS (after installing ShiftTolerant-LPIPS)
import stlpips
st_lpips_fn = stlpips.LPIPS(net='vgg', variant='shift_tolerant')
dist = st_lpips_fn(img_tensor_a, img_tensor_b)  # lower = more similar
score = 1.0 - dist.item()
```

**Complexity:** MS-SSIM: O(H·W·L) where L = number of scales (5). Equivalent to 5× standard SSIM — cheap. ST-LPIPS: same as LPIPS (one VGG forward pass per image).

### Tradeoffs vs Current Metrics

| Aspect | SSIM | MS-SSIM | ST-LPIPS |
|--------|------|---------|---------|
| Shift tolerance | None | Modest | Good (1-5px) |
| Character sensitivity | Low | Low | Low-moderate |
| Computational cost | Low | Low | Moderate (VGG) |
| Implementation effort | Done | Trivial | Easy (pip install) |

**Verdict:** MS-SSIM is a trivial upgrade — just replace `ssim()` with `ms_ssim()`. ST-LPIPS addresses the specific problem of shift-sensitive LPIPS found in this project. Both are low-hanging fruit worth implementing.

---

## Method 6: Feature Pyramid Matching / Spatial Pyramid Matching (SPM)

### How It Works

Lazebnik, Schmid & Ponce (CVPR 2006) introduced Spatial Pyramid Matching as an extension of the "Bag of Visual Words" model. The key idea: compare histograms of local visual features (SIFT descriptors) at multiple spatial granularities simultaneously.

**Procedure:**
1. Extract local features (e.g., SIFT, HOG, or DINOv2 patch tokens) from both images.
2. Quantize features into a visual vocabulary (K-means codebook of size V).
3. For each level l of a pyramid (l = 0, 1, 2 → 1, 4, 16 spatial bins):
   - Divide image into 2^l × 2^l sub-regions.
   - Compute a histogram of visual words in each sub-region.
   - Compute histogram intersection between the two images for each sub-region.
4. Combine across levels with decreasing weights (finer levels weighted more):
   `SPM_similarity = Σ_l (1/2^(L-l)) × Σ_{bin} min(hist_a[bin], hist_b[bin])`

The histogram intersection kernel is a form of set overlap that is inherently tolerant of small positional shifts within each bin: a feature that moves within a bin is still counted in the same histogram.

### Why It Handles Positional Shifts

Within each pyramid level, features are compared as bags (unordered sets within a spatial bin). A feature that shifts slightly within a bin is still captured in that bin's histogram. Shifts that cross a bin boundary are partially handled by the multi-scale structure: at coarser levels, many nearby positions collapse into the same bin.

The coarsest level (l=0, one global bin) is completely shift-invariant. Finer levels add spatial specificity but reduce tolerance. The weighted combination balances the two.

### Character-Level Sensitivity

SPM with fine-grained visual words (large codebook, e.g., V=256–1024) can capture character-level glyph patterns. A different character produces a different set of SIFT/HOG descriptors, which map to different visual words and produce a different histogram. The histogram intersection score will be lower when characters differ.

**Caveat:** The codebook must be trained on document character images to be effective for character distinction. Using a generic codebook trained on natural images will conflate many character pairs.

### Practical Implementation

```python
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import hog
import numpy as np

def compute_spm_features(img, codebook, n_levels=3):
    """Compute spatial pyramid feature vector."""
    descriptors, positions = extract_dense_hog_patches(img)  # HOG per 16×16 patch
    visual_words = codebook.predict(descriptors)  # quantize to codebook

    feature_vector = []
    H, W = img.shape[:2]
    for l in range(n_levels):
        n_bins = 2 ** l
        bin_h, bin_w = H // n_bins, W // n_bins
        for i in range(n_bins):
            for j in range(n_bins):
                # patches in this spatial bin
                in_bin = (
                    (positions[:, 0] >= i * bin_h) & (positions[:, 0] < (i+1) * bin_h) &
                    (positions[:, 1] >= j * bin_w) & (positions[:, 1] < (j+1) * bin_w)
                )
                hist = np.bincount(visual_words[in_bin], minlength=len(codebook.cluster_centers_))
                feature_vector.append(hist)
    return np.concatenate(feature_vector)

def spm_similarity(img_a, img_b, codebook, n_levels=3):
    feat_a = compute_spm_features(img_a, codebook, n_levels)
    feat_b = compute_spm_features(img_b, codebook, n_levels)
    # Histogram intersection
    return np.minimum(feat_a, feat_b).sum() / feat_a.sum()
```

**Libraries:** scikit-learn (K-means), scikit-image (HOG), numpy.

**Complexity:** Feature extraction: O(H·W). K-means prediction: O(N·V). Pyramid construction: O(N·L). Total: moderate.

**Codebook training requirement:** Requires training a codebook on document image patches beforehand. This is a one-time offline cost.

### Tradeoffs vs Current Metrics

| Aspect | SSIM | SPM |
|--------|------|-----|
| Shift tolerance | None | Good (within-bin) |
| Character sensitivity | Low | Moderate (codebook quality dependent) |
| Training required | No | Yes (codebook) |
| Implementation effort | Done | High |
| Interpretability | High | Low |

**Verdict:** SPM is well-studied and theoretically motivated, but the requirement for a domain-specific codebook adds offline complexity. The max-pooling variant (ScSPM with SIFT sparse coding) outperforms histogram-based SPM per the literature. For this project, Method 1 (Patch-CLIP) and Method 3 (DINOv2) achieve similar benefits with lower setup cost. SPM is a third-tier option.

---

## Method 7: Image Registration / Alignment Before Comparison

### How It Works

Rather than making the metric shift-tolerant, this approach makes the images *aligned* before applying any standard metric. If the two images can be brought into pixel-level alignment, SSIM, LPIPS, and CLIP will work correctly.

**Two approaches:**

**A. Feature-based registration (ORB/SIFT + RANSAC + homography):**
1. Detect keypoints in both images (ORB is free; SIFT requires opencv-contrib).
2. Describe keypoints and match with brute-force matcher (Hamming distance for ORB, L2 for SIFT).
3. Filter outliers with RANSAC.
4. Estimate affine or projective transformation matrix.
5. Warp the reconstructed image to align with the original.
6. Apply standard metrics to the aligned pair.

```python
import cv2
import numpy as np

def align_images(img_orig_gray, img_recon_gray, method='ORB'):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img_orig_gray, None)
    kp2, des2 = orb.detectAndCompute(img_recon_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:200]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    h, w = img_orig_gray.shape
    aligned_recon = cv2.warpPerspective(img_recon_array, M, (w, h))
    return aligned_recon
```

**B. Dense optical flow registration (ECC or Lucas-Kanade):**

```python
# ECC (Enhanced Correlation Coefficient) — handles affine/translation/rotation
warp_mode = cv2.MOTION_AFFINE
warp_matrix = np.eye(2, 3, dtype=np.float32)
cc, warp_matrix = cv2.findTransformECC(
    img_orig_gray.astype(np.float32),
    img_recon_gray.astype(np.float32),
    warp_matrix,
    warp_mode
)
aligned_recon = cv2.warpAffine(img_recon_array, warp_matrix, (w, h))
```

### Why It Handles Positional Shifts

By construction: after registration, spatial shifts are eliminated. SSIM, LPIPS, and CLIP then compare semantically corresponding regions.

### Character-Level Sensitivity

Once registered, any remaining visual differences are due to content (character errors, missing text, font differences), not position. All downstream metrics become content-sensitive. This is the cleanest solution conceptually.

**Key limitation for document OCR reconstruction:** The registration assumes there exists a rigid (or locally affine) transformation between the two images. In practice, the reconstructed image has:
- Different font metrics (character widths differ from original).
- Line reflow (characters wrap differently).
- Missing or extra elements.

These are not rigid transformations. ORB/SIFT keypoint matching will fail when there are no reliable shared keypoints (e.g., if the reconstruction uses different fonts, characters look different). The homography will be poor or RANSAC will find no inliers.

**ECC convergence issue:** ECC assumes small initial displacement. If the reconstruction deviates significantly from the original (which it often does for complex documents), ECC may not converge.

**When registration works well:** Documents where the reconstruction closely matches the original (same font, same size, small positional offset only). For this project's Iteration 6+ DocumentAnalyzer output, registration may be viable for simple documents.

### Tradeoffs vs Current Metrics

| Aspect | No registration | Feature-based registration |
|--------|-----------------|---------------------------|
| Shift tolerance | None | Complete (after alignment) |
| Character sensitivity | Metric-dependent | Metric-dependent (but now meaningful) |
| Robustness | High (always runs) | Low-moderate (fails on poor reconstructions) |
| Implementation effort | Done | Moderate (OpenCV) |
| Failure mode | Penalizes all shifts | May produce garbage alignment |

**Verdict:** Registration is compelling in theory but fragile in practice for this use case because document reconstructions differ substantially from originals (not just spatially). It is best applied as an optional pre-processing step when the reconstruction quality is known to be high (many bboxes extracted, text elements match). Always implement a fallback: if RANSAC finds fewer than 10 inliers, skip registration and use the unaligned metrics.

---

## Comparative Summary Table

| Method | Shift Tolerance | Character Sensitivity | Complexity | Priority |
|--------|-----------------|----------------------|------------|----------|
| **1. Patch-CLIP (best-match)** | Good | Moderate | O(N²) | High |
| **2. EMD / Sliced Wasserstein** | Excellent | Good | O(N²)–O(N³) | High |
| **3. DINOv2 patch matching** | Good | Moderate-good | O(H·W·r²) | Medium-high |
| **4a. MS-SSIM** | Modest | Low | O(H·W·L) | Trivial (do now) |
| **4b. ST-LPIPS** | Good (1-5px) | Low | O(VGG pass) | Easy |
| **5. SPM** | Good (within-bin) | Moderate | O(H·W·V) | Low |
| **6. Image registration** | Complete (when works) | Metric-dependent | O(H·W·iter) | Low |

---

## Recommendations for This Project

### Immediate (Low Effort, High Value)

**1. Replace SSIM with MS-SSIM**

One-line change. MS-SSIM handles multi-scale content better and is standard in image quality assessment. Use `piq.ms_ssim()`.

**2. Replace LPIPS with ST-LPIPS**

Install `ShiftTolerant-LPIPS` and swap the LPIPS call. Directly addresses the shift sensitivity found in the project's current metrics. The shifts from font metric differences are in the 1–30 pixel range — exactly the regime ST-LPIPS was designed for.

### Near-term (Moderate Effort, High Value)

**3. Add Patch-CLIP similarity (best-match pooling)**

Extract CLIP patch tokens from both images, compute per-patch cosine similarity, take mean of best-match scores. Does not require bboxes. Addresses the global CLIP weakness identified in iteration 3 (CLIP dropped on complex layouts). Implementation: ~50 lines.

**4. Add DINOv2 dense patch matching with ink-weighted aggregation**

DINOv2 features are superior to CLIP for visual structure matching. Use ink-density weighting to focus on text-bearing regions. Implementation: ~80 lines. Requires `transformers` library (likely already available).

### Near-term (Moderate Effort, High Value — continued)

**5. Add Sliced Wasserstein Distance as EMD approximation**

Using DINOv2 patch features as the distribution (one feature vector per patch, uniform weights), compute Sliced Wasserstein via `ot.sliced_wasserstein_distance()`. This is the most principled global similarity score for this task. Add as an additional correlation experiment once other metrics are stable.

### Not Recommended for This Project

- **SPM with codebook training** — too much setup complexity relative to benefit when Patch-CLIP and DINOv2 are available.
- **Feature-based image registration (ORB/SIFT)** — too fragile for documents with font and line-reflow differences; RANSAC rarely finds enough inliers when reconstruction uses different font metrics.
- **ECC optical flow registration** — same fragility concern; ECC assumes small initial misalignment, which is often not the case here.

---

## Implementation Order

```
Phase 1 (1 day):
  - Replace skimage.metrics.ssim with piq.ms_ssim in visual_reconstruction.py
  - Install and substitute ST-LPIPS for LPIPS in visual_reconstruction.py
  - Run experiments, compare correlation with OmniDocBench

Phase 2 (2–3 days):
  - Implement Patch-CLIP similarity (add to clip_compare/ or as new metric)
  - Implement DINOv2 patch matching (new metric module: metrics/dino_patch/)
  - Run ablation: which metrics correlate best with OmniDocBench edit distance?

Phase 3 (research):
  - Implement Sliced Wasserstein on DINOv2 features
  - Evaluate TokenCLIP OT matching (see Section: Recent Papers 2023–2025)
  - Ablation study: each metric's contribution to overall correlation
  - Write up methodology for paper
```

---

## Key Papers to Read

| Paper | Why Relevant |
|-------|--------------|
| Ghildyal & Liu, ECCV 2022 — "Shift-Tolerant Perceptual Similarity Metric" | Directly addresses this project's LPIPS shift sensitivity; provides drop-in replacement |
| Rubner et al., IJCV 2000 — "The Earth Mover's Distance as a Metric for Image Retrieval" | Foundational EMD image similarity paper; describes visual feature distribution approach |
| Oquab et al., TMLR 2024 — "DINOv2: Learning Robust Visual Features without Supervision" | Explains dense patch tokens and local matching properties |
| Wang et al., Asilomar 2003 — "Multi-Scale Structural Similarity for Image Quality Assessment" | Original MS-SSIM paper; motivated by human contrast sensitivity function |
| Lazebnik et al., CVPR 2006 — "Beyond Bags of Features: Spatial Pyramid Matching" | SPM conceptual foundation; useful if codebook approach is pursued |
| Flamary et al., JMLR 2021 — "POT: Python Optimal Transport" | POT library paper; practical guide to EMD/Sinkhorn implementation |
| Kusner et al., ICML 2015 — "From Word Embeddings to Document Distances" | WMD — exactly the same EMD principle applied to text tokens; shows text+visual analogy |

---

## Recent Papers (2023–2025): Latest Advances

Research conducted April 2026. Papers directly relevant to shift-invariant, character-sensitive document image similarity.

### Shift-Tolerant Perceptual Metrics

**ST-LPIPS — "Shift-Tolerant Perceptual Similarity Metric"**
Ghildyal & Liu. ECCV 2022. https://arxiv.org/abs/2207.13686

Systematically studies anti-aliasing, pooling strategies, and padding in the LPIPS backbone (VGG/AlexNet). Finds that standard LPIPS degrades for as little as 1-pixel misalignment — imperceptible to humans. The fix: Blurpool anti-aliasing layers before strided operations. The resulting ST-LPIPS is a drop-in replacement. For document reconstruction, shifts of 5–30 pixels (font metric differences, line-wrap) are in exactly the range this paper was designed to address.
- Shift tolerance: **High** — core contribution.
- Character sensitivity: Moderate (inherits LPIPS backbone; not text-trained).
- Implementation: `pip install shift-tolerant-lpips`, 2-line code change.

**LipSim — "A Provably Robust Perceptual Similarity Metric"**
Ghazanfari et al. ICLR 2024. https://arxiv.org/abs/2310.18274

Trains a 1-Lipschitz student network mimicking DreamSim. Under adversarial attack, DreamSim collapses to near-random; LipSim holds. The Lipschitz property also means scores are calibrated and stable — useful for setting quality thresholds. Less directly relevant than ST-LPIPS but interesting if score stability is required.

**SUSS — "Structured Uncertainty Similarity Score"**
Seidler et al. arXiv Dec 2024. https://arxiv.org/abs/2512.03701

Models each image via structured multivariate Gaussian distributions (Superimposed Uncertainty Pixel Normals). Similarity = weighted Mahalanobis distance. Key output: a **pixelwise interpretable difference map** showing only perceivable differences. For our use case, this map could localize exactly which bounding-box regions have OCR errors. No confirmed public implementation yet — watch for release.

**SPIPS — "Scene Perceived Image Perceptual Score"**
arXiv April 2025. https://arxiv.org/abs/2504.17234

Combines PSNR/SSIM/MS-SSIM spatial quality maps, deep CNN features (low-level + semantic), and a learned MLP. The **per-region spatial quality map** output enables bbox-level quality diagnostics: identify which document regions are reconstructed poorly. Useful complement to DINOv2 patch matching.

---

### Dense ViT / DINO Patch Features (2023–2025)

**DINOv3 — "High-Resolution Dense Features"**
Meta. 2025. https://arxiv.org/abs/2508.10104

Fixes a known degradation in DINOv2's patch feature quality in large (ViT-L) models using **Gram anchoring** — stabilizing the patch-feature similarity structure during training. Also adds high-resolution training for finer spatial granularity. For character-level analysis where DINOv2-ViT-L is used, DINOv3 should be preferred as it directly fixes the patch degradation issue.
- Shift tolerance: **High** (same nearest-neighbor matching approach as DINOv2).
- Character sensitivity: **High** — improved patch feature quality.
- Implementation: Drop-in DINOv2 replacement via HuggingFace.

**dino.txt — "DINOv2 Meets Text"**
Jose et al. CVPR 2025. https://arxiv.org/abs/2412.16334

Trains a text encoder aligned to frozen DINOv2 via LiT-style training. Enables **text-grounded spatial localization**: embed a character string, get a dense cosine similarity map over image patches showing where that text appears. For reference-free OCR evaluation: embed the OCR-extracted text, compute correspondence with original image patches, and measure match quality. Missing or wrong characters fail to activate their expected regions.
- Shift tolerance: **High** — patch-level dense localization.
- Character sensitivity: **High** — text-grounded, character queries localize individual character positions.
- Implementation: Medium. Requires fine-tuned text encoder (weights not yet widely available).

**Perception Encoder (PE)**
Bolya et al. (Meta FAIR). NeurIPS 2025 Oral. https://arxiv.org/abs/2504.13181

Discovers that intermediate layers of CLIP-trained ViTs contain dense spatial embeddings competitive with DINOv2 on spatial tasks. PE-spatial adds a spatial alignment head on top of CLIP's ViT. Achieves 94.6 on DocVQA — the same encoder understands document content AND provides dense patch-level features. This makes PE-spatial arguably the best single backbone for this project: language grounding (from CLIP training) + fine-grained spatial features (from the alignment head).
- Shift tolerance: **High**.
- Character sensitivity: **High** — document-trained, validated on DocVQA.
- Implementation: Low-Medium (HuggingFace + spatial head).

---

### Optimal Transport Applied to Visual Features (2024–2025)

**TokenCLIP — "OT-based CLIP Patch-to-Text Alignment"**
arXiv 2025. https://arxiv.org/abs/2510.21171

Reformulates CLIP patch-token alignment as an **optimal transport problem**. Each image patch is transported to the most semantically relevant text subspace. Transport plan cost = alignment quality. For our use case: the transport cost between OCR text tokens and image patches directly measures how well each character in the OCR output corresponds to a visual region in the original image. A patch with a wrong character finds no good transport target, increasing cost.
- Shift tolerance: **High** — OT finds globally optimal patch assignment regardless of spatial layout.
- Character sensitivity: **High** — each character token must find a matching image patch; wrong characters increase transport cost.
- Implementation: Medium — OT solver (POT library) + CLIP patch tokens.
- **Priority: High.** The most theoretically principled approach to our exact problem.

---

### Document-Specific Findings (2024–2025)

**"Survey of OCR Evaluation Methods"**
arXiv 2025. https://arxiv.org/abs/2603.25761

Argues that CER/WER miss structural, spatial, and semantic quality dimensions, and calls for visual-grounding evaluation methods. Reviews OCRBench v2, OmniDocBench, olmOCR-Bench. Directly validates this project's direction of pursuing visual metrics over string-edit metrics.

**"Spatially-Grounded Document Retrieval via Patch-to-Region Relevance"**
arXiv Dec 2024. https://arxiv.org/abs/2512.02660

Extends ColPali's late-interaction patch similarity (32×32 grid, 1,024 patches per page) with spatial relevance propagation to semantic regions. Key finding: retrieval effectiveness correlates with **text coverage** — patch similarity is sensitive to missing text blocks. Since our reconstructed images have text coverage proportional to OCR completeness, patch similarity naturally penalizes missed text regions.

**UniSim-Bench — "Unified Multi-Modal Perceptual Metric Benchmark"**
Ghazanfari et al. CVPR Workshop 2025. https://arxiv.org/abs/2412.10594

First benchmark across 7 perceptual similarity tasks (25 datasets). Key finding: **encoder-based VLMs (CLIP, LLaVA-NeXT) generalize better than generative models** as perceptual metrics. This supports our DINOv2/CLIP approaches over diffusion-based metrics for document comparison.

---

### Updated Recommendations (incorporating 2023–2025 findings)

| Priority | Method | Key Reason |
|----------|--------|------------|
| **1 — Immediate** | ST-LPIPS (drop-in) | Directly fixes LPIPS shift penalty from font metric differences |
| **2 — Immediate** | MS-SSIM (drop-in) | One-line change; better multi-scale handling |
| **3 — Near-term** | DINOv2 (or DINOv3) patch cosine + ink weighting | Best character-scale dense features; training-free |
| **4 — Near-term** | Patch-CLIP best-match pooling | Addresses global-CLIP layout weakness |
| **5 — Near-term** | TokenCLIP OT matching | Most principled character-sensitive shift-invariant metric |
| **6 — Research** | PE-spatial (Perception Encoder) | Best single backbone once available; DocVQA-validated |
| **7 — Research** | dino.txt text-grounded localization | Novel reference-free character verification mode |

### Key Synergies

- **DINOv2 patches + ink-density masking**: Focus the patch cosine metric only on text-bearing regions — removes background dilution, gives pure character-level signal.
- **ST-LPIPS + DINOv2 cosine + Sliced Wasserstein**: Three-metric shift-tolerant ensemble covering perceptual, semantic, and distributional aspects. Replaces current SSIM+LPIPS+CLIP with a uniformly shift-tolerant version.
- **TokenCLIP OT + layout-level OT**: Two-level hierarchical matching: layout (bbox positions) at the macro level, character content at the micro level. Simultaneously captures positional and content accuracy.

---

## Connection to Known Project Issues

| Known Issue | Recommended Mitigation |
|-------------|------------------------|
| CLIP drops on complex layouts (The Economist: 0.450 in Iter 2) | Patch-CLIP directly addresses this — complex layouts have locally similar patches even when globally mismatched. |
| LPIPS penalizes font differences (not character errors) | ST-LPIPS reduces this; DINOv2 patch matching focuses on structural features less sensitive to exact font metrics. |
| Text area underweighting (text < 1% of pixels) | Ink-weighted aggregation in DINOv2 method explicitly addresses this by focusing patches on ink-bearing regions. |
