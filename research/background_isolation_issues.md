# Background Isolation: Why Page-Background Canvas Breaks the Metric

**Date**: 2026-04-28
**Status**: dead end — reverted to plain white canvas

## Context

A short experiment replaced the plain-white reconstruction canvas with a
"page background" canvas: take the original page, paint white over every
detected element bbox (text + formula + table + image), and use the
result as the starting canvas for reconstruction. Variant elements draw
on top.

Visual goal: the reconstructed page mimics the source paper colour /
decorations / illustrations between elements rather than showing pure
white, so the rendered artefact looks like the original.

Two follow-up variants were considered:

1. Use the page-background canvas only for the reconstruction.
2. Subtract the page-background image from both the masked original
   (used as comparison GT) and the reconstruction so the metric measures
   only element-region pixels.

Both ideas were abandoned. This document records *why*, so the dead end
isn't re-explored later.

## The OCR-coverage trap

The bbox set used to build the background canvas is exactly the bbox set
the OCR system *detected*. That set is incomplete by definition — when
OCR misses content, the missed content stays visible in the
"background".

**Walk-through**

Consider a page with 5 lines of text. OCR detects line 1 and misses
lines 2-5.

- **`bg_canvas`** = original minus line 1 bbox (painted white).
  Lines 2-5 are still visible in the canvas because they were never
  detected, so the masking step doesn't touch them.

- **Reconstructed image** = render variant elements on `bg_canvas`.
  - Line 1 region: reconstructed glyphs drawn over the whited bbox.
  - Lines 2-5: original ink, copied through from the canvas.

- **`masked_original`** (the comparison target): same masking rule —
  detected element bboxes painted white. Line 1 is whited; lines 2-5
  remain.

- **Pixel-wise comparison**:
  - Lines 2-5: identical original ink in both images, contributes ~0
    to any difference / similarity metric.
  - Line 1 region: white in `masked_original`, reconstructed glyphs in
    the rendering — contributes the only real signal.

The metric collapses to "how faithfully are *captured* elements
reconstructed" and stops measuring "how well does the reconstruction
match the source page". An OCR system that captures 10% of a page can
score almost identically to one that captures 100% as long as both
reconstruct the captured fragments well — which destroys the entire
purpose of using full-page rendering as a reference-free quality proxy.

## Why subtracting the background also fails

The follow-up — subtract the bg canvas from both images so the metric
only sees element regions — has the same root cause. After subtraction,
both images are zero everywhere outside the *detected* element bboxes,
including the locations of OCR misses. Lines 2-5 never enter the
comparison, so missed content contributes nothing whether OCR captured
it or not. Worse, the subtracted images exactly equal what we'd get if
we'd rendered onto plain white in the first place and then masked
non-detected pixels — a more complicated path to the same property,
without the bbox-coverage penalty the plain-white path naturally
provides.

## Why plain white is the right choice

With a plain-white canvas:

- Reconstructed image: white everywhere except where the variant
  rendered an element.
- `masked_original` (current behaviour): original ink everywhere except
  detected image / table bboxes (painted white to avoid penalising
  variants that don't redraw images).

When OCR misses content:

- The missed text appears as ink in `masked_original` (it was never
  detected, so never masked).
- The same region in the reconstruction is plain white (no element to
  draw).
- The pixel-wise / structural difference between ink and white is
  exactly the penalty we want — coverage gaps register in the metric.

This is the property that makes the rendering-based metric a useful
proxy for OCR quality. The page-background canvas hides it.

## Decision

- Revert `scripts/regen_variant_reconstructed.py` to use the plain-white
  canvas.
- Keep the `canvas` parameter additions in `src/.../image_renderer.py`
  (harmless when None, defaults to plain background).
- Re-run all five variants on white.

## Things still worth doing

The original motivation — "the reconstructed image looks visually
unlike the source page" — is real, but it's a *visualisation* concern,
not a *metric* concern. If a user-facing comparison view is ever needed
later, render two outputs per page:

- `reconstructed.png` — plain-white canvas, used for metrics.
- `reconstructed_overlay.png` (optional) — bg-canvas variant, used for
  side-by-side visual inspection only.

Keep them separate so the metric path stays intact.
