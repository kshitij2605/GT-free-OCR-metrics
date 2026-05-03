"""CDM (Character Detection Matching) scorer for formula evaluation.

Faithful port of OmniDocBench/UniMERNet CDM:
  - Tokenise LaTeX via Node.js KaTeX
  - Colour each token with a unique RGB via \\textcolor[RGB]{r,g,b}{token}
  - Render with pdflatex + pdftoppm at 200 DPI
  - Extract per-token bboxes from pixel colours
  - Match GT↔OCR tokens with HungarianMatcher (token + position + order costs)
  - Filter outliers with RANSAC (SimpleAffineTransform)
  - Return F1 score

Differences from the original:
  - xelatex → pdflatex  (no CJK needed for math)
  - ImageMagick → pdftoppm  (avoids Ubuntu PDF policy restriction)
  - \\mathcolor[RGB] → \\textcolor[RGB]  (pdflatex-compatible)
  - Node.js tokeniser called directly from the bundled cdm_js/ directory
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from threading import Timer

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_JS_DIR = Path(__file__).parent / "cdm_js"
_PREPROCESS_JS = _JS_DIR / "preprocess_formula.js"

# ---------------------------------------------------------------------------
# Colour palette  (matches OmniDocBench gen_color_list(num=5800, gap=15))
# ---------------------------------------------------------------------------

def _gen_color_list(num: int = 5800, gap: int = 15) -> list[tuple[int, int, int]]:
    """Generate a palette of ``num`` distinct RGB colors spaced ``gap`` apart for CDM token colorization."""
    single_num = 255 // gap + 1
    max_num = single_num ** 3
    num = min(num + 1, max_num)
    colors: list[tuple[int, int, int]] = []
    for idx in range(num):
        R = idx // single_num ** 2
        GB = idx % single_num ** 2
        G = GB // single_num
        B = GB % single_num
        colors.append((R * gap, G * gap, B * gap))
    return colors[1:]  # skip (0,0,0)


_COLOR_LIST = _gen_color_list()

# ---------------------------------------------------------------------------
# LaTeX tokeniser (Node.js / KaTeX)
# ---------------------------------------------------------------------------

def _run_cmd(cmd: str, timeout_sec: int = 30) -> None:
    """Run a shell command with a timeout, killing the process if it exceeds the limit."""
    proc = subprocess.Popen(cmd, shell=True)
    timer = Timer(timeout_sec, proc.kill)
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()


def _tokenize_latex(latex: str) -> str:
    """Return space-tokenised LaTeX via the KaTeX Node.js tokeniser.

    Falls back to the raw string (already stripped) if Node.js fails.
    """
    if not latex.strip():
        return latex

    # align / split → aligned  (mirrors OmniDocBench tokenise_latex.py)
    prepre = re.sub(
        r"\\begin\{(split|align|alignedat|alignat|eqnarray)\*?\}(.+?)\\end\{\1\*?\}",
        r"\\begin{aligned}\2\\end{aligned}",
        latex,
        flags=re.S,
    )
    prepre = re.sub(
        r"\\begin\{(smallmatrix)\*?\}(.+?)\\end\{\1\*?\}",
        r"\\begin{matrix}\2\\end{matrix}",
        prepre,
        flags=re.S,
    )

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write(prepre)
            tmp_path = tmp.name
        out_path = tmp_path + ".out"
        ret = subprocess.call(
            f"cat {tmp_path} | node {_PREPROCESS_JS} normalize > {out_path}",
            shell=True, timeout=30,
        )
        if ret != 0:
            return latex
        with open(out_path) as f:
            result = f.read().strip()
        Path(tmp_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)
        return result if result else latex
    except Exception:
        return latex

# ---------------------------------------------------------------------------
# LaTeX normalisation (ported from OmniDocBench latex_processor.py)
# ---------------------------------------------------------------------------

_SKIP_PATTERNS = [
    r"\{", r"\}", r"[\[\]]", r"\\begin\{.*?\}", r"\\end\{.*?\}", r"\^", r"\_",
    r"\\.*rule.*", r"\\.*line.*", r"\[[\-.0-9]+[epm][xtm]\]",
]
_SKIP_TOKENS = [
    "\\", "\\\\", "\\index", "\\a", "&", "$", "\\multirow", "\\def",
    "\\raggedright", "\\url", "\\cr", "\\ensuremath", "\\left", "\\right",
    "\\mathchoice", "\\scriptstyle", "\\displaystyle", "\\qquad", "\\quad",
    "\\,", "\\!", "~", "\\boldmath",
]
_PHANTOM_TOKENS = ["\\fontfamily", "\\vphantom", "\\phantom", "\\rowcolor", "\\ref"]
_TWO_TAIL_TOKENS = ["\\frac", "\\binom"]
_AB_TAIL_TOKENS = ["\\xrightarrow", "\\xleftarrow", "\\sqrt"]
_TWO_TAIL_INVISB_TOKENS = ["\\overset", "\\underset", "\\stackrel"]
_ONE_TAIL_TOKENS = [
    "\\widetilde", "\\overline", "\\hat", "\\widehat", "\\tilde", "\\Tilde",
    "\\dot", "\\bar", "\\vec", "\\underline", "\\underbrace", "\\check",
    "\\breve", "\\Bar", "\\Vec", "\\mathring", "\\ddot",
]
_ONE_TAIL_INVISB_TOKENS = [
    "\\boldsymbol", "\\pmb", "\\textbf", "\\mathrm", "\\mathbf", "\\mathbb",
    "\\mathcal", "\\textmd", "\\texttt", "\\textnormal", "\\text", "\\textit",
    "\\textup", "\\mathop", "\\mathbin", "\\smash", "\\operatorname",
    "\\textrm", "\\mathfrak", "\\emph", "\\textsf", "\\textsc",
]


def _find_matching_brace(
    seq: list[str], start: int, brace: tuple[str, str] = ("{", "}")
) -> int:
    """Return the index of the closing brace in ``seq`` matching the opening brace at ``start``.

    Returns -1 if no matching brace is found.
    """
    left, right = brace
    depth = 0
    for i, tok in enumerate(seq[start:], start=start):
        if tok == left:
            depth += 1
        elif tok == right:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _normalize_latex(l: str) -> str:
    """Normalize space-tokenised LaTeX formula (port of latex_processor.normalize_latex)."""
    l = l.strip()
    l = l.replace(r"\pmatrix", r"\mypmatrix").replace(r"\matrix", r"\mymatrix")
    for item in ["\\raggedright", "\\arraybackslash", "\\lowercase", "\\uppercase"]:
        l = l.replace(item, "")

    # collapse \hspace { 1 . 5 cm } → \hspace{1.5cm}
    for pat in [r"\\[hv]space { [.0-9a-z ]+ }"]:
        for old in re.findall(pat, l, re.DOTALL):
            l = l.replace(old, old.replace(" ", ""))

    # \begin{array} { lrc } → compact
    for old in re.findall(r"\\begin\{array\} \{ [lrc ]+ \}", l, re.DOTALL):
        new = old.replace("\\begin{array} ", "<s>").replace(" ", "").replace("<s>", "\\begin{array} ")
        l = l.replace(old, new)

    # \not= → \not=
    for old in re.findall(r"\\not [<>+=\-]", l, re.DOTALL):
        l = l.replace(old, old.replace(" ", ""))

    # ellipsis / trig expansions
    l = " " + l + " "
    for src, dst in [
        (" \\ldots ", " . . . "), (" \\cdots ", " . . . "),
        (" \\dots ", " . . . "), (" \\dotsb ", " . . . "),
        (" \\log ", " \\mathrm { l o g } "), (" \\exp ", " \\mathrm { e x p } "),
        (" \\sin ", " \\mathrm { s i n } "), (" \\cos ", " \\mathrm { c o s } "),
        (" \\tan ", " \\mathrm { t a n } "), (" \\tanh ", " \\mathrm { t a n h } "),
        (" \\cosh ", " \\mathrm { c o s h } "), (" \\sinh ", " \\mathrm { s i n h } "),
    ]:
        l = l.replace(src, dst)

    # \big( → \big(  (join with next bracket token)
    for pat in [r"\\[Bb]ig[g]?[glrm]? [(){}|\[\]] ", r"\\[Bb]ig[g]?[glrm]? \\.*? "]:
        for old in re.findall(pat, l, re.DOTALL):
            l = l.replace(old, old.replace(" ", "") + " ")

    l = l.replace("\\operatorname *", "\\operatorname")
    l = l.replace("\\lefteqn", "")
    l = l.replace("\\footnote ", "^ ")

    # \' x → \'x  (but not \' {)
    for old in re.findall(r"\\\' [^{] ", l, re.DOTALL):
        l = l.replace(old, old.replace(" ", "") + " ")

    # remove original color commands
    pat = (
        r"\\colorbox[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } "
        r"|\\color[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } "
        r"|\\textcolor[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } "
        r"|\\cellcolor[ \[\]RGBrgb]+{ [A-Za-z 0-9,!]+ } "
    )
    for old in re.findall(pat, l, re.DOTALL):
        l = l.replace(old, "")

    # fill missing braces around ONE_TAIL / TWO_TAIL / AB_TAIL tokens
    l_split = l.split(" ")
    idx = 0
    while idx < len(l_split):
        tok = l_split[idx]
        if tok in _ONE_TAIL_TOKENS + _ONE_TAIL_INVISB_TOKENS:
            sub = idx + 1
            while sub < len(l_split) and l_split[sub] in _ONE_TAIL_TOKENS + _ONE_TAIL_INVISB_TOKENS:
                sub += 1
            new = l_split[:idx]
            for ii in range(idx, sub):
                new = new + [l_split[ii], "{"]
            if sub < len(l_split) and l_split[sub] != "{":
                new = new + [l_split[sub]] + ["}"] * (sub - idx)
                l_split = new + l_split[sub + 1:]
            elif sub < len(l_split):
                end = _find_matching_brace(l_split, sub)
                if end != -1:
                    new = new + l_split[sub + 1:end] + ["}"] * (sub - idx)
                    l_split = new + l_split[end + 1:]
        elif tok in _AB_TAIL_TOKENS and idx + 1 < len(l_split):
            if l_split[idx + 1] not in ("[", "{"):
                l_split = l_split[:idx+1] + ["{"] + [l_split[idx+1]] + ["}"] + l_split[idx+2:]
            else:
                if l_split[idx + 1] == "[":
                    end1 = _find_matching_brace(l_split, idx + 1, brace=("[", "]"))
                else:
                    end1 = idx
                if end1 + 1 < len(l_split) and l_split[end1 + 1] != "{":
                    l_split = l_split[:end1+1] + ["{"] + [l_split[end1+1]] + ["}"] + l_split[end1+2:]
        elif tok in _TWO_TAIL_TOKENS + _TWO_TAIL_INVISB_TOKENS and idx + 1 < len(l_split):
            if l_split[idx + 1] != "{":
                l_split = l_split[:idx+1] + ["{"] + [l_split[idx+1]] + ["}"] + l_split[idx+2:]
            end1 = _find_matching_brace(l_split, idx+1)
            if end1 != -1 and end1 + 1 < len(l_split) and l_split[end1 + 1] != "{":
                l_split = l_split[:end1+1] + ["{"] + [l_split[end1+1]] + ["}"] + l_split[end1+2:]
        idx += 1

    return " ".join(l_split)

# ---------------------------------------------------------------------------
# Token colourisation (port of token_add_color_RGB, using \textcolor)
# ---------------------------------------------------------------------------

def _color_cmd(idx: int) -> str:
    """Return the opening textcolor RGB command string for color slot ``idx``."""
    return f"\\textcolor[RGB]{{<color_{idx}>}}{{"


def _token_add_color_rgb(
    l_split: list[str], idx: int, token_list: list[str], brace_color: bool = False
) -> tuple[list[str], int, list[str]]:
    """Wrap the token at l_split[idx] with a unique RGB colour placeholder."""
    tok = l_split[idx]

    if not tok:
        return l_split, idx + 1, token_list

    if tok in _PHANTOM_TOKENS:
        end = (_find_matching_brace(l_split, idx + 1) if idx + 1 < len(l_split) and l_split[idx + 1] == "{" else idx + 1)
        return l_split, end + 1, token_list

    if tok in _TWO_TAIL_TOKENS:
        num_start = idx + 1
        num_end = _find_matching_brace(l_split, num_start)
        den_start = num_end + 1
        den_end = _find_matching_brace(l_split, den_start)
        cc = _color_cmd(len(token_list))
        l_split = l_split[:idx] + [cc + tok] + l_split[idx+1:den_end+1] + ["}"] + l_split[den_end+1:]
        token_list.append(tok)
        return l_split, idx + 1, token_list

    if tok in _ONE_TAIL_TOKENS:
        num_start = idx + 1
        num_end = _find_matching_brace(l_split, num_start)
        cc = _color_cmd(len(token_list))
        if tok != "\\underbrace" and num_end + 1 < len(l_split) and l_split[num_end + 1] == "_":
            l_split = l_split[:idx] + ["{" + cc + tok] + l_split[idx+1:num_end+1] + ["}}"] + l_split[num_end+1:]
        else:
            l_split = l_split[:idx] + [cc + tok] + l_split[idx+1:num_end+1] + ["}"] + l_split[num_end+1:]
        token_list.append(tok)
        return l_split, idx + 1, token_list

    if tok in _ONE_TAIL_INVISB_TOKENS:
        num_start = idx + 1
        num_end = _find_matching_brace(l_split, num_start)
        if num_end - num_start == 2:
            cc = _color_cmd(len(token_list))
            token_list.append(l_split[num_start + 1])
            l_split = l_split[:num_start+1] + [cc + l_split[num_start+1] + "}"] + l_split[num_end:]
        else:
            sub = num_start + 1
            while sub < num_end:
                l_split, sub, token_list = _token_add_color_rgb(l_split, sub, token_list)
        return l_split, num_end + 1, token_list

    if tok in _AB_TAIL_TOKENS and idx + 1 < len(l_split):
        if l_split[idx + 1] == "{":
            num_start = idx + 1
            num_end = _find_matching_brace(l_split, num_start)
            cc = _color_cmd(len(token_list))
            l_split = l_split[:idx] + [cc + tok] + l_split[idx+1:num_end+1] + ["}"] + l_split[num_end+1:]
            token_list.append(tok)
            sub = num_start + 1
            while sub < num_end:
                l_split, sub, token_list = _token_add_color_rgb(l_split, sub, token_list)
            return l_split, num_end + 1, token_list
        if l_split[idx + 1] == "[":
            num_start = idx + 1
            num_end = _find_matching_brace(l_split, num_start, brace=("[", "]"))
            den_start = num_end + 1
            den_end = _find_matching_brace(l_split, den_start)
            cc = _color_cmd(len(token_list))
            l_split = l_split[:idx] + [cc + tok] + l_split[idx+1:den_end+1] + ["}"] + l_split[den_end+1:]
            token_list.append(tok)
            sub = num_start + 1
            while sub < num_end:
                l_split, sub, token_list = _token_add_color_rgb(l_split, sub, token_list, brace_color=True)
            sub = den_start + 1
            while sub < den_end:
                l_split, sub, token_list = _token_add_color_rgb(l_split, sub, token_list)
            return l_split, den_end + 1, token_list

    if tok in _SKIP_TOKENS + _TWO_TAIL_INVISB_TOKENS or any(re.match(p, tok) for p in _SKIP_PATTERNS):
        if (tok == "[" and idx > 0 and l_split[idx-1] != "\\sqrt") or \
           (tok == "]" and idx >= 3 and l_split[idx-3] != "\\sqrt"):
            cc = _color_cmd(len(token_list))
            l_split = l_split[:idx] + [cc + tok + "}"] + l_split[idx+1:]
            token_list.append(tok)
        return l_split, idx + 1, token_list

    # Normal token
    cc = _color_cmd(len(token_list))
    if brace_color or (idx > 1 and l_split[idx-1] == "_"):
        l_split = l_split[:idx] + ["{" + cc + tok + "}}"] + l_split[idx+1:]
    else:
        l_split = l_split[:idx] + [cc + tok + "}"] + l_split[idx+1:]
    token_list.append(tok)
    return l_split, idx + 1, token_list


def _colorize(latex_tokens: str, color_list: list[tuple[int, int, int]]) -> tuple[str, list[str]]:
    """Wrap each token in a unique RGB colour. Returns (coloured_latex, token_list)."""
    l_split = latex_tokens.strip().split(" ")
    token_list: list[str] = []
    idx = 0
    while idx < len(l_split):
        l_split, idx, token_list = _token_add_color_rgb(l_split, idx, token_list)

    rgb_latex = " ".join(l_split)
    used_colors = color_list[: len(token_list)]
    for i, (R, G, B) in enumerate(used_colors):
        rgb_latex = rgb_latex.replace(f"<color_{i}>", f"{R},{G},{B}")
    return rgb_latex, token_list

# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

_TEX_TEMPLATE = r"""\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\geometry{a5paper,scale=0.98}
\pagestyle{empty}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\begin{document}
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""


def _render_to_png(colored_latex: str, tmp_dir: str) -> Path | None:
    """Compile coloured LaTeX to PNG via pdflatex + pdftoppm. Returns PNG path or None."""
    tex_src = _TEX_TEMPLATE % colored_latex
    tex_path = Path(tmp_dir) / "formula.tex"
    tex_path.write_text(tex_src, encoding="utf-8")

    _run_cmd(
        f"pdflatex -interaction=nonstopmode -output-directory={tmp_dir} {tex_path} > /dev/null 2>&1",
        timeout_sec=30,
    )
    pdf_path = Path(tmp_dir) / "formula.pdf"
    if not pdf_path.exists():
        return None

    _run_cmd(
        f"pdftoppm -r 400 -png {pdf_path} {Path(tmp_dir) / 'out'} > /dev/null 2>&1",
        timeout_sec=30,
    )
    pngs = sorted(Path(tmp_dir).glob("out-*.png"))
    return pngs[0] if pngs else None

# ---------------------------------------------------------------------------
# Bbox extraction from pixel colours
# ---------------------------------------------------------------------------

def _extract_bboxes(
    png_path: Path, color_list: list[tuple[int, int, int]]
) -> list[list[int]]:
    """Return [x_min, y_min, x_max, y_max] per colour; [] if colour absent."""
    img = Image.open(png_path).convert("RGB")
    W = img.size[0]
    pixels = list(img.getdata())

    bboxes: list[list[int]] = []
    for target in color_list:
        xs = [i % W for i, p in enumerate(pixels) if p == target]
        ys = [i // W for i, p in enumerate(pixels) if p == target]
        if xs:
            bboxes.append([min(xs)-1, min(ys)-1, max(xs)+1, max(ys)+1])
        else:
            bboxes.append([])
    return bboxes


_DISPLAY_DOLLAR_RE = re.compile(r"^\$\$(.+)\$\$$", re.DOTALL)
_DISPLAY_BRACKET_RE = re.compile(r"^\\\[(.+?)(?<!\\)\\\]$", re.DOTALL)
_INLINE_DOLLAR_RE = re.compile(r"^\$([^$].+[^$])\$$", re.DOTALL)


def _strip_math_delimiters(latex: str) -> str:
    """Strip outer math-mode delimiters ($$, brackets) from a LaTeX string."""
    text = latex.strip()
    for pat in (_DISPLAY_DOLLAR_RE, _DISPLAY_BRACKET_RE, _INLINE_DOLLAR_RE):
        m = pat.match(text)
        if m:
            return m.group(1).strip()
    return text


def _latex_to_token_bboxes(
    latex: str, color_list: list[tuple[int, int, int]], tmp_dir: str
) -> list[dict]:
    """Full pipeline → list of {token, bbox} dicts (empty bbox = invisible token)."""
    latex = _strip_math_delimiters(latex)
    tokenized = _tokenize_latex(latex)
    normalized = _normalize_latex(tokenized)
    colored, token_list = _colorize(normalized, color_list)
    if not token_list:
        return []

    png_path = _render_to_png(colored, tmp_dir)
    if png_path is None:
        return []

    used_colors = color_list[: len(token_list)]
    bboxes = _extract_bboxes(png_path, used_colors)
    return [{"token": t, "bbox": b} for t, b in zip(token_list, bboxes) if b]

# ---------------------------------------------------------------------------
# Hungarian matcher (port of OmniDocBench visual_matcher.HungarianMatcher)
# ---------------------------------------------------------------------------

def _norm_same_token(token: str) -> str:
    """Normalize equivalent LaTeX token representations to a canonical form for CDM matching."""
    special = {
        "\\cdot": ".", "\\mid": "|", "\\to": "\\rightarrow", "\\top": "T",
        "\\Tilde": "\\tilde", "\\cdots": "\\dots", "\\prime": "'",
        "\\ast": "*", "\\left<": "\\langle", "\\right>": "\\rangle",
    }
    token = special.get(token, token)
    if token.startswith("\\left") or token.startswith("\\right"):
        token = token.replace("\\left", "").replace("\\right", "")
    if token.startswith("\\big") or token.startswith("\\Big"):
        token = ("\\" + token[4:].split("\\")[-1]) if "\\" in token[4:] else token[-1]
    if token in ("\\leq", "\\geq"):
        return token[:-1]
    if token in ("\\lVert", "\\rVert", "\\Vert"):
        return "\\|"
    if token in ("\\lvert", "\\rvert", "\\vert"):
        return "|"
    if token.endswith("rightarrow"):
        return "\\rightarrow"
    if token.endswith("leftarrow"):
        return "\\leftarrow"
    if token.startswith("\\wide"):
        return token.replace("wide", "")
    if token.startswith("\\var"):
        return token.replace("\\var", "")
    return token


def _hungarian_match(
    box_gt: list[dict],
    box_pred: list[dict],
    gt_img_size: tuple[int, int],
    pred_img_size: tuple[int, int],
    cost_token: float = 1.0,
    cost_position: float = 0.05,
    cost_order: float = 0.15,
) -> list[tuple[int, int]]:
    """Return list of (gt_idx, pred_idx) matched pairs."""
    W_gt, H_gt = gt_img_size
    W_pr, H_pr = pred_img_size

    # Token cost matrix
    all_tokens = {d["token"] for d in box_gt + box_pred}
    all_norms = {_norm_same_token(t) for t in all_tokens}
    t2id = {t: i for i, t in enumerate(all_tokens)}
    n2id = {t: i for i, t in enumerate(all_norms)}

    n, m = len(box_gt), len(box_pred)
    pred_logits = np.zeros((m, len(t2id)))
    pred_norm_logits = np.zeros((m, len(n2id)))
    gt_ids = np.array([t2id[d["token"]] for d in box_gt])
    gt_norm_ids = np.array([n2id[_norm_same_token(d["token"])] for d in box_gt])
    for j, d in enumerate(box_pred):
        pred_logits[j, t2id[d["token"]]] = 1
        pred_norm_logits[j, n2id[_norm_same_token(d["token"])]] = 1

    token_cost = (1.0 - pred_logits[:, gt_ids]).T          # (n, m)
    norm_cost = (1.0 - pred_norm_logits[:, gt_norm_ids]).T
    token_cost[np.logical_and(token_cost == 1, norm_cost == 0)] = 0.05

    # Position cost (L1, normalised coords)
    def _box_centers(boxes: list[dict], W: int, H: int) -> np.ndarray:
        """Return normalized (x1/W, y1/H, x2/W, y2/H) coordinate arrays for a list of bbox dicts."""
        arr = []
        for d in boxes:
            x1, y1, x2, y2 = d["bbox"]
            arr.append([x1/W, y1/H, x2/W, y2/H])
        return np.array(arr)

    gt_pos = _box_centers(box_gt, W_gt, H_gt)
    pr_pos = _box_centers(box_pred, W_pr, H_pr)
    pos_cost = cdist(gt_pos, pr_pos, "minkowski", p=1) / gt_pos.shape[1]

    # Order cost
    gt_ord = np.array([[i / n] for i in range(n)])
    pr_ord = np.array([[j / m] for j in range(m)])
    ord_cost = cdist(gt_ord, pr_ord, "minkowski", p=1)

    cost = cost_token * token_cost + cost_position * pos_cost + cost_order * ord_cost
    cost[np.isnan(cost) | np.isinf(cost)] = 100

    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))

# ---------------------------------------------------------------------------
# RANSAC outlier filtering
# ---------------------------------------------------------------------------

class _SimpleAffineTransform:
    """Translation + uniform scale transform (no rotation)."""

    def __init__(self) -> None:
        """Initialize with zero translation and unit scale."""
        self.translation = np.zeros(2)
        self.scale = 1.0

    def estimate(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """Estimate translation and uniform scale from corresponding src/dst point arrays."""
        src_c = np.mean(src, axis=0)
        dst_c = np.mean(dst, axis=0)
        self.translation = dst_c - src_c
        src_d = np.linalg.norm(src - src_c, axis=1)
        dst_d = np.linalg.norm(dst - dst_c, axis=1)
        self.scale = float(np.mean(dst_d) / (np.mean(src_d) + 1e-10))
        return True

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Apply the estimated affine transform to an array of coordinates."""
        return self.scale * (coords - np.mean(coords, axis=0)) + np.mean(coords, axis=0) + self.translation

    def residuals(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Return per-point Euclidean residuals between the transformed src and dst arrays."""
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))


def _ransac_inliers(
    matched_idxes: list[tuple[int, int]],
    box_gt: list[dict],
    box_pred: list[dict],
    max_iter: int = 3,
    min_samples: int = 3,
    residual_threshold: float = 25.0,
    max_trials: int = 50,
) -> np.ndarray:
    """Return boolean inlier mask over matched_idxes."""
    from skimage.measure import ransac as sk_ransac

    src, dst = [], []
    for a, b in matched_idxes:
        x1, y1, x2, y2 = box_gt[a]["bbox"]
        x3, y3, x4, y4 = box_pred[b]["bbox"]
        src.append([(y1+y2)/2, (x1+x2)/2])
        dst.append([(y3+y4)/2, (x3+x4)/2])

    src_arr = np.array(src, dtype=float)
    dst_arr = np.array(dst, dtype=float)
    n = len(matched_idxes)
    inliers = np.zeros(n, dtype=bool)

    if n <= min_samples:
        return np.ones(n, dtype=bool)

    for _ in range(max_iter):
        remaining = ~inliers
        if remaining.sum() <= min_samples:
            break
        try:
            _, new_in = sk_ransac(
                (src_arr[remaining], dst_arr[remaining]),
                _SimpleAffineTransform,
                min_samples=min_samples,
                residual_threshold=residual_threshold,
                max_trials=max_trials,
            )
        except Exception:
            break
        if new_in is None or not new_in.any():
            break
        # map new_in back to full index space
        rem_indices = np.where(remaining)[0]
        for ri, flag in zip(rem_indices, new_in):
            if flag:
                inliers[ri] = True
        if inliers.sum() >= n:
            break

    return inliers

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_cdm_pair(gt_latex: str, ocr_latex: str) -> float | None:
    """CDM F1 score ∈ [0, 1] between GT and OCR LaTeX formula strings.

    Returns None if either formula fails to produce any token bboxes
    (rendering failure), so the caller can skip the pair.
    Returns 0.0 if OCR produces no tokens but GT does.
    """
    with tempfile.TemporaryDirectory() as gt_tmp, \
         tempfile.TemporaryDirectory() as ocr_tmp:

        box_gt = _latex_to_token_bboxes(gt_latex, _COLOR_LIST, gt_tmp)
        box_pred = _latex_to_token_bboxes(ocr_latex, _COLOR_LIST, ocr_tmp)

    if not box_gt:
        return None   # GT render failed — skip
    if not box_pred:
        return 0.0    # OCR render failed or empty

    # Need rendered image sizes for normalised position cost
    # Approximate from bbox extents (sufficient for cost normalisation)
    def _img_size(boxes: list[dict]) -> tuple[int, int]:
        """Return (width, height) extent from a list of bbox dicts."""
        xs = [b for d in boxes for b in (d["bbox"][0], d["bbox"][2])]
        ys = [b for d in boxes for b in (d["bbox"][1], d["bbox"][3])]
        return max(xs) + 2, max(ys) + 2

    gt_size = _img_size(box_gt)
    pr_size = _img_size(box_pred)

    matched = _hungarian_match(box_gt, box_pred, gt_size, pr_size)
    if not matched:
        return 0.0

    inliers = _ransac_inliers(matched, box_gt, box_pred)

    # Also discard matches where token cost == 1 (completely different tokens)
    # (mirrors OmniDocBench evaluation.py post-RANSAC filter)
    for idx, (a, b) in enumerate(matched):
        if box_gt[a]["token"] != box_pred[b]["token"] and \
           _norm_same_token(box_gt[a]["token"]) != _norm_same_token(box_pred[b]["token"]):
            inliers[idx] = False

    final = int(inliers.sum())
    return 2.0 * final / (len(box_gt) + len(box_pred))
