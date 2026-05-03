"""Bridge for comparing reference-free metrics with OmniDocBench reference-based metrics."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from apted import APTED
from apted import Config as APTEDConfig
from bs4 import BeautifulSoup, Tag
from pylatexenc.latex2text import LatexNodes2Text
from rapidfuzz.distance import Levenshtein
from scipy import stats
from scipy.optimize import linear_sum_assignment

from .cdm_scorer import compute_cdm_pair as _cdm_scorer_pair

# Formatting macros to strip during formula normalization (mirrors OmniDocBench normalized_formula).
_FORMULA_FILTER = [
    "\\mathbf", "\\mathrm", "\\mathnormal", "\\mathit", "\\mathbb", "\\mathcal",
    "\\mathscr", "\\mathfrak", "\\mathsf", "\\mathtt",
    "\\textbf", "\\text", "\\boldmath", "\\boldsymbol", "\\operatorname", "\\bm",
    "\\symbfit", "\\mathbfcal", "\\symbf", "\\scriptscriptstyle", "\\notag",
    "\\setlength", "\\coloneqq", "\\space", "\\thickspace", "\\thinspace",
    "\\medspace", "\\nobreakspace", "\\negmedspace",
    "\\quad", "\\qquad", "\\enspace", "\\substackw", " ", "$$",
    "\\left", "\\right", "\\displaystyle",
]


# ---------------------------------------------------------------------------
# TEDS helpers
# ---------------------------------------------------------------------------

class _TableNode:
    """Tree node for APTED-based TEDS computation."""

    def __init__(
        self,
        tag: str,
        colspan: int = 1,
        rowspan: int = 1,
        content: str = "",
    ) -> None:
        """Initialize a table tree node with tag, span attributes, text content, and empty children list."""
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children: list[_TableNode] = []


class _TEDSConfig(APTEDConfig):
    """APTED cost config mirroring OmniDocBench CustomConfig."""

    def rename(self, node1: _TableNode, node2: _TableNode) -> float:  # type: ignore[override]
        """Return rename cost between two nodes: 0 for identical structure/content, 1 otherwise."""
        if node1.tag != node2.tag or node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0
        if node1.tag == "td" and (node1.content or node2.content):
            a = OmniDocBenchBridge._clean_string(node1.content)
            b = OmniDocBenchBridge._clean_string(node2.content)
            ml = max(len(a), len(b))
            return Levenshtein.distance(a, b) / ml if ml else 0.0
        return 0.0

    def children(self, node: _TableNode) -> list[_TableNode]:
        """Return the children list of a _TableNode (required by the APTED Config interface)."""
        return node.children


def _html_node_to_table_node(elem: Tag) -> _TableNode | None:
    """Recursively convert a BS4 Tag into a _TableNode tree."""
    if not isinstance(elem, Tag):
        return None
    colspan = int(str(elem.attrs.get("colspan", 1) or 1))
    rowspan = int(str(elem.attrs.get("rowspan", 1) or 1))
    content = elem.get_text(strip=True) if elem.name == "td" else ""
    node = _TableNode(tag=elem.name, colspan=colspan, rowspan=rowspan, content=content)
    for child in elem.children:
        if isinstance(child, Tag):
            child_node = _html_node_to_table_node(child)
            if child_node is not None:
                node.children.append(child_node)
    return node


def _normalize_table_html(html_content: str) -> str:
    """Normalize HTML table for TEDS (mirrors OmniDocBench process_table_html)."""
    soup = BeautifulSoup(html_content, "html.parser")
    for th in soup.find_all("th"):
        if isinstance(th, Tag):
            th.name = "td"
    for tag_name in ("thead", "span", "tbody"):
        for tag in soup.find_all(tag_name):
            if isinstance(tag, Tag):
                tag.unwrap()
    for tag in soup.find_all(True):
        if isinstance(tag, Tag):
            for attr in ("style", "height", "width", "align", "class"):
                tag.attrs.pop(attr, None)
    for tag in soup.find_all(("sup", "sub", "div", "p")):
        if isinstance(tag, Tag):
            tag.unwrap()
    return str(soup)


def _build_table_tree(html_content: str) -> _TableNode:
    """Build a _TableNode tree from an HTML string."""
    normalized = _normalize_table_html(html_content)
    soup = BeautifulSoup(normalized, "html.parser")
    root = _TableNode("root")
    table = soup.find("table")
    if table and isinstance(table, Tag):
        child = _html_node_to_table_node(table)
        if child is not None:
            root.children.append(child)
    return root


def _count_nodes(node: _TableNode) -> int:
    """Count all nodes in the _TableNode tree recursively (root + all descendants)."""
    return 1 + sum(_count_nodes(c) for c in node.children)


def _compute_teds_pair(gt_html: str, ocr_html: str) -> float:
    """TEDS score in [0, 1] between two HTML table strings (1 = perfect)."""
    gt_tree = _build_table_tree(gt_html)
    ocr_tree = _build_table_tree(ocr_html)
    n_nodes = max(_count_nodes(gt_tree), _count_nodes(ocr_tree))
    if n_nodes == 0:
        return 1.0
    distance = float(APTED(gt_tree, ocr_tree, _TEDSConfig()).compute_edit_distance())
    return max(0.0, 1.0 - distance / n_nodes)


# ---------------------------------------------------------------------------
# Core dataclasses and utilities
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of correlating reference-free and reference-based metrics."""

    pearson: float
    spearman: float
    n_samples: int
    pearson_pvalue: float = 1.0
    spearman_pvalue: float = 1.0


def compute_correlation(ref_scores: list, our_scores: list) -> ComparisonResult:
    """Compute Pearson and Spearman correlation between two score lists."""
    if len(ref_scores) < 3 or len(our_scores) < 3:
        return ComparisonResult(pearson=0.0, spearman=0.0, n_samples=len(ref_scores))
    pearson_r, pearson_p = stats.pearsonr(ref_scores, our_scores)
    spearman_r, spearman_p = stats.spearmanr(ref_scores, our_scores)
    return ComparisonResult(
        pearson=float(pearson_r),
        spearman=float(spearman_r),
        n_samples=len(ref_scores),
        pearson_pvalue=float(pearson_p),
        spearman_pvalue=float(spearman_p),
    )


_DEFAULT_ANNOTATIONS = (
    Path(__file__).parents[3] / "data" / "annotations" / "sample_pages_annotations.json"
)

_LATEX_CONVERTER = LatexNodes2Text()
_INLINE_DOLLAR_RE = re.compile(r"\$([^$\n]+)\$")
_INLINE_PAREN_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


class OmniDocBenchBridge:
    """Bridge to OmniDocBench for comparing reference-free and reference-based metrics."""

    def __init__(self, omnidocbench_path: str | None = None, annotations_path: str | None = None):
        """Initialize bridge with annotations JSON path or full OmniDocBench dataset directory."""
        if annotations_path:
            self._annotations_path = Path(annotations_path)
            self.dataset_path = None
        elif omnidocbench_path:
            self.omnidocbench_path = Path(omnidocbench_path)
            self.dataset_path = self.omnidocbench_path / "OpenDataLab___OmniDocBench"
            self._annotations_path = None
        else:
            self._annotations_path = _DEFAULT_ANNOTATIONS
            self.dataset_path = None

    def load_ground_truth(self, json_path: str | None = None) -> list:
        """Load OmniDocBench ground truth JSON."""
        if json_path is None:
            if self._annotations_path:
                json_path = str(self._annotations_path)
            elif self.dataset_path:
                json_path = str(self.dataset_path / "OmniDocBench.json")
            else:
                raise ValueError("No annotations path or OmniDocBench path configured")
        with open(json_path) as f:
            return json.load(f)

    _TEXT_CATEGORIES = {"text_block", "title", "code_txt", "code_txt_caption", "reference"}
    _FORMULA_CATEGORIES = {"equation_isolated"}
    _TABLE_CATEGORIES = {"table"}

    @staticmethod
    def _textblock2unicode(text: str) -> str:
        """Convert inline LaTeX formulas to Unicode (mirrors OmniDocBench textblock2unicode)."""
        def _convert(formula: str) -> str:
            """Convert inline LaTeX to Unicode text; return the original string on failure."""
            try:
                return _LATEX_CONVERTER.latex_to_text(formula)
            except Exception:
                return formula

        text = _INLINE_DOLLAR_RE.sub(lambda m: _convert(m.group(1)), text)
        text = _INLINE_PAREN_RE.sub(lambda m: _convert(m.group(1)), text)
        for ch in ("\\", "_", "&", "%", "^"):
            text = text.replace(ch, "")
        return text

    @staticmethod
    def _clean_string(text: str) -> str:
        """Full OmniDocBench normalization: textblock2unicode then clean_string."""
        text = OmniDocBenchBridge._textblock2unicode(text)
        text = (
            text.replace("\\t", "").replace("\\n", "")
            .replace("\t", "").replace("\n", "")
            .replace("/t", "").replace("/n", "")
        )
        return re.sub(r"[^\w\u4e00-\u9fff]", "", text)

    @staticmethod
    def _normalized_formula(text: str) -> str:
        """Normalize display formula LaTeX (mirrors OmniDocBench normalized_formula)."""
        text = text.strip().strip("$").strip("\n")
        m = re.search(r"\\\[(.+?)(?<!\\)\\\]", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        text = re.sub(r"\\tag\{.*?\}", "", text)
        text = re.sub(r"\\hspace\{.*?\}", "", text)
        text = re.sub(r"\\begin\{.*?\}", "", text)
        text = re.sub(r"\\end\{.*?\}", "", text)
        text = re.sub(r"\\arraycolsep.*?\}", "", text)
        text = text.strip(".")
        for f in _FORMULA_FILTER:
            text = text.replace(f, "")
        return text.lower()

    @staticmethod
    def _extract_table_text(html_content: str) -> str:
        """Extract concatenated cell text from an HTML table for edit distance."""
        soup = BeautifulSoup(html_content, "html.parser")
        cells = [td.get_text(separator=" ", strip=True) for td in soup.find_all("td")]
        return " ".join(cells) if cells else soup.get_text(strip=True)

    def _merge_truncated_dets(self, page_data: dict) -> list[dict]:
        """Merge layout_dets elements linked by truncated relations."""
        dets = list(page_data.get("layout_dets", []))
        relations = page_data.get("extra", {}).get("relation", [])

        truncated_pairs = [
            (r["source_anno_id"], r["target_anno_id"])
            for r in relations
            if r.get("relation_type") == "truncated"
        ]
        if not truncated_pairs:
            return dets

        parent: dict[int, int] = {}

        def find(x: int) -> int:
            """Path-compression find for union-find over annotation IDs."""
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            """Union two annotation IDs in the truncation union-find structure."""
            parent[find(x)] = find(y)

        for src, tgt in truncated_pairs:
            union(src, tgt)

        groups: dict[int, list[dict]] = {}
        for det in dets:
            aid = det.get("anno_id")
            if aid is not None and aid in parent:
                root = find(aid)
                groups.setdefault(root, []).append(det)

        merged_set: set[int] = set()
        merged_dets: list[dict] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            sorted_group = sorted(group, key=lambda d: d.get("order", 0))
            base = dict(sorted_group[0])
            base["text"] = "".join(d.get("text", "") for d in sorted_group)
            merged_dets.append(base)
            merged_set.update(d["anno_id"] for d in group)

        result = [d for d in dets if d.get("anno_id") not in merged_set]
        result.extend(merged_dets)
        return result

    @staticmethod
    def _poly_to_bbox(poly: list[float]) -> tuple[float, float, float, float]:
        """Convert a flat polygon coordinate list [x0,y0,...] to an axis-aligned bounding box."""
        xs = poly[0::2]
        ys = poly[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _iou(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        """Compute intersection-over-union of two axis-aligned bounding boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _hungarian_edit_distance(
        gt_norms: list[str],
        ocr_norms: list[str],
        match_threshold: float = 0.7,
    ) -> float:
        """Aggregate edit distance using Hungarian text-similarity matching."""
        valid = [(i, g) for i, g in enumerate(gt_norms) if g]
        if not valid:
            return 0.0

        matches: dict[int, int] = {}
        if ocr_norms:
            n, m = len(valid), len(ocr_norms)
            cost = np.ones((n, m))
            for vi, (_, gc) in enumerate(valid):
                for j, oc in enumerate(ocr_norms):
                    ml = max(len(gc), len(oc))
                    cost[vi][j] = Levenshtein.distance(gc, oc) / ml if ml else 0.0
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r][c] <= match_threshold:
                    matches[r] = c

        total_edits = 0
        total_max_len = 0
        for vi, (_, gc) in enumerate(valid):
            oc = ocr_norms[matches[vi]] if vi in matches else ""
            ml = max(len(gc), len(oc))
            if ml:
                total_edits += Levenshtein.distance(gc, oc)
                total_max_len += ml

        return total_edits / total_max_len if total_max_len else 0.0

    def _match_gt_to_ocr(
        self,
        gt_norms: list[str],
        ocr_norms: list[str],
        match_threshold: float = 0.7,
    ) -> dict[int, int]:
        """Return Hungarian-matched gt_idx → ocr_idx mapping."""
        if not gt_norms or not ocr_norms:
            return {}
        n, m = len(gt_norms), len(ocr_norms)
        cost = np.ones((n, m))
        for i, gc in enumerate(gt_norms):
            for j, oc in enumerate(ocr_norms):
                ml = max(len(gc), len(oc))
                cost[i][j] = Levenshtein.distance(gc, oc) / ml if ml else 0.0
        rows, cols = linear_sum_assignment(cost)
        return {r: c for r, c in zip(rows, cols) if cost[r][c] <= match_threshold}

    # ------------------------------------------------------------------
    # Edit distance methods
    # ------------------------------------------------------------------

    def compute_page_edit_distance(
        self,
        page_data: dict,
        ocr_elements: list[dict],
        match_threshold: float = 0.7,
    ) -> float:
        """Element-level edit distance using Hungarian text-similarity matching."""
        dets = self._merge_truncated_dets(page_data)

        gt_texts_raw = []
        for det in sorted(dets, key=lambda d: d.get("order") or 0):
            if det.get("ignore", False):
                continue
            if det.get("category_type") not in self._TEXT_CATEGORIES:
                continue
            text = det.get("text", "")
            if text:
                gt_texts_raw.append(text)

        if not gt_texts_raw:
            return 0.0

        gt_norms = [self._clean_string(t) for t in gt_texts_raw]
        ocr_norms = [self._clean_string(o.get("text", "")) for o in ocr_elements]

        valid = [(i, g) for i, g in enumerate(gt_norms) if g]
        if not valid:
            return 0.0

        matches: dict[int, int] = {}
        if ocr_norms:
            n, m = len(valid), len(ocr_norms)
            cost = np.ones((n, m))
            for vi, (_, gc) in enumerate(valid):
                for j, oc in enumerate(ocr_norms):
                    ml = max(len(gc), len(oc))
                    cost[vi][j] = Levenshtein.distance(gc, oc) / ml if ml else 0.0
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r][c] <= match_threshold:
                    matches[r] = c

        total_edits = 0
        total_max_len = 0
        for vi, (_, gc) in enumerate(valid):
            oc = ocr_norms[matches[vi]] if vi in matches else ""
            ml = max(len(gc), len(oc))
            if ml:
                total_edits += Levenshtein.distance(gc, oc)
                total_max_len += ml

        return total_edits / total_max_len if total_max_len else 0.0

    def compute_page_formula_edit_distance(
        self,
        page_data: dict,
        ocr_formula_elements: list[dict],
        match_threshold: float = 0.7,
    ) -> float:
        """Edit distance for display formulas (equation_isolated category)."""
        dets = self._merge_truncated_dets(page_data)
        gt_latex_raw = [
            det.get("latex", "") or det.get("text", "")
            for det in sorted(dets, key=lambda d: d.get("order") or 0)
            if not det.get("ignore", False)
            and det.get("category_type") in self._FORMULA_CATEGORIES
            and (det.get("latex") or det.get("text"))
        ]
        if not gt_latex_raw:
            return 0.0

        gt_norms = [self._normalized_formula(t) for t in gt_latex_raw]
        ocr_norms = [self._normalized_formula(o.get("text", "")) for o in ocr_formula_elements]
        return self._hungarian_edit_distance(gt_norms, ocr_norms, match_threshold)

    def compute_page_table_edit_distance(
        self,
        page_data: dict,
        ocr_table_elements: list[dict],
        match_threshold: float = 0.7,
    ) -> float:
        """Edit distance for tables using text extracted from HTML content."""
        dets = self._merge_truncated_dets(page_data)
        gt_html_raw = [
            det.get("html", "") or det.get("text", "")
            for det in sorted(dets, key=lambda d: d.get("order") or 0)
            if not det.get("ignore", False)
            and det.get("category_type") in self._TABLE_CATEGORIES
            and (det.get("html") or det.get("text"))
        ]
        if not gt_html_raw:
            return 0.0

        gt_norms = [self._clean_string(self._extract_table_text(h)) for h in gt_html_raw]
        ocr_norms = [
            self._clean_string(self._extract_table_text(o.get("html", "") or o.get("text", "")))
            for o in ocr_table_elements
        ]
        return self._hungarian_edit_distance(gt_norms, ocr_norms, match_threshold)

    # ------------------------------------------------------------------
    # TEDS (Tree Edit Distance Score) for tables
    # ------------------------------------------------------------------

    def compute_page_teds(
        self,
        page_data: dict,
        ocr_table_elements: list[dict],
        match_threshold: float = 0.7,
    ) -> float | None:
        """Mean TEDS over GT-OCR table pairs on the page.

        GT tables are matched to OCR tables via text-based Hungarian matching
        (same pairing as table edit distance). Unmatched GT tables score 0.
        Returns None when no GT tables exist (dimension not applicable).
        """
        dets = self._merge_truncated_dets(page_data)
        gt_entries = [
            det
            for det in sorted(dets, key=lambda d: d.get("order") or 0)
            if not det.get("ignore", False)
            and det.get("category_type") in self._TABLE_CATEGORIES
            and (det.get("html") or det.get("text"))
        ]
        if not gt_entries:
            return None

        gt_html_raw = [d.get("html", "") or d.get("text", "") for d in gt_entries]
        gt_norms = [self._clean_string(self._extract_table_text(h)) for h in gt_html_raw]
        ocr_norms = [
            self._clean_string(self._extract_table_text(o.get("html", "") or o.get("text", "")))
            for o in ocr_table_elements
        ]

        matches = self._match_gt_to_ocr(gt_norms, ocr_norms, match_threshold)

        teds_scores = []
        for i, gt_html in enumerate(gt_html_raw):
            if i in matches:
                ocr_elem = ocr_table_elements[matches[i]]
                ocr_html = ocr_elem.get("html", "") or ocr_elem.get("text", "")
                score = _compute_teds_pair(gt_html, ocr_html)
            else:
                score = 0.0
            teds_scores.append(score)

        return sum(teds_scores) / len(teds_scores)

    # ------------------------------------------------------------------
    # CDM (Character-level Distribution Metric) for formulas
    # ------------------------------------------------------------------

    def compute_page_cdm(
        self,
        page_data: dict,
        ocr_formula_elements: list[dict],
        match_threshold: float = 0.7,
    ) -> float | None:
        """Mean CDM over GT-OCR formula pairs on the page.

        GT formulas matched to OCR formulas via normalized_formula Hungarian
        matching. Each matched pair is scored by rendering both LaTeX strings,
        extracting per-token bboxes, and computing F1 on positionally matched
        tokens (see cdm_scorer.py).

        Returns None when no GT formulas exist or all GT formulas fail to
        compile (dimension not applicable / not the OCR's fault).
        Returns 0.0–1.0 when at least one GT formula compiled successfully;
        OCR render failures are counted as 0.0.
        """
        dets = self._merge_truncated_dets(page_data)
        gt_entries = [
            det
            for det in sorted(dets, key=lambda d: d.get("order") or 0)
            if not det.get("ignore", False)
            and det.get("category_type") in self._FORMULA_CATEGORIES
            and (det.get("latex") or det.get("text"))
        ]
        if not gt_entries:
            return None

        gt_latex_raw = [d.get("latex", "") or d.get("text", "") for d in gt_entries]
        gt_norms = [self._normalized_formula(t) for t in gt_latex_raw]
        ocr_norms = [self._normalized_formula(o.get("text", "")) for o in ocr_formula_elements]

        matches = self._match_gt_to_ocr(gt_norms, ocr_norms, match_threshold)

        # Build (gt_latex, ocr_latex | None) pairs; None = unmatched GT formula.
        pairs = [
            (gt_latex, ocr_formula_elements[matches[i]].get("text", "") if i in matches else None)
            for i, gt_latex in enumerate(gt_latex_raw)
        ]

        def _score_pair(gt: str, ocr: str | None) -> float | None:
            """Score a (gt_latex, ocr_latex|None) formula pair with CDM; return None if GT fails to compile."""
            if ocr is None:
                return 0.0  # no OCR match for this GT formula
            result = _cdm_scorer_pair(gt, ocr)
            return result  # None if GT failed to compile; float otherwise

        n_workers = min(len(pairs), 8)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_score_pair, gt, ocr) for gt, ocr in pairs]
            cdm_scores = [f.result() for f in futures]

        # Filter: None means GT failed to compile — skip those pairs.
        valid = [s for s in cdm_scores if s is not None]
        # None = all GT formulas failed to compile (not evaluable)
        return sum(valid) / len(valid) if valid else None

    # ------------------------------------------------------------------
    # Legacy / utility methods
    # ------------------------------------------------------------------

    def extract_text_blocks(self, page_data: dict) -> list[str]:
        """Return ordered text strings from text-category layout_dets."""
        elements = []
        for det in page_data.get("layout_dets", []):
            if det.get("category_type") in self._TEXT_CATEGORIES:
                text = det.get("text", "")
                if text:
                    elements.append((det.get("order", 0), text))
        elements.sort(key=lambda x: x[0])
        return [text for _, text in elements]

    def get_page_text(self, page_data: dict) -> str:
        """Return all text blocks on the page joined by newlines."""
        return "\n".join(self.extract_text_blocks(page_data))

    @staticmethod
    def compute_edit_distance(text_a: str, text_b: str) -> float:
        """Normalized character-level edit distance in [0, 1] between two strings (0 = identical)."""
        a = OmniDocBenchBridge._clean_string(text_a)
        b = OmniDocBenchBridge._clean_string(text_b)
        if not a and not b:
            return 0.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 0.0
        return Levenshtein.distance(a, b) / max_len

    def run_comparison(
        self,
        ref_free_scores: list,
        ref_based_scores: list,
    ) -> ComparisonResult:
        """Correlate reference-free and reference-based score lists; returns a ComparisonResult."""
        return compute_correlation(ref_based_scores, ref_free_scores)
