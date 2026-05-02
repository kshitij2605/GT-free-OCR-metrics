"""Parse QwenVL HTML output to extract text elements and regions."""

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag


@dataclass
class TextElement:
    """A text element with bounding box."""

    text: str
    bbox: tuple[int, int, int, int]
    tag: str


@dataclass
class Region:
    """An image or table region with bounding box."""

    bbox: tuple[int, int, int, int]
    region_type: str


@dataclass
class TableElement:
    """A table element with bounding box and HTML content."""

    html_content: str
    bbox: tuple[int, int, int, int]


@dataclass
class ParsedDocument:
    """Parsed document containing text elements and regions."""

    text_elements: list[TextElement]
    image_regions: list[Region]
    table_regions: list[Region]
    table_elements: list[TableElement] = field(default_factory=list)
    plain_text: str = ""

    def to_pixel_coords(self, image_width: int, image_height: int) -> "ParsedDocument":
        """Convert [0,999] normalized coords to pixel coords."""

        def convert_bbox(
            bbox: tuple[int, int, int, int],
        ) -> tuple[int, int, int, int]:
            x1 = int(bbox[0] / 1000 * image_width)
            y1 = int(bbox[1] / 1000 * image_height)
            x2 = int(bbox[2] / 1000 * image_width)
            y2 = int(bbox[3] / 1000 * image_height)
            return (x1, y1, x2, y2)

        text_elements = [
            TextElement(text=te.text, bbox=convert_bbox(te.bbox), tag=te.tag)
            for te in self.text_elements
        ]
        image_regions = [
            Region(bbox=convert_bbox(r.bbox), region_type=r.region_type)
            for r in self.image_regions
        ]
        table_regions = [
            Region(bbox=convert_bbox(r.bbox), region_type=r.region_type)
            for r in self.table_regions
        ]
        table_elements = [
            TableElement(html_content=te.html_content, bbox=convert_bbox(te.bbox))
            for te in self.table_elements
        ]
        return ParsedDocument(
            text_elements=text_elements,
            image_regions=image_regions,
            table_regions=table_regions,
            table_elements=table_elements,
            plain_text=self.plain_text,
        )


TEXT_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td"}


class QwenVLHTMLParser:
    """Parse QwenVL HTML output with data-bbox attributes."""

    @staticmethod
    def _clean_html(html: str) -> str:
        """Strip markdown code fences and escape < > inside LaTeX formula content."""
        html = re.sub(r'^```html\s*\n?', '', html.strip())
        html = re.sub(r'\n?```\s*$', '', html.strip())
        # LaTeX inside formula divs can contain raw < > (e.g. \prod_{i<j}) which
        # BeautifulSoup treats as HTML tags, truncating the formula text.
        # Escape only the LaTeX content between the inner <div> and </div> tags,
        # preserving the surrounding HTML structure.
        def _escape_formula(m: re.Match) -> str:
            return m.group(1) + m.group(2).replace('<', '&lt;').replace('>', '&gt;') + m.group(3)
        html = re.sub(
            r'(<div[^>]*class="(?:formula|equation)"[^>]*>(?:<img[^>]*/>\s*)?<div>)(.*?)(</div>\s*</div>)',
            _escape_formula,
            html,
            flags=re.DOTALL,
        )
        return html

    def parse(self, html: str) -> ParsedDocument:
        """Parse HTML and extract text elements, image regions, and table regions."""
        html = self._clean_html(html)
        soup = BeautifulSoup(html, "html.parser")
        text_elements: list[TextElement] = []
        image_regions: list[Region] = []
        table_regions: list[Region] = []
        table_elements: list[TableElement] = []
        processed_elements: set[int] = set()

        # Process image divs (div with class containing "image")
        for div in soup.find_all("div", class_="image"):
            if not isinstance(div, Tag):
                continue
            bbox_str = div.get("data-bbox")
            if not isinstance(bbox_str, str):
                continue
            bbox = self._parse_bbox(bbox_str)
            image_regions.append(Region(bbox=bbox, region_type="image"))
            for child in div.descendants:
                processed_elements.add(id(child))
            processed_elements.add(id(div))

        # Process div.formula and div.equation elements before standalone img tags,
        # so that <img> children inside formula divs are marked as processed and
        # not mistakenly added to image_regions by the img loop below.
        for div in soup.find_all("div", class_=["formula", "equation"]):
            if not isinstance(div, Tag) or id(div) in processed_elements:
                continue
            bbox_str = div.get("data-bbox")
            if not isinstance(bbox_str, str):
                continue
            text = div.get_text(strip=True)
            if not text:
                continue
            bbox = self._parse_bbox(bbox_str)
            text_elements.append(TextElement(text=text, bbox=bbox, tag="formula"))
            for child in div.descendants:
                processed_elements.add(id(child))
            processed_elements.add(id(div))

        # Process standalone img tags (not inside a div.image or div.formula)
        for img in soup.find_all("img"):
            if not isinstance(img, Tag) or id(img) in processed_elements:
                continue
            bbox_str = img.get("data-bbox")
            if not isinstance(bbox_str, str):
                continue
            bbox = self._parse_bbox(bbox_str)
            image_regions.append(Region(bbox=bbox, region_type="image"))
            processed_elements.add(id(img))

        # Process table tags
        for table in soup.find_all("table"):
            if not isinstance(table, Tag):
                continue
            bbox_str = table.get("data-bbox")
            if not isinstance(bbox_str, str):
                continue
            bbox = self._parse_bbox(bbox_str)
            table_regions.append(Region(bbox=bbox, region_type="table"))
            table_elements.append(TableElement(html_content=str(table), bbox=bbox))
            for child in table.descendants:
                processed_elements.add(id(child))
            processed_elements.add(id(table))

        # Process div.text elements (primary format used by Qwen3-VL)
        for div in soup.find_all("div", class_="text"):
            if not isinstance(div, Tag) or id(div) in processed_elements:
                continue
            bbox_str = div.get("data-bbox")
            if not isinstance(bbox_str, str):
                continue
            text = div.get_text(strip=True)
            if not text:
                continue
            bbox = self._parse_bbox(bbox_str)
            text_elements.append(TextElement(text=text, bbox=bbox, tag="div"))
            for child in div.descendants:
                processed_elements.add(id(child))
            processed_elements.add(id(div))

        # Process div.table elements (Qwen3-VL format)
        for div in soup.find_all("div", class_="table"):
            if not isinstance(div, Tag) or id(div) in processed_elements:
                continue
            bbox_str = div.get("data-bbox")
            if not isinstance(bbox_str, str):
                continue
            bbox = self._parse_bbox(bbox_str)
            table_regions.append(Region(bbox=bbox, region_type="table"))
            inner_table = div.find("table")
            html_content = str(inner_table) if inner_table else ""
            table_elements.append(TableElement(html_content=html_content, bbox=bbox))
            for child in div.descendants:
                processed_elements.add(id(child))
            processed_elements.add(id(div))

        # Process text tags in document order (fallback for non-div format)
        for el in soup.find_all(list(TEXT_TAGS)):
            if not isinstance(el, Tag) or id(el) in processed_elements:
                continue
            bbox_str = el.get("data-bbox")
            if not isinstance(bbox_str, str):
                continue
            text = el.get_text(strip=True)
            if not text:
                continue
            bbox = self._parse_bbox(bbox_str)
            text_elements.append(TextElement(text=text, bbox=bbox, tag=el.name))
            processed_elements.add(id(el))

        # Fallback: handle any <div data-bbox="..."> not yet processed.
        # Covers models that emit non-standard class names (e.g. Chinese class names).
        for div in soup.find_all("div"):
            if not isinstance(div, Tag) or id(div) in processed_elements:
                continue
            bbox_str = div.get("data-bbox")
            if not isinstance(bbox_str, str):
                continue
            bbox = self._parse_bbox(bbox_str)
            inner_table = div.find("table")
            if inner_table:
                table_regions.append(Region(bbox=bbox, region_type="table"))
                table_elements.append(TableElement(html_content=str(inner_table), bbox=bbox))
            else:
                text = div.get_text(strip=True)
                if text:
                    text_elements.append(TextElement(text=text, bbox=bbox, tag="div"))
            for child in div.descendants:
                processed_elements.add(id(child))
            processed_elements.add(id(div))

        plain_text = soup.get_text(separator=" ", strip=True)

        return ParsedDocument(
            text_elements=text_elements,
            image_regions=image_regions,
            table_regions=table_regions,
            table_elements=table_elements,
            plain_text=plain_text,
        )

    @staticmethod
    def _parse_bbox(bbox_str: str) -> tuple[int, int, int, int]:
        """Parse a space-separated bbox string into a tuple of ints."""
        parts = bbox_str.strip().split()
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
