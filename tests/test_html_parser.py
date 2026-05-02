"""Tests for QwenVL HTML parser."""
from reference_free_ocr_metric.reconstruction.html_parser import (
    QwenVLHTMLParser,
    TableElement,
)

SAMPLE_HTML = '''<html><body>
<h1 data-bbox="100 50 500 80">Title Text</h1>
<p data-bbox="100 100 500 130">Paragraph text here.</p>
<div class="image" data-bbox="100 150 400 350">
  <img data-bbox="100 150 400 350"/>
</div>
<table data-bbox="100 400 500 600">
  <tr><td>Cell 1</td><td>Cell 2</td></tr>
</table>
</body></html>'''


def test_parse_text_elements():
    parser = QwenVLHTMLParser()
    result = parser.parse(SAMPLE_HTML)
    text_elements = result.text_elements
    assert len(text_elements) >= 2
    assert text_elements[0].text == "Title Text"
    assert text_elements[0].bbox == (100, 50, 500, 80)


def test_parse_image_regions():
    parser = QwenVLHTMLParser()
    result = parser.parse(SAMPLE_HTML)
    assert len(result.image_regions) >= 1
    assert result.image_regions[0].bbox == (100, 150, 400, 350)


def test_parse_table_regions():
    parser = QwenVLHTMLParser()
    result = parser.parse(SAMPLE_HTML)
    assert len(result.table_regions) >= 1
    assert result.table_regions[0].bbox == (100, 400, 500, 600)


def test_normalized_to_pixel():
    parser = QwenVLHTMLParser()
    result = parser.parse(SAMPLE_HTML)
    pixel_doc = result.to_pixel_coords(image_width=2000, image_height=1000)
    first = pixel_doc.text_elements[0]
    # 100/1000 * 2000 = 200, 50/1000 * 1000 = 50, etc.
    assert first.bbox == (200, 50, 1000, 80)


def test_empty_html():
    parser = QwenVLHTMLParser()
    result = parser.parse("<html><body></body></html>")
    assert len(result.text_elements) == 0
    assert len(result.image_regions) == 0
    assert len(result.table_regions) == 0


QWENVL_HTML = '''<html><body>
<div class="image" data-bbox="104 29 455 75"></div>
<div class="text" data-bbox="91 101 346 127">宇通客车 (600066)</div>
<div class="text" data-bbox="91 140 489 167">7月销量转正</div>
<div class="table" data-bbox="108 298 386 446"><table><tr><td>行业</td><td>汽车</td></tr></table></div>
<div class="formula" data-bbox="248 599 571 638"><img/><div>$$E=mc^2$$</div></div>
</body></html>'''


def test_parse_div_text_elements():
    parser = QwenVLHTMLParser()
    result = parser.parse(QWENVL_HTML)
    texts = [te.text for te in result.text_elements]
    assert "宇通客车 (600066)" in texts
    assert "7月销量转正" in texts
    assert len(result.text_elements) >= 2


def test_parse_div_table_with_bbox():
    parser = QwenVLHTMLParser()
    result = parser.parse(QWENVL_HTML)
    assert len(result.table_regions) >= 1
    assert result.table_regions[0].bbox == (108, 298, 386, 446)


CODE_FENCED_HTML = '''```html
<html><body>
<div class="text" data-bbox="91 101 346 127">Fenced text</div>
</body></html>
```'''


def test_clean_html_strips_code_fences():
    cleaned = QwenVLHTMLParser._clean_html(CODE_FENCED_HTML)
    assert "```" not in cleaned
    assert '<div class="text"' in cleaned


def test_parse_strips_code_fences():
    parser = QwenVLHTMLParser()
    result = parser.parse(CODE_FENCED_HTML)
    assert len(result.text_elements) >= 1
    assert result.text_elements[0].text == "Fenced text"


NO_BBOX_HTML = '''<html><body>
<p>This paragraph has no bounding box.</p>
<p data-bbox="100 100 500 130">This one does.</p>
<div class="formula"><div>$$x^2 + y^2 = z^2$$</div></div>
</body></html>'''


def test_plain_text_includes_all_text():
    parser = QwenVLHTMLParser()
    result = parser.parse(NO_BBOX_HTML)
    assert "This paragraph has no bounding box." in result.plain_text
    assert "This one does." in result.plain_text
    assert "x^2 + y^2 = z^2" in result.plain_text


def test_plain_text_empty_for_empty_html():
    parser = QwenVLHTMLParser()
    result = parser.parse("<html><body></body></html>")
    assert result.plain_text == ""


def test_parse_table_elements_html_content():
    parser = QwenVLHTMLParser()
    result = parser.parse(QWENVL_HTML)
    assert len(result.table_elements) >= 1
    elem = result.table_elements[0]
    assert isinstance(elem, TableElement)
    assert elem.bbox == (108, 298, 386, 446)
    assert "行業" in elem.html_content or "行业" in elem.html_content


def test_parse_table_elements_standalone():
    parser = QwenVLHTMLParser()
    result = parser.parse(SAMPLE_HTML)
    assert len(result.table_elements) >= 1
    assert "Cell 1" in result.table_elements[0].html_content


def test_table_elements_pixel_coords():
    parser = QwenVLHTMLParser()
    result = parser.parse(QWENVL_HTML)
    pixel = result.to_pixel_coords(image_width=2000, image_height=1000)
    assert len(pixel.table_elements) >= 1
    # 108/1000*2000=216, 298/1000*1000=298, 386/1000*2000=772, 446/1000*1000=446
    assert pixel.table_elements[0].bbox == (216, 298, 772, 446)
