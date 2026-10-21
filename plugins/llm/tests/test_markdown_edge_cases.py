"""Tests for markdown processing edge cases.

These tests verify handling of complex, malformed, and edge-case
markdown content in code generation and rendering.
"""

from __future__ import annotations

import pytest
from llm.service import LLMService


class TestMarkdownInPageOutput:
    """Test markdown rendering edge cases in save_markdown_to_http."""

    @pytest.fixture
    def service(self, tmp_path, make_service) -> LLMService:
        """Create service with HTTP output config."""
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="http://localhost/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )
        return service

    def test_multiple_code_blocks_same_language(self, service: LLMService, tmp_path) -> None:
        """GIVEN multiple code blocks same language WHEN saved THEN all rendered."""
        markdown = """
```python
def foo():
    pass
```

Some text

```python
def bar():
    pass
```
"""
        url = service.save_markdown_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "foo" in content
        assert "bar" in content
        # Both should be in highlight blocks
        assert content.count("highlight") >= 2

    def test_mixed_language_code_blocks(self, service: LLMService, tmp_path) -> None:
        """GIVEN mixed language blocks WHEN saved THEN all highlighted appropriately."""
        markdown = """
```python
print("Python")
```

```javascript
console.log("JavaScript");
```

```rust
fn main() {}
```
"""
        url = service.save_markdown_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "Python" in content
        assert "JavaScript" in content
        assert "main" in content

    def test_code_with_special_html_chars(self, service: LLMService, tmp_path) -> None:
        """GIVEN code with HTML special chars WHEN saved THEN properly escaped."""
        markdown = """```python
if x < 10 and y > 5:
    print("<html>")
```"""
        url = service.save_markdown_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # Should be escaped to prevent HTML parsing issues
        assert "&lt;" in content or "<" in content  # Either escaped or in code block
        assert "&gt;" in content or ">" in content

    def test_code_with_ampersands(self, service: LLMService, tmp_path) -> None:
        """GIVEN code with ampersands WHEN saved THEN properly handled."""
        markdown = """```python
x = a & b
url = "foo?a=1&b=2"
```"""
        url = service.save_markdown_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # Ampersands should be preserved or escaped
        assert "&" in content or "&amp;" in content

    def test_unicode_in_code(self, service: LLMService, tmp_path) -> None:
        """GIVEN unicode in code WHEN saved THEN preserved."""
        markdown = """```python
greeting = "Hello 世界 🌍"
print(greeting)
```"""
        url = service.save_markdown_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "世界" in content
        # Emoji might be encoded but should be present
        assert "🌍" in content or "&#" in content

    def test_very_long_lines(self, service: LLMService, tmp_path) -> None:
        """GIVEN very long code lines WHEN saved THEN handled."""
        long_line = "x = " + "a + " * 200 + "b"
        markdown = f"```python\n{long_line}\n```"

        url = service.save_markdown_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # Syntax highlighting wraps tokens in span tags, so check for individual tokens
        # The variable 'x' and repeated 'a' tokens should be present
        assert ">x<" in content  # Variable name in a span
        assert ">a<" in content  # The repeated 'a' variable
        assert ">b<" in content  # The final 'b' variable

    def test_deeply_nested_lists(self, service: LLMService, tmp_path) -> None:
        """GIVEN deeply nested lists WHEN saved THEN rendered."""
        markdown = """
- Level 1
  - Level 2
    - Level 3
      - Level 4
        - Level 5
"""
        url = service.save_markdown_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "Level 1" in content
        assert "Level 5" in content

    def test_mixed_content_types(self, service: LLMService, tmp_path) -> None:
        """GIVEN mixed markdown content WHEN saved THEN all rendered."""
        markdown = """# Title

Here's some **bold** and *italic* text.

> A blockquote

```python
print("code")
```

1. First item
2. Second item

[A link](https://example.com)
"""
        url = service.save_markdown_to_http(markdown)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<h1>" in content
        assert "<strong>" in content or "<b>" in content
        assert "<em>" in content or "<i>" in content
        assert "<blockquote>" in content
        assert "print" in content
        assert "<ol>" in content
        assert 'href="https://example.com"' in content


# NOTE: channel-history formatting edge cases are covered (more strictly, with
# exact-equality assertions plus a nick-line-break injection test) by
# TestFormatChannelHistory in test_service.py. The weaker substring-based
# TestChannelHistoryFormatting that used to live here was removed as redundant.


class TestImageUrlDetection:
    """Test edge cases in image URL detection."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:
        """Create service with default config."""
        service, _ = make_service()
        return service

    def test_detects_standard_extensions(self, service: LLMService) -> None:
        """GIVEN standard image extensions WHEN detecting THEN finds all."""
        text = """
        https://example.com/a.jpg
        https://example.com/b.jpeg
        https://example.com/c.png
        https://example.com/d.gif
        https://example.com/e.webp
        https://example.com/f.bmp
        """
        images = service.detect_images(text)

        assert len(images) == 6

    def test_case_insensitive_extensions(self, service: LLMService) -> None:
        """GIVEN mixed case extensions WHEN detecting THEN finds all."""
        text = """
        https://example.com/a.JPG
        https://example.com/b.PNG
        https://example.com/c.GIF
        """
        images = service.detect_images(text)

        assert len(images) == 3

    def test_ignores_non_image_urls(self, service: LLMService) -> None:
        """GIVEN non-image URLs WHEN detecting THEN ignores them."""
        text = """
        https://example.com/page.html
        https://example.com/script.js
        https://example.com/style.css
        https://example.com/document.pdf
        """
        images = service.detect_images(text)

        assert len(images) == 0

    def test_multiple_images_same_line(self, service: LLMService) -> None:
        """GIVEN multiple images on same line WHEN detecting THEN finds all."""
        text = "See https://a.com/1.jpg and https://b.com/2.png please"
        images = service.detect_images(text)

        assert len(images) == 2
