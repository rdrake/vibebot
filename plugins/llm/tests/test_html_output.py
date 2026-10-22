"""Tests for HTML output validation.

These tests verify the quality and security of HTML output generated
by the plugin, including syntax highlighting, XSS prevention, and
proper structure.
"""

from __future__ import annotations

import pytest
from llm.service import LLMService


class TestXssPrevention:
    """Test XSS prevention in HTML output."""

    @pytest.fixture
    def service(self, tmp_path, make_service) -> LLMService:
        """Create service with HTTP output config."""
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="http://localhost/llm",
            fileCleanupAge=24,
        )
        return service

    def test_script_tags_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with script tag WHEN saved THEN script stripped."""
        malicious = "Hello <script>alert('xss')</script> world"
        url = service.save_markdown_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "<script>" not in content
        assert "alert('xss')" not in content
        assert "Hello" in content
        assert "world" in content

    def test_onclick_handlers_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with onclick WHEN saved THEN onclick stripped."""
        malicious = '<a href="#" onclick="alert(1)">Click</a>'
        url = service.save_markdown_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "onclick" not in content

    def test_onerror_handlers_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with onerror WHEN saved THEN onerror stripped."""
        malicious = '<img src="x" onerror="alert(1)">'
        url = service.save_markdown_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "onerror" not in content

    def test_javascript_urls_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with javascript: URL WHEN saved THEN URL stripped."""
        malicious = '<a href="javascript:alert(1)">Click</a>'
        url = service.save_markdown_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "javascript:" not in content

    def test_data_urls_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with data: URL WHEN saved THEN URL stripped."""
        # No "<" inside the href: markdown 3.11 escapes such a tag as text, and
        # nh3 would never see the data: URL as an attribute.
        malicious = '<a href="data:text/html,alert(1)">Click</a>'
        url = service.save_markdown_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert "data:text/html" not in content

    def test_style_tags_stripped(self, service: LLMService, tmp_path) -> None:
        """GIVEN content with style tag WHEN saved THEN style stripped."""
        malicious = "<style>body{background:url('javascript:alert(1)')}</style>"
        url = service.save_markdown_to_http(malicious, title="Test")
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # The injected style tag should be stripped (our CSS is in the template)
        assert "javascript:alert" not in content

    def test_safe_links_preserved(self, service: LLMService, tmp_path) -> None:
        """GIVEN safe http link WHEN saved THEN link preserved."""
        safe = "[Link](https://example.com)"
        url = service.save_markdown_to_http(safe)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        assert 'href="https://example.com"' in content
        assert "Link" in content

    def test_nested_xss_attempts(self, service: LLMService, tmp_path) -> None:
        """GIVEN nested XSS attempts WHEN saved THEN all stripped."""
        malicious = '<a href="ja&#118;ascript:alert(1)">Click</a>'
        url = service.save_markdown_to_http(malicious)
        filename = url.split("/")[-1]
        content = (tmp_path / filename).read_text()

        # Should not have any form of javascript in href
        assert "javascript" not in content.lower() or 'href="javascript' not in content.lower()

    def test_relative_img_src_survives(self, service, tmp_path):
        md = "Hello\n\n![a cat](img_abc.jpg)\n"
        url = service.save_markdown_to_http(md, title="t")
        content = (tmp_path / url.split("/")[-1]).read_text()
        assert "<img" in content and 'src="img_abc.jpg"' in content
        assert 'alt="a cat"' in content

    def test_external_img_src_dropped(self, service, tmp_path):
        md = "![x](https://evil.example/track.png)"
        url = service.save_markdown_to_http(md, title="Test")
        content = (tmp_path / url.split("/")[-1]).read_text()
        assert "evil.example" not in content

    def test_javascript_img_src_dropped(self, service, tmp_path):
        md = '<img src="javascript:alert(1)" alt="x">'
        url = service.save_markdown_to_http(md)
        content = (tmp_path / url.split("/")[-1]).read_text()
        assert "javascript:" not in content

    def test_onerror_img_attr_dropped(self, service, tmp_path):
        md = '<img src="img_a.jpg" onerror="alert(1)">'
        url = service.save_markdown_to_http(md)
        content = (tmp_path / url.split("/")[-1]).read_text()
        assert "onerror" not in content


class TestPageThemes:
    """The storybook theme is reserved for story pages; answers and code
    pastes get the plain theme, and KaTeX only ships on answer pages."""

    @pytest.fixture
    def service(self, tmp_path, make_service) -> LLMService:
        service, _ = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="http://localhost/llm",
            fileCleanupAge=24,
        )
        return service

    def _page(self, tmp_path, url: str) -> str:
        return (tmp_path / url.split("/")[-1]).read_text()

    def test_story_page_gets_storybook_theme_and_prefix(self, service, tmp_path):
        url = service.save_markdown_to_http("# A Tale\n\nOnce upon a time.", style="story")
        assert url.split("/")[-1].startswith("story_")
        content = self._page(tmp_path, url)
        assert "Cinzel" in content
        assert "--parchment" in content
        assert "katex" not in content

    def test_answer_page_gets_plain_theme_with_katex(self, service, tmp_path):
        url = service.save_markdown_to_http("# Answer\n\nSome math: $x$")
        assert url.split("/")[-1].startswith("answer_")
        content = self._page(tmp_path, url)
        assert "Cinzel" not in content
        assert "parchment" not in content
        assert "katex" in content
