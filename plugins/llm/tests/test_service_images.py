"""Service images: saving, download, generation, draw context/rewrite, SSRF protection."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import openai
import pytest
from llm import service as service_module

from .conftest import FAKE_PROVIDER_KEYS, make_completion_response

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


class TestImageSaving:
    """Tests for save_image_to_http and _save_image_bytes functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            httpRoot="/tmp/test_llm_images",
            httpUrlBase="https://example.com/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )

    # --- save_image_to_http tests ---

    def test_save_image_to_http_success(self, tmp_path: object) -> None:
        """GIVEN valid base64 image WHEN saving THEN returns URL."""
        import base64

        # Mock config to use temp directory
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
                "fileCleanupAge": 24,
                "fileCleanupMax": 1000,
            }.get(key)
        )

        # Create simple PNG-like data
        image_data = b"\x89PNG\r\n\x1a\n" + b"fake image data"
        b64_data = base64.b64encode(image_data).decode()

        result = self.service.save_image_to_http(b64_data)

        assert result is not None
        assert result.startswith("https://example.com/llm/img_")
        assert result.endswith(".png")

    def test_save_image_to_http_custom_extension(self, tmp_path: object) -> None:
        """GIVEN custom extension WHEN saving THEN uses that extension."""
        import base64

        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
                "fileCleanupAge": 24,
                "fileCleanupMax": 1000,
            }.get(key)
        )

        image_data = b"fake jpeg data"
        b64_data = base64.b64encode(image_data).decode()

        result = self.service.save_image_to_http(b64_data, extension="jpg")

        assert result is not None
        assert result.endswith(".jpg")

    def test_save_image_to_http_invalid_base64(self) -> None:
        """GIVEN invalid base64 WHEN saving THEN returns None and logs error."""
        result = self.service.save_image_to_http("not valid base64!!!")

        # Error is logged via service's own logger (not plugin.log)
        assert result is None

    # --- _save_image_bytes tests ---

    def test_save_image_bytes_success(self, tmp_path: object) -> None:
        """GIVEN valid image bytes WHEN saving THEN returns URL."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        image_data = b"\x89PNG\r\n\x1a\n" + b"fake image data"
        result = self.service._save_image_bytes(image_data)

        assert result is not None
        assert result.startswith("https://example.com/llm/img_")
        assert result.endswith(".png")

    def test_save_image_bytes_custom_extension(self, tmp_path: object) -> None:
        """GIVEN custom extension WHEN saving THEN uses that extension."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        result = self.service._save_image_bytes(b"fake jpeg data", extension="jpg")

        assert result is not None
        assert result.endswith(".jpg")

    def test_save_image_bytes_magic_bytes_override_extension(self, tmp_path: object) -> None:
        """GIVEN JPEG magic bytes but extension='png' WHEN saving THEN uses jpg."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        jpeg_data = b"\xff\xd8\xff\xe0" + b"fake jpeg payload"
        result = self.service._save_image_bytes(jpeg_data, extension="png")

        assert result is not None
        assert result.endswith(".jpg")

    # --- _detect_image_format tests ---

    def test_detect_image_format_png(self) -> None:
        """GIVEN PNG magic bytes THEN returns 'png'."""
        assert self.service._detect_image_format(b"\x89PNG\r\n\x1a\ndata") == "png"

    def test_detect_image_format_jpeg(self) -> None:
        """GIVEN JPEG magic bytes THEN returns 'jpg'."""
        assert self.service._detect_image_format(b"\xff\xd8\xff\xe0data") == "jpg"

    def test_detect_image_format_webp(self) -> None:
        """GIVEN WebP magic bytes THEN returns 'webp'."""
        assert self.service._detect_image_format(b"RIFF\x00\x00\x00\x00WEBPdata") == "webp"

    def test_detect_image_format_gif(self) -> None:
        """GIVEN GIF magic bytes THEN returns 'gif'."""
        assert self.service._detect_image_format(b"GIF89adata") == "gif"

    def test_detect_image_format_unknown(self) -> None:
        """GIVEN unknown bytes THEN returns None."""
        assert self.service._detect_image_format(b"unknown data") is None

    def test_convert_png_to_jpeg(self) -> None:
        """GIVEN a real PNG image WHEN converting THEN returns JPEG bytes."""
        from io import BytesIO

        from PIL import Image

        # Create a real 1x1 red PNG
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        jpeg_bytes, ext = self.service._convert_png_to_jpeg(png_bytes)
        assert ext == "jpg"
        assert jpeg_bytes[:3] == b"\xff\xd8\xff"

    def test_convert_png_to_jpeg_rgba(self) -> None:
        """GIVEN RGBA PNG WHEN converting THEN strips alpha and returns JPEG."""
        from io import BytesIO

        from PIL import Image

        img = Image.new("RGBA", (1, 1), color=(255, 0, 0, 128))
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        jpeg_bytes, ext = self.service._convert_png_to_jpeg(png_bytes)
        assert ext == "jpg"
        assert jpeg_bytes[:3] == b"\xff\xd8\xff"

    def test_convert_invalid_png_falls_back(self) -> None:
        """GIVEN invalid PNG data WHEN converting THEN falls back to original."""
        bad_data = b"\x89PNG\r\n\x1a\ngarbage"
        result_bytes, ext = self.service._convert_png_to_jpeg(bad_data)
        assert ext == "png"
        assert result_bytes == bad_data

    def test_save_real_png_becomes_jpeg(self, tmp_path: object) -> None:
        """GIVEN a real PNG image WHEN saving THEN file is saved as JPEG."""
        from io import BytesIO
        from pathlib import Path

        from PIL import Image

        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        img = Image.new("RGB", (1, 1), color=(0, 128, 255))
        buf = BytesIO()
        img.save(buf, format="PNG")

        result = self.service._save_image_bytes(buf.getvalue())
        assert result is not None
        assert result.endswith(".jpg")

        jpg_files = list(Path(str(tmp_path)).glob("img_*.jpg"))
        assert len(jpg_files) == 1

    def test_save_image_bytes_writes_file(self, tmp_path: object) -> None:
        """GIVEN image bytes WHEN saving THEN file exists on disk."""
        from pathlib import Path

        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key)
        )

        image_data = b"\x89PNG\r\n\x1a\nfake"
        self.service._save_image_bytes(image_data)

        png_files = list(Path(str(tmp_path)).glob("img_*.png"))
        assert len(png_files) == 1
        assert png_files[0].read_bytes() == image_data


class TestDownloadAndSaveImage:
    """Tests for _download_and_save_image functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            httpRoot="/tmp/test_llm_images",
            httpUrlBase="https://example.com/llm",
            drawTimeout=60,
            timeout=30,
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )

    def test_download_success(self) -> None:
        """GIVEN valid image URL WHEN downloading THEN returns local URL."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"\x89PNG\r\n\x1a\nfake"
        mock_resp.headers = {"Content-Type": "image/png"}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        mock_save = self.mocker.patch.object(
            self.service,
            "_save_image_bytes",
            return_value="https://example.com/llm/img_abc.png",
        )
        result = self.service._download_and_save_image("https://provider.com/img.png")

        assert result == "https://example.com/llm/img_abc.png"
        mock_save.assert_called_once_with(b"\x89PNG\r\n\x1a\nfake", "png")

    def test_download_jpeg_content_type(self) -> None:
        """GIVEN JPEG content type WHEN downloading THEN uses jpg extension."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"fake jpeg"
        mock_resp.headers = {"Content-Type": "image/jpeg"}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        mock_save = self.mocker.patch.object(self.service, "_save_image_bytes", return_value="url")
        self.service._download_and_save_image("https://provider.com/img")

        mock_save.assert_called_once_with(b"fake jpeg", "jpg")

    def test_download_infers_extension_from_url(self) -> None:
        """GIVEN no content type WHEN URL has extension THEN infers from URL."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"fake webp"
        mock_resp.headers = {"Content-Type": "application/octet-stream"}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        mock_save = self.mocker.patch.object(self.service, "_save_image_bytes", return_value="url")
        self.service._download_and_save_image("https://provider.com/img.webp")

        mock_save.assert_called_once_with(b"fake webp", "webp")

    def test_download_defaults_to_png(self) -> None:
        """GIVEN no content type and no URL extension WHEN downloading THEN defaults to png."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"mystery image"
        mock_resp.headers = {"Content-Type": ""}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        mock_save = self.mocker.patch.object(self.service, "_save_image_bytes", return_value="url")
        self.service._download_and_save_image("https://provider.com/generate?id=123")

        mock_save.assert_called_once_with(b"mystery image", "png")

    def test_download_too_large(self) -> None:
        """GIVEN image exceeds 20 MB WHEN downloading THEN returns None."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = b"x" * (20 * 1024 * 1024 + 1)
        mock_resp.headers = {"Content-Type": "image/png"}
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        result = self.service._download_and_save_image("https://provider.com/huge.png")

        assert result is None

    def test_download_network_error(self) -> None:
        """GIVEN network error WHEN downloading THEN returns None."""
        import urllib.error

        mock_opener = self.mocker.Mock()
        mock_opener.open.side_effect = urllib.error.URLError("connection refused")
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        result = self.service._download_and_save_image("https://provider.com/img.png")

        assert result is None

    def test_download_rejects_non_http_scheme(self) -> None:
        """GIVEN file:// URL WHEN downloading THEN refuses without fetching."""
        mock_build = self.mocker.patch("urllib.request.build_opener")
        result = self.service._download_and_save_image("file:///etc/passwd")

        assert result is None
        mock_build.assert_not_called()

    def test_download_rejects_loopback_literal(self) -> None:
        """GIVEN 127.0.0.1 URL WHEN downloading THEN refuses without fetching."""
        mock_build = self.mocker.patch("urllib.request.build_opener")
        result = self.service._download_and_save_image("http://127.0.0.1/img.png")

        assert result is None
        mock_build.assert_not_called()

    def test_download_rejects_private_literal(self) -> None:
        """GIVEN 192.168.x.x URL WHEN downloading THEN refuses without fetching."""
        mock_build = self.mocker.patch("urllib.request.build_opener")
        result = self.service._download_and_save_image("http://192.168.1.1/img.png")

        assert result is None
        mock_build.assert_not_called()

    def test_download_rejects_link_local_literal(self) -> None:
        """GIVEN 169.254.x.x URL WHEN downloading THEN refuses without fetching."""
        mock_build = self.mocker.patch("urllib.request.build_opener")
        result = self.service._download_and_save_image("http://169.254.169.254/latest")

        assert result is None
        mock_build.assert_not_called()

    def test_download_disables_redirects(self) -> None:
        """GIVEN download path WHEN building opener THEN installs a no-redirect handler."""
        import urllib.request

        captured: dict[str, object] = {}
        real_build = urllib.request.build_opener

        def capture_build(*handlers: object) -> object:
            captured["handlers"] = handlers
            return real_build(*handlers)

        self.mocker.patch("urllib.request.build_opener", side_effect=capture_build)
        # Force network call to bail without actually fetching
        self.mocker.patch.object(self.service, "_save_image_bytes", return_value=None)

        # We don't care if the open call fails — only that build_opener was
        # called with a HTTPRedirectHandler subclass that vetoes redirects.
        self.service._download_and_save_image("https://nonexistent.invalid/img.png")

        handlers = captured.get("handlers", ())
        assert any(
            isinstance(h, urllib.request.HTTPRedirectHandler)
            and h.redirect_request(None, None, None, None, None) is None  # type: ignore[arg-type]
            for h in handlers  # type: ignore[union-attr]
        )


class TestImageGenerationWithBase64:
    """Tests for image_generation with base64 handling and typing indicators."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            imageModel="gemini/imagen-4.0-generate-001",
            timeout=30,
            maxPromptLength=10000,
            httpRoot="/tmp/test",
            httpUrlBase="https://example.com/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
            drawAutoRewriteMax=0,
        )

    def _make_mock_irc(self, capabilities: set | None = None) -> Mock:
        """Create mock IRC with capability negotiation."""
        irc = self.mocker.Mock()
        irc.state = self.mocker.Mock()
        irc.state.capabilities_ack = capabilities or {"message-tags"}
        irc.queueMsg = self.mocker.Mock()
        return irc

    def _make_mock_msg(self, channel: str = "#test") -> Mock:
        """Create mock message."""
        msg = self.mocker.Mock()
        msg.args = (channel,)
        return msg

    def test_image_generation_with_url_response(self) -> None:
        """GIVEN provider returns URL WHEN generating THEN downloads and returns local URL."""
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="https://provider.com/image.png", b64_json=None)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        mock_download = self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_abc123.png",
        )
        result = self.service.image_generation("a cat")

        mock_download.assert_called_once_with("https://provider.com/image.png")
        assert result.content == "https://example.com/llm/img_abc123.png"

    def test_image_generation_url_download_failure_falls_back(self) -> None:
        """GIVEN provider returns URL and download fails WHEN generating THEN falls back to provider URL."""
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="https://provider.com/image.png", b64_json=None)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        result = self.service.image_generation("a cat")

        assert result.content == "https://provider.com/image.png"

    def test_image_generation_with_base64_response(self, tmp_path: object) -> None:
        """GIVEN provider returns base64 WHEN generating THEN saves and returns URL."""
        import base64

        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "imageModel": "gemini/imagen",
                "timeout": 30,
                "maxPromptLength": 10000,
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
                "fileCleanupAge": 24,
                "fileCleanupMax": 1000,
            }.get(key)
        )

        image_data = b"\x89PNG\r\n\x1a\nfake image"
        b64_data = base64.b64encode(image_data).decode()

        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url=None, b64_json=b64_data)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        result = self.service.image_generation("a cat")

        assert result.content.startswith("https://example.com/llm/img_")
        assert result.content.endswith(".png")

    def test_image_generation_sends_typing_indicator(self) -> None:
        """GIVEN irc context WHEN generating THEN sends typing indicators."""
        irc = self._make_mock_irc()
        msg = self._make_mock_msg()

        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="https://example.com/image.png", b64_json=None)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        self.service.image_generation("a cat", irc=irc, msg=msg)

        # Should have called queueMsg twice - once for active, once for done
        assert irc.queueMsg.call_count == 2

        # First call should be typing=active
        first_msg = irc.queueMsg.call_args_list[0][0][0]
        assert first_msg.server_tags == {"+typing": "active"}

        # Second call should be typing=done
        second_msg = irc.queueMsg.call_args_list[1][0][0]
        assert second_msg.server_tags == {"+typing": "done"}

    def test_image_generation_sends_done_on_error(self) -> None:
        """GIVEN error during generation WHEN generating THEN still sends done indicator."""
        irc = self._make_mock_irc()
        msg = self._make_mock_msg()

        self.mocker.patch(
            "llm.service.litellm.image_generation", side_effect=Exception("API error")
        )
        result = self.service.image_generation("a cat", irc=irc, msg=msg)

        assert "Error" in result.content

        # Should still send typing=done in finally block
        assert irc.queueMsg.call_count == 2
        second_msg = irc.queueMsg.call_args_list[1][0][0]
        assert second_msg.server_tags == {"+typing": "done"}

    def test_image_generation_no_data_in_response(self) -> None:
        """GIVEN empty response WHEN generating THEN returns content filter error."""
        mock_response = self.mocker.Mock()
        mock_response.data = []

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        result = self.service.image_generation("a cat")

        assert "No image generated" in result.content
        assert "content safety filters" in result.content

    def test_image_generation_without_irc_context(self) -> None:
        """GIVEN no irc context WHEN generating THEN works without typing indicators."""
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="https://example.com/image.png", b64_json=None)]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("a cat")

        assert result.content == "https://example.com/llm/img_local.png"


class TestCleanupWithImages:
    """Tests for _cleanup_old_files with image extensions."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service) -> None:
        """Set up test fixtures."""
        self.service, self.mock_plugin = make_service(
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )

    def test_cleanup_collects_image_files(self, tmp_path: object) -> None:
        """GIVEN image files exist WHEN cleanup runs THEN collects them."""
        from pathlib import Path

        # Create test files of various types
        (Path(str(tmp_path)) / "code_abc.html").write_text("code")
        (Path(str(tmp_path)) / "img_def.png").write_bytes(b"png")
        (Path(str(tmp_path)) / "img_ghi.jpg").write_bytes(b"jpg")
        (Path(str(tmp_path)) / "img_jkl.jpeg").write_bytes(b"jpeg")
        (Path(str(tmp_path)) / "img_mno.webp").write_bytes(b"webp")
        (Path(str(tmp_path)) / "img_pqr.gif").write_bytes(b"gif")
        (Path(str(tmp_path)) / "other.txt").write_text("ignored")

        # Set max_files to 0 to force cleanup of all
        self.service._cleanup_old_files(str(tmp_path), max_age_hours=0, max_files=0)

        # All recognized files should be deleted, txt should remain
        assert not (Path(str(tmp_path)) / "code_abc.html").exists()
        assert not (Path(str(tmp_path)) / "img_def.png").exists()
        assert not (Path(str(tmp_path)) / "img_ghi.jpg").exists()
        assert not (Path(str(tmp_path)) / "img_jkl.jpeg").exists()
        assert not (Path(str(tmp_path)) / "img_mno.webp").exists()
        assert not (Path(str(tmp_path)) / "img_pqr.gif").exists()
        assert (Path(str(tmp_path)) / "other.txt").exists()


class TestCleanupLock:
    """Test that _cleanup_old_files uses a lock for thread safety (Fix 5)."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service) -> None:
        """Set up test fixtures."""
        self.service, self.mock_plugin = make_service(
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )

    def test_cleanup_lock_exists(self) -> None:
        """GIVEN service WHEN initialized THEN _cleanup_lock exists."""
        assert hasattr(self.service, "_cleanup_lock")

    def test_cleanup_serializes_concurrent_calls(self, tmp_path: object) -> None:
        """GIVEN concurrent cleanup calls WHEN running THEN lock prevents races."""
        from pathlib import Path

        # Create a test file
        (Path(str(tmp_path)) / "img_test.png").write_bytes(b"png")

        errors: list[Exception] = []

        def run_cleanup() -> None:
            try:
                self.service._cleanup_old_files(str(tmp_path), max_age_hours=0, max_files=0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_cleanup) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestHTTPFileManagement:
    """Tests for HTTP file storage, URL generation, and cleanup."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_get_http_paths_localhost_fallback(self) -> None:
        """GIVEN no httpRoot/httpUrlBase and no publicUrl WHEN get_http_paths called THEN falls back to localhost with port."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, *a, **kw: {"httpRoot": "", "httpUrlBase": ""}.get(key, "")
        )
        mock_conf = self.mocker.patch("llm.service.conf")
        mock_conf.supybot.directories.data.web.dirize.return_value = "/tmp/web"
        mock_conf.supybot.servers.http.publicUrl.return_value = ""
        mock_conf.supybot.servers.http.port.return_value = 8080

        http_root, url_base = self.service.get_http_paths()

        assert http_root == "/tmp/web"
        assert "localhost:8080" in url_base

    def test_save_code_to_http_oserror_returns_none(self) -> None:
        """GIVEN mkdir raises OSError WHEN save_code_to_http called THEN returns None."""
        self.mocker.patch.object(
            self.service,
            "get_http_paths",
            return_value=("/nonexistent/path", "http://x"),
        )
        self.mocker.patch("llm.service.Path.mkdir", side_effect=OSError("disk full"))

        result = self.service.save_code_to_http("# hello world")

        assert result is None

    def test_cleanup_old_files_deletes_old_preserves_new(self, tmp_path: object) -> None:
        """GIVEN old and new files WHEN _cleanup_old_files called THEN deletes old, keeps new."""
        import os
        import time
        from pathlib import Path

        dir_path = Path(str(tmp_path))
        old_file = dir_path / "old_code.html"
        new_file = dir_path / "new_code.html"
        old_file.write_text("old")
        new_file.write_text("new")

        # Backdate old file by 25 hours
        old_mtime = time.time() - (25 * 3600)
        os.utime(str(old_file), (old_mtime, old_mtime))

        self.service._cleanup_old_files(str(dir_path), max_age_hours=24, max_files=100)

        assert not old_file.exists()
        assert new_file.exists()

    def test_cleanup_old_files_caps_recent_files(self, tmp_path: object) -> None:
        """GIVEN 5 recent files WHEN max_files=2 THEN only 2 newest remain."""
        import time
        from pathlib import Path

        dir_path = Path(str(tmp_path))
        files = []
        for i in range(5):
            f = dir_path / f"code_{i}.html"
            f.write_text(f"content {i}")
            # Stagger mtimes so ordering is deterministic
            import os

            mtime = time.time() - (10 * (4 - i))  # oldest first
            os.utime(str(f), (mtime, mtime))
            files.append(f)

        self.service._cleanup_old_files(str(dir_path), max_age_hours=9999, max_files=2)

        remaining = list(dir_path.glob("*.html"))
        assert len(remaining) == 2

    def test_cleanup_old_files_nonexistent_dir_no_error(self) -> None:
        """GIVEN nonexistent directory WHEN _cleanup_old_files called THEN no error raised."""
        self.service._cleanup_old_files("/nonexistent/path", max_age_hours=24, max_files=100)


class TestDrawContext:
    """Tests for context integration in image generation."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            imageModel="gemini/imagen",
            timeout=30,
            maxPromptLength=10000,
            drawAutoRewriteMax=0,
        )

    def test_image_generation_uses_raw_prompt(self) -> None:
        """GIVEN a prompt WHEN generating image THEN uses prompt as-is."""
        prompt_used = []

        def capture_prompt(**kwargs):
            prompt_used.append(kwargs.get("prompt", ""))
            mock_response = self.mocker.Mock()
            mock_response.data = [
                self.mocker.Mock(url="https://example.com/img.png", b64_json=None)
            ]
            return mock_response

        self.mocker.patch("llm.service.litellm.image_generation", side_effect=capture_prompt)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        self.service.image_generation("a sunset")

        assert prompt_used[0] == "a sunset"


class TestImageUrlSsrfProtection:
    """Tests for SSRF protection in image URL validation."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_blocks_localhost(self) -> None:
        """GIVEN localhost URL WHEN validated THEN rejected."""
        assert self.service.validate_image_url("http://localhost/image.png") is False
        assert self.service.validate_image_url("http://127.0.0.1/image.png") is False

    def test_blocks_private_ranges(self) -> None:
        """GIVEN private IP range URLs WHEN validated THEN rejected."""
        assert self.service.validate_image_url("http://192.168.1.1/image.png") is False
        assert self.service.validate_image_url("http://10.0.0.1/image.png") is False
        assert self.service.validate_image_url("http://172.16.0.1/image.png") is False

    def test_blocks_metadata_endpoints(self) -> None:
        """GIVEN cloud metadata endpoint WHEN validated THEN rejected."""
        assert self.service.validate_image_url("http://169.254.169.254/image.png") is False

    def test_allows_public_urls(self) -> None:
        """GIVEN public URL WHEN validated THEN accepted."""
        # Note: This test requires DNS resolution, so we mock the resolver
        self.mocker.patch.object(self.service, "_resolves_to_public", return_value=True)
        assert self.service.validate_image_url("https://example.com/image.png") is True

    def test_resolver_fails_closed(self) -> None:
        """GIVEN DNS resolution failure WHEN checking host THEN blocked."""
        assert (
            self.service._resolves_to_public(
                "http://definitely-not-a-valid-hostname-12345.invalid/x.png"
            )
            is False
        )


class TestDrawAutoRewrite:
    """Tests for automatic prompt rewriting on content safety failures."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            imageModel="vertex_ai/imagen-4.0-generate-001",
            assistantModel="gemini/gemini-flash-latest",
            timeout=30,
            maxPromptLength=10000,
            httpRoot="/tmp/test",
            httpUrlBase="https://example.com/llm",
            drawAutoRewriteMax=3,
        )
        self.config_values = {
            "imageModel": "vertex_ai/imagen-4.0-generate-001",
            "assistantModel": "gemini/gemini-flash-latest",
            "timeout": 30,
            "maxPromptLength": 10000,
            "httpRoot": "/tmp/test",
            "httpUrlBase": "https://example.com/llm",
            "drawAutoRewriteMax": 3,
        }

    def _make_success_response(self, url: str = "https://example.com/img.png") -> Mock:
        """Create a mock successful image generation response."""
        response = self.mocker.Mock()
        response.data = [self.mocker.Mock(url=url, b64_json=None)]
        response.usage = self.mocker.Mock(prompt_tokens=5, completion_tokens=0)
        return response

    def _make_empty_response(self) -> Mock:
        """Create a mock empty (content-blocked) image generation response."""
        response = self.mocker.Mock()
        response.data = []
        response.usage = self.mocker.Mock(prompt_tokens=5, completion_tokens=0)
        return response

    def _make_rewrite_response(self, rewritten: str = "a safe cat") -> Mock:
        """Create a mock completion response for prompt rewriting."""
        return make_completion_response(rewritten, prompt_tokens=20, completion_tokens=10)

    def test_auto_rewrite_on_empty_data_succeeds(self) -> None:
        """GIVEN empty response data WHEN auto-rewrite enabled THEN retries with rewritten prompt."""
        empty_resp = self._make_empty_response()
        success_resp = self._make_success_response()
        rewrite_resp = self._make_rewrite_response("a friendly cat")

        self.mocker.patch(
            "llm.service.litellm.image_generation", side_effect=[empty_resp, success_resp]
        )
        self.mocker.patch("llm.service.litellm.completion", return_value=rewrite_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("a dangerous cat")

        assert result.content == "https://example.com/llm/img_local.png"
        assert result.rewritten_prompt == "a friendly cat"

    def test_auto_rewrite_on_content_policy_error_succeeds(self) -> None:
        """GIVEN ContentPolicyViolationError WHEN auto-rewrite enabled THEN retries."""
        import litellm as litellm_module

        rewrite_resp = self._make_rewrite_response("a safe prompt")
        success_resp = self._make_success_response()

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[
                litellm_module.ContentPolicyViolationError(
                    message="blocked", model="imagen", llm_provider="vertex_ai"
                ),
                success_resp,
            ],
        )
        self.mocker.patch("llm.service.litellm.completion", return_value=rewrite_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("bad prompt")

        assert result.content == "https://example.com/llm/img_local.png"
        assert result.rewritten_prompt == "a safe prompt"

    def test_auto_rewrite_multiple_retries_succeeds_on_third(self) -> None:
        """GIVEN multiple blocks WHEN retrying THEN succeeds on later attempt."""
        empty_resp = self._make_empty_response()
        success_resp = self._make_success_response()
        rewrite1 = self._make_rewrite_response("rewrite v1")
        rewrite2 = self._make_rewrite_response("rewrite v2")

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[empty_resp, empty_resp, success_resp],
        )
        self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[rewrite1, rewrite2],
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("test prompt")

        assert result.content == "https://example.com/llm/img_local.png"
        assert result.rewritten_prompt == "rewrite v2"

    def test_auto_rewrite_exhausts_all_retries(self) -> None:
        """GIVEN all retries fail WHEN max reached THEN says how many rewordings failed."""
        empty_resp = self._make_empty_response()
        rewrite1 = self._make_rewrite_response("rewrite v1")
        rewrite2 = self._make_rewrite_response("rewrite v2")
        rewrite3 = self._make_rewrite_response("rewrite v3")

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[empty_resp, empty_resp, empty_resp, empty_resp],
        )
        self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[rewrite1, rewrite2, rewrite3],
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)
        result = self.service.image_generation("test prompt")

        assert "3 rewordings" in result.content
        assert "different subject" in result.content

    def test_auto_rewrite_disabled_when_max_zero(self) -> None:
        """GIVEN drawAutoRewriteMax=0 WHEN content blocked THEN no rewrite attempted."""
        self.config_values["drawAutoRewriteMax"] = 0
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: self.config_values.get(key)
        )
        empty_resp = self._make_empty_response()

        self.mocker.patch("llm.service.litellm.image_generation", return_value=empty_resp)
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service.image_generation("test prompt")

        assert "content safety filters" in result.content
        mock_completion.assert_not_called()

    def test_auto_rewrite_llm_failure_falls_back(self) -> None:
        """GIVEN rewrite LLM fails WHEN retrying THEN reports the block, not a count.

        No rewrite got made, so mentioning one -- or "0 rewordings" -- describes
        an internal counter rather than anything the user can act on.
        """
        empty_resp = self._make_empty_response()

        self.mocker.patch("llm.service.litellm.image_generation", return_value=empty_resp)
        self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=Exception("LLM unavailable"),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service.image_generation("test prompt")

        assert result.error is not None
        assert result.content in service_module._DRAW_BLOCKED_LINES

    def test_auto_rewrite_skipped_when_ask_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN no key for the rewrite model's provider WHEN blocked THEN skips rewrite.

        The rewriter runs on ``assistantModel`` (gemini/gemini-flash-latest), so
        GEMINI_API_KEY is the variable that gates it. ``imageModel`` here is a
        vertex_ai model, which is unmanaged and so unaffected either way.
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        empty_resp = self._make_empty_response()

        self.mocker.patch("llm.service.litellm.image_generation", return_value=empty_resp)
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service.image_generation("test prompt")

        assert result.error is not None
        assert result.content in service_module._DRAW_BLOCKED_LINES
        mock_completion.assert_not_called()

    def test_auto_rewrite_aggregates_costs(self) -> None:
        """GIVEN successful rewrite WHEN costs tracked THEN aggregated in result."""
        empty_resp = self._make_empty_response()
        success_resp = self._make_success_response()
        rewrite_resp = self._make_rewrite_response("safe prompt")

        self.mocker.patch(
            "llm.service.litellm.image_generation", side_effect=[empty_resp, success_resp]
        )
        self.mocker.patch("llm.service.litellm.completion", return_value=rewrite_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.005)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        result = self.service.image_generation("test prompt")

        # Should include both rewrite and generation costs
        assert result.prompt_tokens > 0
        assert result.cost > 0

    def test_non_content_error_does_not_trigger_rewrite(self) -> None:
        """GIVEN timeout error WHEN generating THEN no rewrite attempted."""
        import litellm as litellm_module

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm_module.Timeout(
                message="timed out", model="imagen", llm_provider="vertex_ai"
            ),
        )
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        result = self.service.image_generation("test prompt")

        assert "timed out" in result.content.lower()
        mock_completion.assert_not_called()

    def test_auth_error_does_not_trigger_rewrite(self) -> None:
        """GIVEN authentication error WHEN generating THEN no rewrite attempted."""
        import litellm as litellm_module

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm_module.AuthenticationError(
                message="invalid key", model="imagen", llm_provider="vertex_ai"
            ),
        )
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        result = self.service.image_generation("test prompt")

        assert "Invalid API key" in result.content
        mock_completion.assert_not_called()

    def test_prior_rewrites_passed_to_subsequent_attempts(self) -> None:
        """GIVEN multiple rewrite attempts WHEN calling rewriter THEN prior history passed."""
        self.config_values["drawAutoRewriteMax"] = 2
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: self.config_values.get(key)
        )
        empty_resp = self._make_empty_response()
        rewrite1 = self._make_rewrite_response("rewrite v1")
        rewrite2 = self._make_rewrite_response("rewrite v2")

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[empty_resp, empty_resp, empty_resp],
        )
        mock_completion = self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[rewrite1, rewrite2],
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)
        self.service.image_generation("test prompt")

        # Second rewrite call should include prior_rewrites in the user message
        assert mock_completion.call_count == 2
        second_call_messages = mock_completion.call_args_list[1][1]["messages"]
        user_msg = second_call_messages[1]["content"]
        assert "rewrite v1" in user_msg
        assert "Previous rewrite attempts" in user_msg

    def test_rewritten_prompt_not_set_on_first_success(self) -> None:
        """GIVEN first attempt succeeds WHEN no rewrite needed THEN rewritten_prompt is None."""
        success_resp = self._make_success_response()

        self.mocker.patch("llm.service.litellm.image_generation", return_value=success_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)
        result = self.service.image_generation("a cat")

        assert result.rewritten_prompt is None

    def test_auto_rewrite_on_bad_request_moderation_blocked(self) -> None:
        """GIVEN BadRequestError with moderation_blocked WHEN auto-rewrite enabled THEN retries."""
        import litellm as litellm_module

        rewrite_resp = self._make_rewrite_response("a safe prompt")
        success_resp = self._make_success_response()

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[
                litellm_module.BadRequestError(
                    message=(
                        "OpenAIException - Error code: 400 - {'error': {'code': "
                        "'moderation_blocked'}}"
                    ),
                    model="imagen",
                    llm_provider="vertex_ai",
                ),
                success_resp,
            ],
        )
        self.mocker.patch("llm.service.litellm.completion", return_value=rewrite_resp)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)
        self.mocker.patch.object(
            self.service,
            "_download_and_save_image",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("bad prompt")

        assert result.content == "https://example.com/llm/img_local.png"
        assert result.rewritten_prompt == "a safe prompt"

    def test_non_moderation_bad_request_does_not_trigger_rewrite(self) -> None:
        """GIVEN BadRequestError without moderation keywords WHEN generating THEN no rewrite."""
        import litellm as litellm_module

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm_module.BadRequestError(
                message="Invalid image size parameter",
                model="imagen",
                llm_provider="vertex_ai",
            ),
        )
        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        result = self.service.image_generation("test prompt")

        assert "Error" in result.content
        mock_completion.assert_not_called()


class TestImageGenerationValidation:
    """Tests for image_generation() early validation error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.make_service = make_service
        self.service, self.mock_plugin = make_service()

    def test_image_generation_invalid_prompt(self) -> None:
        """GIVEN empty prompt WHEN image_generation called THEN returns error in result."""
        result = self.service.image_generation("")

        assert result.error is not None
        assert "Error" in result.content

    def test_image_generation_missing_draw_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN no key for the image model's provider THEN returns an error.

        The default ``imageModel`` is "dall-e-3", which LiteLLM places on
        openai, so OPENAI_API_KEY is what this guard reads.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        service, _ = self.make_service()

        result = service.image_generation("A beautiful sunset")

        assert result.error is not None
        assert "OPENAI_API_KEY" in result.content


class TestImageGenerationPaths:
    """Tests for image generation rewrite loop and error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            imageModel="dall-e-3",
            assistantModel="gemini/gemini-flash-latest",
            timeout=30,
            drawTimeout=30,
            maxPromptLength=10000,
            httpRoot="/tmp/test",
            httpUrlBase="https://example.com/llm",
            drawAutoRewriteMax=3,
        )

    def test_rewrite_empty_response(self) -> None:
        """GIVEN LLM returns empty content WHEN _rewrite_prompt_for_safety called THEN returns None tuple."""
        response = self.mocker.Mock()
        response.choices = [self.mocker.Mock(message=self.mocker.Mock(content=""))]

        self.mocker.patch("llm.service.litellm.completion", return_value=response)

        result = self.service._rewrite_prompt_for_safety("bad prompt", "blocked", [], "#chan")

        assert result == (None, 0, 0, 0.0)

    def test_xai_model_kwargs(self) -> None:
        """GIVEN xai model WHEN _attempt_image_generation called THEN passes extra kwargs."""
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="http://img.png", b64_json=None)]

        mock_img_gen = self.mocker.patch(
            "llm.service.litellm.image_generation", return_value=mock_response
        )
        self.mocker.patch.object(self.service, "_extract_usage", return_value=(0, 0, 0.0))
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)

        self.service._attempt_image_generation("cat", "xai/grok-2-image", 30)

        call_kwargs = mock_img_gen.call_args
        assert call_kwargs[1]["quality"] == "high"
        assert call_kwargs[1]["resolution"] == "2k"
        # No aspect_ratio: xAI's default is `auto`, the best ratio for the
        # prompt. A pinned 9:16 made every draw portrait.
        assert "aspect_ratio" not in call_kwargs[1]

    @pytest.mark.parametrize(
        ("aspect", "ratio"),
        [("portrait", "2:3"), ("landscape", "3:2"), ("square", "1:1")],
    )
    def test_xai_aspect_maps_to_a_ratio(self, aspect: str, ratio: str) -> None:
        """GIVEN the planner chose a shape WHEN drawing on xAI THEN it rides as aspect_ratio.

        `auto` reads the prompt as a scene: a 1950s cinema poster came back
        16:9 (2026-09-19). The planner knows it is a poster; it says so.
        """
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="http://img.png", b64_json=None)]
        mock_img_gen = self.mocker.patch(
            "llm.service.litellm.image_generation", return_value=mock_response
        )
        self.mocker.patch.object(self.service, "_extract_usage", return_value=(0, 0, 0.0))
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)

        self.service._attempt_image_generation("cat", "xai/grok-2-image", 30, aspect=aspect)

        assert mock_img_gen.call_args[1]["aspect_ratio"] == ratio

    def test_unknown_aspect_is_ignored(self) -> None:
        """GIVEN a shape the map does not know WHEN drawing THEN xAI keeps `auto`."""
        mock_response = self.mocker.Mock()
        mock_response.data = [self.mocker.Mock(url="http://img.png", b64_json=None)]
        mock_img_gen = self.mocker.patch(
            "llm.service.litellm.image_generation", return_value=mock_response
        )
        self.mocker.patch.object(self.service, "_extract_usage", return_value=(0, 0, 0.0))
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)

        self.service._attempt_image_generation("cat", "xai/grok-2-image", 30, aspect="wide")

        assert "aspect_ratio" not in mock_img_gen.call_args[1]

    def test_image_generation_threads_aspect_to_the_attempt(self) -> None:
        """GIVEN an aspect WHEN image_generation runs THEN the attempt receives it."""
        from llm.service import ImageResult

        attempt = self.mocker.patch.object(
            self.service,
            "_attempt_image_generation",
            return_value=ImageResult(content="http://img.png"),
        )
        self.mocker.patch.object(self.service, "validate_prompt", return_value=(True, ""))
        self.mocker.patch.object(self.service, "_missing_image_key_error", return_value=None)

        self.service.image_generation("a poster", aspect="portrait")

        assert attempt.call_args.kwargs["aspect"] == "portrait"

    def test_b64_json_save_failure(self) -> None:
        """GIVEN b64_json data but save fails WHEN _attempt_image_generation called THEN returns error."""
        image_data = self.mocker.Mock()
        image_data.url = None
        image_data.b64_json = "base64data"

        mock_response = self.mocker.Mock()
        mock_response.data = [image_data]

        self.mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        self.mocker.patch.object(self.service, "_extract_usage", return_value=(0, 0, 0.0))
        self.mocker.patch.object(self.service, "save_image_to_http", return_value=None)

        result = self.service._attempt_image_generation("cat", "dall-e-3", 30)

        assert result is not None
        assert result.error is not None

    def test_timeout_not_stashed(self) -> None:
        """GIVEN image_generation times out and stashing fails WHEN called THEN returns error."""
        import litellm as litellm_module

        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm_module.Timeout(
                message="Request timed out", model="dall-e-3", llm_provider="openai"
            ),
        )
        self.mocker.patch.object(self.service, "_stash_timeout", return_value=False)

        result = self.service.image_generation("a cat")

        assert result.error is not None

    def test_non_content_error_in_rewrite_loop(self) -> None:
        """GIVEN first attempt blocked and retry raises non-content error WHEN generating THEN returns error."""
        self.mocker.patch.object(
            self.service,
            "_attempt_image_generation",
            side_effect=[None, RuntimeError("network")],
        )
        self.mocker.patch.object(
            self.service,
            "_rewrite_prompt_for_safety",
            return_value=("rewritten", 10, 5, 0.01),
        )
        self.mocker.patch.object(
            self.service,
            "_is_content_safety_error",
            return_value=False,
        )

        result = self.service.image_generation("a cat")

        assert result.error is not None

    def test_outer_exception_handler(self) -> None:
        """GIVEN unexpected error in validate_prompt WHEN image_generation called THEN returns graceful error."""
        self.mocker.patch.object(
            self.service,
            "validate_prompt",
            side_effect=RuntimeError("unexpected"),
        )

        result = self.service.image_generation("a cat")

        assert result.error is not None


class TestRetryImage:
    """Tests for _retry_image error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up service with mock plugin."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def _make_task(self, **overrides):
        """Create a PendingTaskRow with draw defaults."""
        from llm.persistence import PendingTaskRow

        defaults = {
            "id": 1,
            "task_type": "draw",
            "nick": "user",
            "reply_target": "#chan",
            "is_channel": 1,
            "prompt_preview": "test",
            "model": "dall-e-3",
            "request_data": '{"prompt": "cat"}',
            "submitted_at": 100.0,
            "expires_at": 200.0,
            "attempt_count": 0,
            "next_attempt_at": 100.0,
            "claimed_until": 0,
            "last_error": "",
            "delivery_state": "pending",
            "result_payload": "",
            "last_delivery_error": "",
            "delivery_attempt_count": 0,
            "origin_request_id": "",
            "account": None,
        }
        defaults.update(overrides)
        return PendingTaskRow(**defaults)

    def test_retry_image_malformed_data(self) -> None:
        """GIVEN request_data missing prompt key WHEN _retry_image called THEN returns failed_terminal with Malformed reason."""
        task = self._make_task()

        result = self.service._retry_image(task, {"not_prompt": "x"})

        assert result.status == "failed_terminal"
        assert "Malformed" in result.reason

    def test_retry_image_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN no key for the stashed model's provider THEN failed_terminal.

        The retry resolves from ``task.model`` ("dall-e-3" -> openai), not from
        a per-command registry key that could name a different provider.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        task = self._make_task()

        result = self.service._retry_image(task, {"prompt": "cat"})

        assert result.status == "failed_terminal"
        assert "OPENAI_API_KEY" in result.reason

    def test_retry_image_carries_the_aspect(self) -> None:
        """GIVEN a stashed draw with an aspect WHEN retried THEN the shape survives the stash."""
        from llm.service import ImageResult

        task = self._make_task(model="xai/grok-imagine-image")
        self.mocker.patch.object(self.service, "_missing_image_key_error", return_value=None)
        attempt = self.mocker.patch.object(
            self.service,
            "_attempt_image_generation",
            return_value=ImageResult(content="http://img.png"),
        )

        self.service._retry_image(task, {"prompt": "cat", "aspect": "portrait"})

        assert attempt.call_args.kwargs["aspect"] == "portrait"

    def test_retry_image_content_blocked(self) -> None:
        """GIVEN _attempt_image_generation returns None WHEN _retry_image called THEN returns failed_terminal with blocked reason."""
        task = self._make_task()
        self.mocker.patch.object(self.service, "_attempt_image_generation", return_value=None)

        result = self.service._retry_image(task, {"prompt": "cat"})

        assert result.status == "failed_terminal"
        assert "blocked" in result.reason


class TestExternalImageUpload:
    """Tests for uploading generated images to an external host (imageUploadUrl)."""

    UPLOAD_URL = "https://paste.example.com/img/"

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture, tmp_path) -> None:
        """Set up a service whose images upload to an external host."""
        self.mocker = mocker
        self.tmp_path = tmp_path
        self.service, self.mock_plugin = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="https://example.com/llm",
            imageUploadUrl=self.UPLOAD_URL,
            drawTimeout=60,
            timeout=30,
        )

    def _mock_upload_reply(self, payload: str) -> Mock:
        """Patch the opener so an upload POST returns ``payload``."""
        mock_resp = self.mocker.Mock()
        mock_resp.read.return_value = payload.encode()
        mock_resp.__enter__ = self.mocker.Mock(return_value=mock_resp)
        mock_resp.__exit__ = self.mocker.Mock(return_value=False)

        mock_opener = self.mocker.Mock()
        mock_opener.open.return_value = mock_resp
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)
        return mock_opener

    def _local_files(self) -> list[str]:
        """Names of image files written to the local HTTP root."""
        return [p.name for p in self.tmp_path.iterdir()]

    def test_upload_success_returns_remote_url(self) -> None:
        """GIVEN the host accepts the image WHEN saving THEN returns its URL and writes nothing locally."""
        self._mock_upload_reply(
            '{"results": [{"success": true, "filePath": "/img/img_abc123.png"}]}'
        )

        result = self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        assert result == "https://paste.example.com/img/img_abc123.png"
        assert self._local_files() == []

    def test_upload_posts_multipart_images_field(self) -> None:
        """GIVEN an image WHEN uploading THEN posts multipart on the images[] field."""
        mock_opener = self._mock_upload_reply(
            '{"results": [{"success": true, "filePath": "/img/x.png"}]}'
        )

        self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        req = mock_opener.open.call_args[0][0]
        assert req.full_url == self.UPLOAD_URL
        assert req.get_header("Content-type").startswith("multipart/form-data; boundary=")
        assert b'name="images[]"' in req.data
        assert b'name="strip_exif"' in req.data
        assert b"\x89PNG\r\n\x1a\nfake image data" in req.data

    def test_upload_rejected_falls_back_to_local(self) -> None:
        """GIVEN the host rejects the upload WHEN saving THEN falls back to local storage."""
        self._mock_upload_reply('{"results": [{"success": false, "error": "Unsupported type"}]}')

        result = self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        assert result is not None
        assert result.startswith("https://example.com/llm/img_")
        assert len(self._local_files()) == 1

    def test_upload_network_error_falls_back_to_local(self) -> None:
        """GIVEN the host is unreachable WHEN saving THEN falls back to local storage."""
        import urllib.error

        mock_opener = self.mocker.Mock()
        mock_opener.open.side_effect = urllib.error.URLError("connection refused")
        self.mocker.patch("urllib.request.build_opener", return_value=mock_opener)

        result = self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        assert result.startswith("https://example.com/llm/img_")

    def test_upload_reply_on_other_host_rejected(self) -> None:
        """GIVEN the reply points at another host WHEN saving THEN the URL is not used."""
        self._mock_upload_reply(
            '{"results": [{"success": true, "filePath": "https://evil.example.net/x.png"}]}'
        )

        result = self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        assert result.startswith("https://example.com/llm/img_")

    def test_upload_reply_with_non_image_path_rejected(self) -> None:
        """GIVEN the reply names a non-image path WHEN saving THEN the URL is not used."""
        self._mock_upload_reply('{"results": [{"success": true, "filePath": "/img/payload.html"}]}')

        result = self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        assert result.startswith("https://example.com/llm/img_")

    def test_upload_malformed_reply_falls_back_to_local(self) -> None:
        """GIVEN the host replies with garbage WHEN saving THEN falls back to local storage."""
        self._mock_upload_reply("not json at all")

        result = self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        assert result.startswith("https://example.com/llm/img_")

    def test_oversize_image_skips_upload(self) -> None:
        """GIVEN an image over the upload limit WHEN saving THEN no request is made."""
        mock_opener = self._mock_upload_reply(
            '{"results": [{"success": true, "filePath": "/img/x.png"}]}'
        )
        huge = b"\x89PNG\r\n\x1a\n" + b"x" * (10 * 1024 * 1024)

        result = self.service._save_image_bytes(huge)

        mock_opener.open.assert_not_called()
        assert result.startswith("https://example.com/llm/img_")

    def test_unsafe_endpoint_skips_upload(self) -> None:
        """GIVEN a non-http upload endpoint WHEN saving THEN no request is made."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(self.tmp_path),
                "httpUrlBase": "https://example.com/llm",
                "imageUploadUrl": "file:///etc/passwd",
            }.get(key, "")
        )
        mock_opener = self._mock_upload_reply(
            '{"results": [{"success": true, "filePath": "/img/x.png"}]}'
        )

        result = self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        mock_opener.open.assert_not_called()
        assert result.startswith("https://example.com/llm/img_")

    def test_uploads_disabled_stores_locally(self) -> None:
        """GIVEN imageUploadUrl is empty WHEN saving THEN stores locally without a request."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpRoot": str(self.tmp_path),
                "httpUrlBase": "https://example.com/llm",
            }.get(key, "")
        )
        mock_opener = self._mock_upload_reply(
            '{"results": [{"success": true, "filePath": "/img/x.png"}]}'
        )

        result = self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        mock_opener.open.assert_not_called()
        assert result.startswith("https://example.com/llm/img_")

    # --- page embedding ---

    def test_page_image_ref_keeps_remote_url_absolute(self) -> None:
        """GIVEN an externally-hosted image WHEN embedding THEN keeps the absolute URL."""
        url = "https://paste.example.com/img/img_abc.png"

        assert self.service._page_image_ref(url) == url

    def test_page_image_ref_shortens_local_url(self) -> None:
        """GIVEN a locally-hosted image WHEN embedding THEN uses the bare filename."""
        assert self.service._page_image_ref("https://example.com/llm/img_abc.png") == "img_abc.png"

    def test_restrict_img_srcs_allows_upload_host(self) -> None:
        """GIVEN images on the upload host WHEN rendering a page THEN they survive."""
        html = (
            '<img src="https://paste.example.com/img/img_abc.png">'
            '<img src="img_local.png">'
            '<img src="https://tracker.example.net/pixel.png">'
        )

        out = self.service._restrict_img_srcs(html, "https://example.com/llm")

        assert "paste.example.com/img/img_abc.png" in out
        assert "img_local.png" in out
        assert "tracker.example.net" not in out

    def test_restrict_img_srcs_drops_upload_host_when_disabled(self) -> None:
        """GIVEN uploads are off WHEN rendering a page THEN foreign-host images are dropped."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "httpUrlBase": "https://example.com/llm",
            }.get(key, "")
        )
        html = '<img src="https://paste.example.com/img/img_abc.png">'

        assert self.service._restrict_img_srcs(html, "https://example.com/llm") == ""


class TestImageBoundaryKeyResolution:
    """The key litellm.image_generation receives follows the image model.

    Draw is the one outbound boundary whose kwargs nothing else asserts on:
    before these tests, deleting ``api_key=apikeys.api_key_for(model)`` from the
    ``litellm.image_generation`` call left the whole suite green.
    """

    @pytest.mark.parametrize(
        ("image_model", "expected_env"),
        [
            ("xai/grok-2-image", "XAI_API_KEY"),
            ("dall-e-3", "OPENAI_API_KEY"),
            ("gemini/imagen-4.0-generate-001", "GEMINI_API_KEY"),
        ],
    )
    def test_managed_image_model_gets_its_own_provider_key(
        self, make_service, mocker: MockerFixture, image_model: str, expected_env: str
    ) -> None:
        """GIVEN an image model WHEN drawing THEN that provider's variable is sent."""
        service, _ = make_service(imageModel=image_model, drawAutoRewriteMax=0)
        response = mocker.Mock()
        response.data = [mocker.Mock(url="https://provider.com/image.png", b64_json=None)]
        image_generation = mocker.patch(
            "llm.service.litellm.image_generation", return_value=response
        )
        mocker.patch.object(service, "_download_and_save_image", return_value=None)

        service.image_generation("a cat")

        kwargs = image_generation.call_args.kwargs
        assert kwargs["model"] == image_model
        assert kwargs["api_key"] == FAKE_PROVIDER_KEYS[expected_env]

    def test_unmanaged_image_model_passes_none(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN a vertex_ai image model WHEN drawing THEN api_key is None.

        vertex_ai is unmanaged regardless of what imageModel is configured to
        (shipped default is gemini; prod sets xai/grok-imagine-image
        explicitly) — None is what makes LiteLLM fall back to ADC for it, and
        a mapped key here would break service-account auth on the draw path.
        """
        service, _ = make_service(
            imageModel="vertex_ai/imagen-4.0-generate-001", drawAutoRewriteMax=0
        )
        response = mocker.Mock()
        response.data = [mocker.Mock(url="https://provider.com/image.png", b64_json=None)]
        image_generation = mocker.patch(
            "llm.service.litellm.image_generation", return_value=response
        )
        mocker.patch.object(service, "_download_and_save_image", return_value=None)

        service.image_generation("a cat")

        assert image_generation.call_args.kwargs["api_key"] is None


class TestXaiModerationArmsTheRewrite:
    """The auto-rewrite loop must fire on xAI's moderation refusal.

    It did not, for as long as prod has been drawing on grok-imagine.
    ``_is_content_safety_error`` knew OpenAI's and Google's phrasing only, so
    ``imagine:content-moderated`` was classified as an ordinary bad request and
    ``image_generation`` returned without retrying. Measured over six hours on
    2026-08-15: 18 draws, 10 refused, zero rewrites attempted, with
    ``drawAutoRewriteMax`` set to 3 the whole time.
    """

    # Verbatim from prod, 2026-08-15T22:25:59Z. Pinned rather than paraphrased:
    # the whole defect was a keyword list that did not match reality, so the
    # test is only worth anything if it carries the real string.
    XAI_MODERATION_ERROR = (
        "litellm.BadRequestError: XaiException - Error code: 400 - "
        "{'code': 'imagine:content-moderated', 'error': 'Generated image "
        "rejected by content moderation.', 'usage': {'cost_in_usd_ticks': 200000000}}"
    )

    def _xai_error(self) -> Exception:
        import litellm as litellm_module

        return litellm_module.BadRequestError(
            message=self.XAI_MODERATION_ERROR,
            model="xai/grok-imagine-image",
            llm_provider="xai",
        )

    def test_classifier_recognises_the_refusal(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """The exact prod error is a content block, not a generic bad request."""
        service, _ = make_service()
        assert service._is_content_safety_error(self._xai_error()) is True

    def test_a_real_bad_request_is_still_not_a_content_block(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """Widening the list must not swallow ordinary 400s.

        A malformed request has to keep returning immediately: retrying it
        spends a rewrite completion and another billed image call on something
        no wording change can fix.
        """
        import litellm as litellm_module

        service, _ = make_service()
        error = litellm_module.BadRequestError(
            message="litellm.BadRequestError: XaiException - Error code: 400 - "
            "{'error': 'Invalid value for parameter n'}",
            model="xai/grok-imagine-image",
            llm_provider="xai",
        )
        assert service._is_content_safety_error(error) is False

    def test_refusal_now_gets_one_rewrite_and_recovers(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """End to end: refused draw -> rewritten prompt -> image delivered."""
        service, _ = make_service(
            imageModel="xai/grok-imagine-image",
            assistantModel="gemini/gemini-flash-latest",
            drawAutoRewriteMax=1,
            httpUrlBase="https://example.com/llm",
        )
        success = mocker.Mock()
        success.data = [mocker.Mock(url="https://provider.com/image.png", b64_json=None)]
        success.usage = mocker.Mock(prompt_tokens=5, completion_tokens=0)

        image_generation = mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[self._xai_error(), success],
        )
        mocker.patch(
            "llm.service.litellm.completion",
            return_value=make_completion_response("a tasteful party scene"),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        mocker.patch.object(
            service, "_download_and_save_image", return_value="https://example.com/llm/img_a.png"
        )

        result = service.image_generation("bunga bunga party")

        assert result.error is None
        assert result.url == "https://example.com/llm/img_a.png"
        assert result.rewritten_prompt == "a tasteful party scene"
        assert image_generation.call_count == 2

    def test_the_cap_bounds_the_billed_calls(self, make_service, mocker: MockerFixture) -> None:
        """A prompt the provider will never pass costs one extra call, not three.

        Every rewrite spends a billed image call on top of the one already
        refused, which is why the default came down from 3 to 1.
        """
        service, _ = make_service(
            imageModel="xai/grok-imagine-image",
            assistantModel="gemini/gemini-flash-latest",
            drawAutoRewriteMax=1,
            httpUrlBase="https://example.com/llm",
        )
        image_generation = mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[self._xai_error(), self._xai_error()],
        )
        mocker.patch(
            "llm.service.litellm.completion",
            return_value=make_completion_response("something milder"),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.image_generation("bunga bunga party")

        assert result.error is not None
        assert image_generation.call_count == 2


class TestRewriteFidelity:
    """The rewrite has to come back as the same picture, said another way.

    Two failure modes, opposite directions. Timid: the rewrite is the refused
    prompt with one synonym swapped, the filter refuses it again, and with
    ``drawAutoRewriteMax`` at 1 in prod that was the only shot. Loose: it drifts
    off the subject, and a redraw that quietly becomes a different picture reads
    to the channel as the bot ignoring the request.

    The user's intent is what is fixed; their wording is not. So these tests
    hold the instruction to "keep the subject, change the wording as far as you
    need to" and pin the padding tolerance — not a length limit, because saying
    a thing another way is sometimes wordier.
    """

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            imageModel="xai/grok-imagine-image",
            assistantModel="gemini/gemini-flash-latest",
            httpUrlBase="https://example.com/llm",
            drawAutoRewriteMax=1,
        )

    def _rewrite(self, returned: str, original: str = "a cat on fire in a burning house"):
        """Run one rewrite, return the (messages, result) the call produced."""
        response = self.mocker.Mock()
        response.choices = [self.mocker.Mock(message=self.mocker.Mock(content=returned))]
        response.usage = self.mocker.Mock(prompt_tokens=10, completion_tokens=5)
        completion = self.mocker.patch("llm.service.litellm.completion", return_value=response)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service._rewrite_prompt_for_safety(original, "content moderation", [], None)
        return completion.call_args[1]["messages"], result

    def test_rewriter_is_told_to_keep_the_subject_and_change_the_wording(self) -> None:
        """Both halves have to be in there, or it fails in one direction.

        "Keep" alone produces a rewrite the filter refuses again; "change"
        alone produces a different picture.
        """
        messages, _ = self._rewrite("a cat in a house")
        system = messages[0]["content"].lower()

        assert "keep:" in system
        assert "the subject" in system
        assert "change:" in system
        assert "as far as you need to" in system

    def test_rewriter_is_allowed_to_go_as_far_as_the_filter_requires(self) -> None:
        """The escape hatch: a word that cannot be softened gets said differently.

        Without this the rewriter treats an unsoftenable word as a dead end and
        hands back something the filter has already refused.
        """
        messages, _ = self._rewrite("a cat in a house")
        system = messages[0]["content"].lower()

        assert "different angle" in system
        assert "another way to describe" in system

    def test_rewriter_is_told_not_to_pad(self) -> None:
        """The one thing it may not do with its freedom.

        The drift this guards is style and quality words accreting onto a
        picture the user never asked for them on.
        """
        messages, _ = self._rewrite("a cat in a house")
        system = messages[0]["content"].lower()

        assert "do not" in system
        assert "pad the prompt" in system

    def test_rewriter_is_given_the_original_length_as_a_target(self) -> None:
        """A concrete number is checkable by the model; "keep it short" is not."""
        original = "a cat on fire in a burning house"
        messages, _ = self._rewrite("a cat in a house", original=original)

        assert str(len(original)) in messages[1]["content"]

    def test_rewriter_is_briefed_on_false_positives_not_evasion(self) -> None:
        """It is re-reading a misfire, and it is told so.

        The prompts this fires on are benign ones the filter misread. A
        rewriter briefed as though it were smuggling writes like one, and the
        instruction not to dress up a genuinely refusable request is what keeps
        the loop pointed at false positives.
        """
        messages, _ = self._rewrite("a cat in a house")
        system = messages[0]["content"].lower()

        assert "misread something" in system
        assert "genuinely should stay refused" in system

    def test_a_padded_rewrite_is_logged_with_both_lengths(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Prod drops INFO, so the fidelity signal has to ride a WARNING.

        Without it, the only way to know the rewriter is padding again is to
        re-read the prompts by hand.
        """
        original = "a cat on fire"
        padded = (
            "a magnificent, highly detailed cat lounging in a sunlit house, "
            "cinematic lighting, 8k, trending on artstation"
        )
        with caplog.at_level("WARNING"):
            self._rewrite(padded, original=original)

        assert "prompt_rewrite_fidelity" in caplog.text
        assert f"orig_chars={len(original)}" in caplog.text
        assert f"new_chars={len(padded)}" in caplog.text

    def test_a_reword_that_runs_slightly_long_is_not_padding(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Saying a thing another way is sometimes wordier, and that is the job.

        Both prod rewrites that prompted this tolerance grew by one character
        (267→268, 392→394). Warning on those trains the reader to ignore the
        warning.
        """
        original = "a cat on fire in a burning house"
        reworded = "a cat beside a bonfire in a house with flames"
        assert len(reworded) > len(original)

        with caplog.at_level("WARNING"):
            self._rewrite(reworded, original=original)

        assert "prompt_rewrite_fidelity" not in caplog.text

    def test_a_shorter_rewrite_is_not_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        """The common case: the offending clause comes out and nothing replaces it."""
        with caplog.at_level("WARNING"):
            self._rewrite("a cat in a house")

        assert "prompt_rewrite_fidelity" not in caplog.text


class TestRefusalMessagesSayWhatHappened:
    """ "Sorry, something went wrong." is what the user got for a REFUSAL.

    Observed in prod 2026-08-19 21:32:39Z on request `351cc180`: xAI turned down
    the chat completion itself with

        {"code":"permission-denied",
         "error":"Content violates usage guidelines. Failed check: ..."}

    and every failure class in `assistant_completion` — malformed request,
    unknown model, refusal — lands on the same flat string. 113 of them in the
    log. The user cannot tell "reword this" from "the bot is broken", and the
    two need opposite responses.

    Deliberately NOT fixed by teaching `_is_content_safety_error` this shape:
    that switch arms the rewrite loop, and a prompt that tripped a hard safety
    check is the one thing the loop must not go back and reword. Classify for
    the message, not for the retry.
    """

    XAI_HARD_REFUSAL = (
        'litellm.APIError: APIError: XaiException - {"code":"permission-denied",'
        '"error":"Content violates usage guidelines. Failed check: '
        'SAFETY_CHECK_TYPE_EXAMPLE"}'
    )

    def test_a_refusal_is_named_as_one(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """The one thing the user needs: it was the content, not the bot."""
        service, _plugin = make_service()

        message = service._handle_llm_error(
            openai.APIError(self.XAI_HARD_REFUSAL, None, body=None), "chat"
        )

        assert message in service_module._CHAT_REFUSED_LINES
        assert "check logs" not in message.lower()

    def test_a_refusal_does_not_repeat_the_providers_category(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """The category is an accusation, and the filter has false positives.

        It goes to the operator in the log, not to the channel, where naming it
        lands on whoever happened to be talking.
        """
        service, _plugin = make_service()

        message = service._handle_llm_error(
            openai.APIError(self.XAI_HARD_REFUSAL, None, body=None), "chat"
        )

        assert "SAFETY_CHECK" not in message
        assert "permission-denied" not in message

    def test_a_permission_error_that_is_not_about_content_is_not_a_refusal(
        self,
        make_service,  # type: ignore[no-untyped-def]
    ) -> None:
        """A key without access to a model is a config bug, not the user's fault.

        Keying on the bare "permission-denied" code would tell the channel to
        reword a perfectly good prompt forever.
        """
        service, _plugin = make_service()
        not_entitled = (
            'litellm.APIError: APIError: XaiException - {"code":"permission-denied",'
            '"error":"The API key does not have access to model grok-9"}'
        )

        message = service._handle_llm_error(openai.APIError(not_entitled, None, body=None), "chat")

        assert message not in service_module._CHAT_REFUSED_LINES

    def test_assistant_completion_stops_saying_something_went_wrong(
        self,
        make_service,  # type: ignore[no-untyped-def]
        mocker: MockerFixture,
    ) -> None:
        """End to end on the path that produced the line in the channel."""
        service, _plugin = make_service(assistantModel="xai/grok-4-1-fast-non-reasoning")
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=openai.APIError(self.XAI_HARD_REFUSAL, None, body=None),
        )

        result = service.assistant_completion(
            prompt="draw something",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content != "Sorry, something went wrong."
        assert result.content in service_module._CHAT_REFUSED_LINES

    def test_an_unrecognised_failure_still_falls_back(
        self,
        make_service,  # type: ignore[no-untyped-def]
        mocker: MockerFixture,
    ) -> None:
        """Classifying the known shapes must not invent a diagnosis for the rest."""
        service, _plugin = make_service()
        mocker.patch("llm.service.litellm.completion", side_effect=RuntimeError("disk on fire"))

        result = service.assistant_completion(
            prompt="hello",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content not in service_module._CHAT_REFUSED_LINES
        assert result.error is not None


class TestExhaustedRewriteMessage:
    """What the user asked for: the line after the loop gives up.

    The old one read "Error: No image generated. The prompt was blocked by
    content safety filters even after 1 rewrite attempt(s)." — machine voice,
    "attempt(s)", and it buries the only useful part, which is that the bot
    already tried rewording it once so rewording it again is not the move.
    """

    def test_it_says_the_reword_was_tried_too(self, make_service, mocker: MockerFixture) -> None:
        import litellm as litellm_module

        service, _plugin = make_service(
            imageModel="xai/grok-imagine-image",
            assistantModel="gemini/gemini-flash-latest",
            drawAutoRewriteMax=1,
            httpUrlBase="https://example.com/llm",
        )
        refusal = litellm_module.BadRequestError(
            message=(
                "litellm.BadRequestError: XaiException - {'code': 'imagine:content-moderated', "
                "'error': 'Generated image rejected by content moderation.'}"
            ),
            model="xai/grok-imagine-image",
            llm_provider="xai",
        )
        mocker.patch("llm.service.litellm.image_generation", side_effect=[refusal, refusal])
        mocker.patch(
            "llm.service.litellm.completion",
            return_value=make_completion_response("something milder"),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.image_generation("bunga bunga party")

        assert "attempt(s)" not in result.content
        assert result.content in service_module._DRAW_REWORDED_LINES

    def test_more_than_one_reword_is_counted(self, make_service, mocker: MockerFixture) -> None:
        import litellm as litellm_module

        service, _plugin = make_service(
            imageModel="xai/grok-imagine-image",
            assistantModel="gemini/gemini-flash-latest",
            drawAutoRewriteMax=2,
            httpUrlBase="https://example.com/llm",
        )
        refusal = litellm_module.BadRequestError(
            message=(
                "litellm.BadRequestError: XaiException - {'code': 'imagine:content-moderated', "
                "'error': 'Generated image rejected by content moderation.'}"
            ),
            model="xai/grok-imagine-image",
            llm_provider="xai",
        )
        mocker.patch(
            "llm.service.litellm.image_generation", side_effect=[refusal, refusal, refusal]
        )
        mocker.patch(
            "llm.service.litellm.completion",
            return_value=make_completion_response("something milder"),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.image_generation("bunga bunga party")

        assert "2" in result.content
        assert "attempt(s)" not in result.content


class TestRefusalCopyHasSomeLifeInIt:
    """A refusal is the bot's most-read line on a bad day. It can have a pulse.

    The constraint is that every variant keeps the actionable half. "Try a
    different subject" is the only part that changes what the user does next,
    so a joke that replaces it is a worse message than the flat one it replaced.
    """

    def test_every_blocked_line_still_says_what_to_do(self) -> None:
        for line in service_module._DRAW_BLOCKED_LINES:
            assert "different subject" in line.lower()

    def test_every_reworded_line_says_the_reword_was_tried(self) -> None:
        for line in service_module._DRAW_REWORDED_LINES:
            assert "reword" in line.lower() or "rewrite" in line.lower()

    def test_every_refusal_line_tells_the_user_to_reword(self) -> None:
        for line in service_module._CHAT_REFUSED_LINES:
            assert "reword" in line.lower() or "rephras" in line.lower()

    def test_no_variant_repeats_the_providers_category(self) -> None:
        """Same rule as the flat version: the category stays in the log."""
        every = (
            service_module._DRAW_BLOCKED_LINES
            + service_module._DRAW_REWORDED_LINES
            + service_module._CHAT_REFUSED_LINES
        )
        for line in every:
            assert "safety_check" not in line.lower()
            assert "csam" not in line.lower()

    def test_the_line_varies(self, make_service, mocker: MockerFixture) -> None:
        """One line every time is the thing being fixed; pin that it rotates."""
        service, _plugin = make_service()
        chosen = mocker.patch("llm.service.random.choice", side_effect=lambda seq: seq[-1])

        message = service._handle_llm_error(
            openai.APIError(
                'XaiException - {"error":"Content violates usage guidelines."}', None, body=None
            ),
            "chat",
        )

        assert chosen.called
        assert message == service_module._CHAT_REFUSED_LINES[-1]
