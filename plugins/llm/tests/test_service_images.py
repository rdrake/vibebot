"""Service images: saving, download, generation, draw context/rewrite, SSRF protection."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from llm import provider
from llm import service as service_module

from .conftest import make_completion_response

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


PNG = b"\x89PNG\r\n\x1a\nfake image"


def _picture(cost: float = 0.0) -> provider.Picture:
    """A delivered picture, as provider.draw returns it."""
    return provider.Picture(data=PNG, cost=cost)


def _blocked() -> provider.Picture:
    """No picture back: xAI's filter blocked it."""
    return provider.Picture(data=None)


class TestImageSaving:
    """Tests for _save_image_bytes functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            httpRoot="/tmp/test_llm_images",
            httpUrlBase="https://example.com/llm",
            fileCleanupAge=24,
        )

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

    def test_detect_image_format_jpeg(self) -> None:
        """GIVEN JPEG magic bytes THEN returns 'jpg'."""
        assert self.service._detect_image_format(b"\xff\xd8\xff\xe0data") == "jpg"

    def test_detect_image_format_webp(self) -> None:
        """GIVEN WebP magic bytes THEN returns 'webp'."""
        assert self.service._detect_image_format(b"RIFF\x00\x00\x00\x00WEBPdata") == "webp"

    def test_detect_image_format_unknown(self) -> None:
        """GIVEN unknown bytes THEN returns None."""
        assert self.service._detect_image_format(b"unknown data") is None

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


class TestImageGenerationWithBase64:
    """Tests for image_generation with base64 handling and typing indicators."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            timeout=30,
            httpRoot="/tmp/test",
            httpUrlBase="https://example.com/llm",
            fileCleanupAge=24,
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
        msg.prefix = "testnick!user@host"
        return msg

    def test_image_generation_saves_the_picture(self, tmp_path: object) -> None:
        """GIVEN provider returns picture bytes WHEN generating THEN saves and returns URL."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: {
                "timeout": 30,
                "httpRoot": str(tmp_path),
                "httpUrlBase": "https://example.com/llm",
                "fileCleanupAge": 24,
            }.get(key)
        )

        self.mocker.patch("llm.service.provider.draw", return_value=_picture())
        result = self.service.image_generation("a cat")

        assert result.content.startswith("https://example.com/llm/img_")
        assert result.content.endswith(".png")

    def test_image_generation_sends_typing_indicator(self) -> None:
        """GIVEN irc context WHEN generating THEN sends typing indicators."""
        irc = self._make_mock_irc()
        msg = self._make_mock_msg()

        self.mocker.patch("llm.service.provider.draw", return_value=_picture())
        self.mocker.patch.object(self.service, "_save_image_bytes", return_value=None)
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

        self.mocker.patch("llm.service.provider.draw", side_effect=Exception("API error"))
        result = self.service.image_generation("a cat", irc=irc, msg=msg)

        assert "Error" in result.content

        # Should still send typing=done in finally block
        assert irc.queueMsg.call_count == 2
        second_msg = irc.queueMsg.call_args_list[1][0][0]
        assert second_msg.server_tags == {"+typing": "done"}

    def test_image_generation_no_data_in_response(self) -> None:
        """GIVEN empty response WHEN generating THEN returns content filter error."""
        self.mocker.patch("llm.service.provider.draw", return_value=_blocked())
        result = self.service.image_generation("a cat")

        assert "No image generated" in result.content
        assert "content safety filters" in result.content


class TestCleanupWithImages:
    """Tests for _cleanup_old_files with image extensions."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service) -> None:
        """Set up test fixtures."""
        self.service, self.mock_plugin = make_service(
            fileCleanupAge=24,
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
        )

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
        self.make_service = make_service
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

    def test_get_http_paths_uses_limnoria_public_url(self) -> None:
        """Empty httpUrlBase falls back to Limnoria's publicUrl plus /llm, so
        posted links are ones people can open (configuration.md)."""
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, *a, **kw: {"httpRoot": "/srv/web", "httpUrlBase": ""}.get(
                key, ""
            )
        )
        mock_conf = self.mocker.patch("llm.service.conf")
        mock_conf.supybot.servers.http.publicUrl.return_value = "https://bot.example/"
        mock_conf.supybot.servers.http.port.return_value = 8080

        assert self.service.get_http_paths() == ("/srv/web", "https://bot.example/llm")

    def test_scheduled_cleanup_uses_the_configured_age(self, tmp_path: object) -> None:
        """The scheduler passes no arguments: fileCleanupAge decides what goes."""
        import os
        import time
        from pathlib import Path

        dir_path = Path(str(tmp_path))
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, *a, **kw: {
                "httpRoot": str(dir_path),
                "httpUrlBase": "https://example.com/llm",
                "fileCleanupAge": 24,
            }.get(key, "")
        )
        old_file = dir_path / "old.html"
        new_file = dir_path / "new.html"
        old_file.write_text("old")
        new_file.write_text("new")
        old_mtime = time.time() - 25 * 3600
        os.utime(str(old_file), (old_mtime, old_mtime))

        self.service.run_scheduled_cleanup()

        assert not old_file.exists()
        assert new_file.exists()

    def test_own_image_hosts_include_the_upload_host(self) -> None:
        """An image URL on the upload host that the turn did not mint is
        rejected, so that host must count as ours (tuning-monitoring.md)."""
        service, _plugin = self.make_service(
            imageUploadUrl="https://paste.example/img/", httpUrlBase="https://bot.example/"
        )
        assert service.own_image_hosts() == {"paste.example", "bot.example"}

    def test_save_markdown_to_http_oserror_returns_none(self) -> None:
        """GIVEN mkdir raises OSError WHEN save_markdown_to_http called THEN returns None."""
        self.mocker.patch.object(
            self.service,
            "get_http_paths",
            return_value=("/nonexistent/path", "http://x"),
        )
        self.mocker.patch("llm.service.Path.mkdir", side_effect=OSError("disk full"))

        result = self.service.save_markdown_to_http("# hello world")

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


class TestDownloadImageBytes:
    """@meme fetches a custom template from a user's image URL."""

    URL = "https://images.example.com/cat"
    DATA = PNG + b"x" * 1024

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def _serve(self, content_type: str) -> Mock:
        """An opener whose response honours read's size, as a socket does."""
        data = self.DATA
        resp = self.mocker.MagicMock()
        resp.headers = {"Content-Type": content_type}
        resp.read.side_effect = lambda n=-1: data[:n] if n >= 0 else data
        resp.__enter__.return_value = resp
        opener = self.mocker.Mock()
        opener.open.return_value = resp
        self.mocker.patch("llm.service._no_redirect_opener", return_value=opener)
        return opener

    def test_a_public_image_downloads(self) -> None:
        """A bare Content-Type with no parameters names the extension."""
        self.mocker.patch.object(self.service, "_resolves_to_public", return_value=True)
        self._serve("image/png")

        assert self.service._download_image_bytes(self.URL) == (self.DATA, "png")

    def test_an_unsafe_url_is_never_fetched(self) -> None:
        self.mocker.patch.object(self.service, "_resolves_to_public", return_value=True)
        opener = self._serve("image/png")

        assert self.service._download_image_bytes("http://127.0.0.1/cat.png") is None
        opener.open.assert_not_called()

    def test_a_host_resolving_to_a_private_address_is_never_fetched(self) -> None:
        """DNS-rebinding guard (e09a4dd): the hostname passes the static check
        but resolves privately, so the fetch must not happen."""
        self.mocker.patch.object(self.service, "_resolves_to_public", return_value=False)
        opener = self._serve("image/png")

        assert self.service._download_image_bytes(self.URL) is None
        opener.open.assert_not_called()


class TestFilterImages:
    """Image URLs pasted into @ask reach the vision model once validated."""

    def test_valid_urls_pass_and_invalid_ones_drop(self, make_service, mocker) -> None:  # type: ignore[no-untyped-def]
        service, _plugin = make_service()
        mocker.patch.object(service, "_resolves_to_public", return_value=True)

        kept = service._filter_images(
            ["https://x.example/a.png", "javascript:alert(1).png", "https://x.example/b.jpg"]
        )

        assert kept == ["https://x.example/a.png", "https://x.example/b.jpg"]

    def test_nothing_valid_is_none(self, make_service) -> None:  # type: ignore[no-untyped-def]
        service, _plugin = make_service()
        assert service._filter_images(["javascript:alert(1).png"]) is None
        assert service._filter_images([]) is None


class TestDrawAutoRewrite:
    """Tests for automatic prompt rewriting on content safety failures."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            assistantModel="anthropic/claude-haiku-5-5",
            timeout=30,
            httpRoot="/tmp/test",
            httpUrlBase="https://example.com/llm",
            drawAutoRewriteMax=3,
        )
        self.config_values = {
            "assistantModel": "anthropic/claude-haiku-5-5",
            "timeout": 30,
            "httpRoot": "/tmp/test",
            "httpUrlBase": "https://example.com/llm",
            "drawAutoRewriteMax": 3,
        }

    def _make_success_response(self) -> provider.Picture:
        """A delivered picture."""
        return _picture(cost=0.02)

    def _make_empty_response(self) -> provider.Picture:
        """No picture back (content-blocked)."""
        return _blocked()

    def _make_rewrite_response(self, rewritten: str = "a safe cat") -> provider.Reply:
        """A completion response for prompt rewriting."""
        return make_completion_response(
            rewritten, prompt_tokens=20, completion_tokens=10, cost=0.001
        )

    def test_auto_rewrite_on_empty_data_succeeds(self) -> None:
        """GIVEN empty response data WHEN auto-rewrite enabled THEN retries with rewritten prompt."""
        empty_resp = self._make_empty_response()
        success_resp = self._make_success_response()
        rewrite_resp = self._make_rewrite_response("a friendly cat")

        self.mocker.patch("llm.service.provider.draw", side_effect=[empty_resp, success_resp])
        self.mocker.patch("llm.service.provider.chat", return_value=rewrite_resp)
        self.mocker.patch.object(
            self.service,
            "_save_image_bytes",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("a dangerous cat")

        assert result.content == "https://example.com/llm/img_local.png"
        assert result.rewritten_prompt == "a friendly cat"

    def test_auto_rewrite_on_content_policy_error_succeeds(self) -> None:
        """GIVEN ContentPolicyViolationError WHEN auto-rewrite enabled THEN retries."""
        rewrite_resp = self._make_rewrite_response("a safe prompt")
        success_resp = self._make_success_response()

        self.mocker.patch(
            "llm.service.provider.draw",
            side_effect=[
                provider.ContentPolicyViolationError("blocked"),
                success_resp,
            ],
        )
        self.mocker.patch("llm.service.provider.chat", return_value=rewrite_resp)
        self.mocker.patch.object(
            self.service,
            "_save_image_bytes",
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
            "llm.service.provider.draw",
            side_effect=[empty_resp, empty_resp, success_resp],
        )
        self.mocker.patch(
            "llm.service.provider.chat",
            side_effect=[rewrite1, rewrite2],
        )
        self.mocker.patch.object(
            self.service,
            "_save_image_bytes",
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
            "llm.service.provider.draw",
            side_effect=[empty_resp, empty_resp, empty_resp, empty_resp],
        )
        self.mocker.patch(
            "llm.service.provider.chat",
            side_effect=[rewrite1, rewrite2, rewrite3],
        )
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

        self.mocker.patch("llm.service.provider.draw", return_value=empty_resp)
        mock_completion = self.mocker.patch("llm.service.provider.chat")
        result = self.service.image_generation("test prompt")

        assert "content safety filters" in result.content
        mock_completion.assert_not_called()

    def test_auto_rewrite_llm_failure_falls_back(self) -> None:
        """GIVEN rewrite LLM fails WHEN retrying THEN reports the block, not a count.

        No rewrite got made, so mentioning one -- or "0 rewordings" -- describes
        an internal counter rather than anything the user can act on.
        """
        empty_resp = self._make_empty_response()

        self.mocker.patch("llm.service.provider.draw", return_value=empty_resp)
        self.mocker.patch(
            "llm.service.provider.chat",
            side_effect=Exception("LLM unavailable"),
        )
        result = self.service.image_generation("test prompt")

        assert result.error is not None
        assert result.content in service_module._DRAW_BLOCKED_LINES

    def test_auto_rewrite_skipped_when_ask_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN no key for the rewrite model's provider WHEN blocked THEN skips rewrite.

        The rewriter runs on ``assistantModel`` (anthropic/claude-haiku-5-5), so
        ANTHROPIC_API_KEY is the variable that gates it. ``IMAGE_MODEL`` is an
        xai/ model, whose key stays set.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        empty_resp = self._make_empty_response()

        self.mocker.patch("llm.service.provider.draw", return_value=empty_resp)
        mock_completion = self.mocker.patch("llm.service.provider.chat")
        result = self.service.image_generation("test prompt")

        assert result.error is not None
        assert result.content in service_module._DRAW_BLOCKED_LINES
        mock_completion.assert_not_called()

    def test_auto_rewrite_aggregates_costs(self) -> None:
        """GIVEN successful rewrite WHEN costs tracked THEN aggregated in result."""
        empty_resp = self._make_empty_response()
        success_resp = self._make_success_response()
        rewrite_resp = self._make_rewrite_response("safe prompt")

        self.mocker.patch("llm.service.provider.draw", side_effect=[empty_resp, success_resp])
        self.mocker.patch("llm.service.provider.chat", return_value=rewrite_resp)
        self.mocker.patch.object(
            self.service, "_save_image_bytes", return_value="https://example.com/llm/img_a.png"
        )
        result = self.service.image_generation("test prompt")

        # Should include both rewrite and generation costs
        assert result.prompt_tokens > 0
        assert result.cost > 0

    def test_non_content_error_does_not_trigger_rewrite(self) -> None:
        """GIVEN timeout error WHEN generating THEN no rewrite attempted."""
        self.mocker.patch(
            "llm.service.provider.draw",
            side_effect=provider.RequestTimeoutError("timed out"),
        )
        mock_completion = self.mocker.patch("llm.service.provider.chat")
        result = self.service.image_generation("test prompt")

        assert "timed out" in result.content.lower()
        mock_completion.assert_not_called()

    def test_auth_error_does_not_trigger_rewrite(self) -> None:
        """GIVEN authentication error WHEN generating THEN no rewrite attempted."""
        self.mocker.patch(
            "llm.service.provider.draw",
            side_effect=provider.AuthenticationError("invalid key"),
        )
        mock_completion = self.mocker.patch("llm.service.provider.chat")
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
            "llm.service.provider.draw",
            side_effect=[empty_resp, empty_resp, empty_resp],
        )
        mock_completion = self.mocker.patch(
            "llm.service.provider.chat",
            side_effect=[rewrite1, rewrite2],
        )
        self.service.image_generation("test prompt")

        # Second rewrite call should include prior_rewrites in the user message
        assert mock_completion.call_count == 2
        second_call_messages = mock_completion.call_args_list[1][1]["messages"]
        user_msg = second_call_messages[1]["content"]
        assert "rewrite v1" in user_msg
        assert "Previous rewrite attempts" in user_msg

    def test_non_moderation_bad_request_does_not_trigger_rewrite(self) -> None:
        """GIVEN BadRequestError without moderation keywords WHEN generating THEN no rewrite."""
        self.mocker.patch(
            "llm.service.provider.draw",
            side_effect=provider.BadRequestError(
                "Invalid image size parameter",
            ),
        )
        mock_completion = self.mocker.patch("llm.service.provider.chat")
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

    def test_an_invalid_prompt_never_reaches_the_billed_call(self) -> None:
        """The user sees the validator's own message, and nothing is drawn."""
        from llm.service import MAX_PROMPT_LENGTH

        draw = self.mocker.patch("llm.service.provider.draw")

        empty = self.service.image_generation("")
        too_long = self.service.image_generation("x" * (MAX_PROMPT_LENGTH + 1))

        assert empty.content == "Error: Prompt cannot be empty"
        assert too_long.content == f"Error: Prompt too long (max {MAX_PROMPT_LENGTH} characters)"
        draw.assert_not_called()

    def test_image_generation_missing_draw_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN no key for the image model's provider THEN returns an error.

        An xai/ ``IMAGE_MODEL`` makes XAI_API_KEY the variable this guard reads.
        """
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = self.make_service()

        result = service.image_generation("A beautiful sunset")

        assert result.error is not None
        assert "XAI_API_KEY" in result.content


class TestImageGenerationPaths:
    """Tests for image generation rewrite loop and error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            assistantModel="xai/grok-4.3",
            timeout=30,
            httpRoot="/tmp/test",
            httpUrlBase="https://example.com/llm",
            drawAutoRewriteMax=3,
        )

    def test_rewrite_empty_response(self) -> None:
        """GIVEN the rewriter answers with nothing THEN there is no rewrite, but
        the call was made and billed, so its usage still comes back."""
        self.mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response(
                "  ", prompt_tokens=10, completion_tokens=5, cost=0.002
            ),
        )

        result = self.service._rewrite_prompt_for_safety("bad prompt", "blocked", [], "#chan")

        assert result == (None, 10, 5, 0.002)

    def test_an_empty_rewrite_is_booked_on_the_blocked_draw(self) -> None:
        """The draw that gave up still books the rewriter spend it paid for."""
        self.mocker.patch("llm.service.provider.draw", return_value=_blocked())
        self.mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response(
                "", prompt_tokens=10, completion_tokens=5, cost=0.002
            ),
        )

        result = self.service.image_generation("a cat")

        assert result.error is not None
        assert result.cost == pytest.approx(0.002)
        assert (result.prompt_tokens, result.completion_tokens) == (10, 5)

    def test_xai_model_kwargs(self) -> None:
        """GIVEN xai model WHEN _attempt_image_generation called THEN asks for high quality."""
        mock_img_gen = self.mocker.patch("llm.service.provider.draw", return_value=_picture())
        self.mocker.patch.object(self.service, "_save_image_bytes", return_value=None)

        self.service._attempt_image_generation("cat", "xai/grok-imagine-image", 30)

        call_kwargs = mock_img_gen.call_args.kwargs
        assert call_kwargs["hires"] is True
        # No aspect_ratio: xAI's default is `auto`, the best ratio for the
        # prompt. A pinned 9:16 made every draw portrait.
        assert call_kwargs["aspect_ratio"] is None

    @pytest.mark.parametrize(
        ("aspect", "ratio"),
        [("portrait", "2:3"), ("landscape", "3:2"), ("square", "1:1")],
    )
    def test_xai_aspect_maps_to_a_ratio(self, aspect: str, ratio: str) -> None:
        """GIVEN the planner chose a shape WHEN drawing on xAI THEN it rides as aspect_ratio.

        `auto` reads the prompt as a scene: a 1950s cinema poster came back
        16:9 (2026-09-19). The planner knows it is a poster; it says so.
        """
        mock_img_gen = self.mocker.patch("llm.service.provider.draw", return_value=_picture())
        self.mocker.patch.object(self.service, "_save_image_bytes", return_value=None)

        self.service._attempt_image_generation("cat", "xai/grok-imagine-image", 30, aspect=aspect)

        assert mock_img_gen.call_args.kwargs["aspect_ratio"] == ratio

    def test_unknown_aspect_is_ignored(self) -> None:
        """GIVEN a shape the map does not know WHEN drawing THEN xAI keeps `auto`."""
        mock_img_gen = self.mocker.patch("llm.service.provider.draw", return_value=_picture())
        self.mocker.patch.object(self.service, "_save_image_bytes", return_value=None)

        self.service._attempt_image_generation("cat", "xai/grok-imagine-image", 30, aspect="wide")

        assert mock_img_gen.call_args.kwargs["aspect_ratio"] is None

    def test_save_failure(self) -> None:
        """GIVEN a picture but save fails WHEN _attempt_image_generation called
        THEN returns error, still carrying the generation the provider billed."""
        self.mocker.patch("llm.service.provider.draw", return_value=_picture(cost=0.07))
        self.mocker.patch.object(self.service, "_save_image_bytes", return_value=None)

        result = self.service._attempt_image_generation("cat", "xai/grok-imagine-image", 30)

        assert result is not None
        assert result.error is not None
        assert result.cost == pytest.approx(0.07)
        assert result.model == "xai/grok-imagine-image"

    def test_timeout_not_stashed(self) -> None:
        """GIVEN image_generation times out and stashing fails WHEN called THEN returns error."""
        self.mocker.patch(
            "llm.service.provider.draw",
            side_effect=provider.RequestTimeoutError("Request timed out"),
        )
        self.mocker.patch.object(self.service, "_stash_timeout", return_value=False)

        result = self.service.image_generation("a cat")

        assert result.error is not None
        # Nothing will arrive later, so nothing is promised.
        assert "keep trying" not in result.content
        assert result.content == (
            "Error: Image generation timed out. Try again or simplify your request."
        )

    def test_a_stashed_timeout_promises_the_image(self) -> None:
        """The background retry will deliver it, so the user is told to expect it."""
        self.mocker.patch(
            "llm.service.provider.draw",
            side_effect=provider.RequestTimeoutError("Request timed out"),
        )
        self.mocker.patch.object(self.service, "_stash_timeout", return_value=True)

        result = self.service.image_generation("a cat")

        assert result.content == (
            "Timed out, but I'll keep trying and deliver the image when ready."
        )

    def test_the_channel_rewrite_cap_applies(self) -> None:
        """drawAutoRewriteMax is per channel, and each rewrite buys a billed draw.
        The channel is the PRIVMSG target, msg.args[0]."""
        values = {"drawAutoRewriteMax": 1, "assistantModel": "xai/grok-4.3", "timeout": 30}
        self.mock_plugin.registryValue = self.mocker.Mock(
            side_effect=lambda key, channel=None: (
                0 if (key, channel) == ("drawAutoRewriteMax", "#c") else values.get(key, "")
            )
        )
        self.mocker.patch("llm.service.provider.draw", return_value=_blocked())
        rewrite = self.mocker.patch.object(self.service, "_rewrite_prompt_for_safety")
        msg = self.mocker.Mock(args=("#c", "a cat"))

        result = self.service.image_generation("a cat", None, msg)

        assert result.error is not None
        rewrite.assert_not_called()

    def test_a_billed_failure_mid_loop_is_booked(self) -> None:
        """Earlier refusal + rewrite + the billed failure that ended the loop."""
        price = provider.image_price(service_module.IMAGE_MODEL)
        self.mocker.patch(
            "llm.service.provider.draw",
            side_effect=[
                provider.ContentPolicyViolationError("moderated", billed=True),
                provider.ProviderError("upstream 500", billed=True),
            ],
        )
        self.mocker.patch.object(
            self.service, "_rewrite_prompt_for_safety", return_value=("a calm cat", 10, 5, 0.001)
        )

        result = self.service.image_generation("a cat")

        assert result.error is not None
        assert result.cost == pytest.approx(2 * price + 0.001)

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
            "model": "xai/grok-imagine-image",
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

        The retry resolves from ``task.model`` ("xai/..." -> xai), not from
        a per-command registry key that could name a different provider.
        """
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        task = self._make_task()

        result = self.service._retry_image(task, {"prompt": "cat"})

        assert result.status == "failed_terminal"
        assert "XAI_API_KEY" in result.reason

    def test_retry_image_carries_the_aspect(self) -> None:
        """GIVEN a stashed draw with an aspect WHEN retried THEN the shape survives the stash."""
        from llm.service import ImageResult

        task = self._make_task(model="xai/grok-imagine-image")
        attempt = self.mocker.patch.object(
            self.service,
            "_attempt_image_generation",
            return_value=ImageResult(content="http://img.png"),
        )

        self.service._retry_image(task, {"prompt": "cat", "aspect": "portrait"})

        assert attempt.call_args.kwargs["aspect"] == "portrait"

    def test_a_recovered_draw_is_delivered(self) -> None:
        """A paid, recovered draw completes and carries the image for delivery."""
        from llm.service import ImageResult

        task = self._make_task()
        url = "https://example.com/llm/img_recovered.jpg"
        self.mocker.patch.object(
            self.service,
            "_attempt_image_generation",
            return_value=ImageResult(content=url, cost=0.07, model=task.model, url=url),
        )

        result = self.service._retry_image(task, {"prompt": "cat"})

        assert result.status == "completed"
        assert result.content == url
        assert result.reply_target == "#chan"
        assert result.cost == pytest.approx(0.07)

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
            timeout=30,
        )

    def _mock_upload_reply(self, payload: str) -> Mock:
        """Patch the opener so an upload POST returns ``payload``."""
        data = payload.encode()
        mock_resp = self.mocker.Mock()
        # Honour read's size limit, as a real response does.
        mock_resp.read.side_effect = lambda n=-1: data[:n] if n >= 0 else data
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

    @pytest.mark.parametrize(
        "reply",
        [
            # success:false wins even when a filePath rides along.
            '{"results": [{"success": false, "error": "quota", "filePath": "/img/x.png"}]}',
            '{"results": ["x"]}',
            '{"results": []}',
            '{"results": {"success": true, "filePath": "/img/x.png"}}',
            '{"ok": true}',
        ],
    )
    def test_upload_reply_without_a_successful_result_falls_back_to_local(self, reply: str) -> None:
        """The reply is untrusted: anything but a successful first result is
        refused, and the image is stored locally instead of raising."""
        self._mock_upload_reply(reply)

        result = self.service._save_image_bytes(b"\x89PNG\r\n\x1a\nfake image data")

        assert result is not None
        assert result.startswith("https://example.com/llm/img_")

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


class TestXaiModerationArmsTheRewrite:
    """The auto-rewrite loop must fire on xAI's moderation refusal.

    It did not, for as long as prod has been drawing on grok-imagine:
    ``imagine:content-moderated`` was classified as an ordinary bad request and
    ``image_generation`` returned without retrying. Measured over six hours on
    2026-08-15: 18 draws, 10 refused, zero rewrites attempted, with
    ``drawAutoRewriteMax`` set to 3 the whole time. The classification now
    lives in provider.py (test_provider.py); this pins what the service does
    with the result.
    """

    XAI_MODERATION_ERROR = (
        "{'code': 'imagine:content-moderated', 'error': 'Generated image "
        "rejected by content moderation.', 'usage': {'cost_in_usd_ticks': 200000000}}"
    )

    def _xai_error(self) -> Exception:
        return provider.ContentPolicyViolationError(
            self.XAI_MODERATION_ERROR,
            billed=True,
        )

    def test_refusal_now_gets_one_rewrite_and_recovers(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """End to end: refused draw -> rewritten prompt -> image delivered."""
        service, _ = make_service(
            assistantModel="xai/grok-4.3",
            drawAutoRewriteMax=1,
            httpUrlBase="https://example.com/llm",
        )
        image_generation = mocker.patch(
            "llm.service.provider.draw",
            side_effect=[self._xai_error(), _picture()],
        )
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("a tasteful party scene"),
        )
        mocker.patch.object(
            service, "_save_image_bytes", return_value="https://example.com/llm/img_a.png"
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
            assistantModel="xai/grok-4.3",
            drawAutoRewriteMax=1,
            httpUrlBase="https://example.com/llm",
        )
        image_generation = mocker.patch(
            "llm.service.provider.draw",
            side_effect=[self._xai_error(), self._xai_error()],
        )
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("something milder"),
        )

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
            assistantModel="xai/grok-4.3",
            httpUrlBase="https://example.com/llm",
            drawAutoRewriteMax=1,
        )

    def _rewrite(self, returned: str, original: str = "a cat on fire in a burning house"):
        """Run one rewrite, return the (messages, result) the call produced."""
        completion = self.mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response(returned, prompt_tokens=10, completion_tokens=5),
        )
        result = self.service._rewrite_prompt_for_safety(original, "content moderation", [], None)
        return completion.call_args[1]["messages"], result

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


class TestHandleLlmErrorMessages:
    """Each failure class tells the user what to do next."""

    def test_timeout(self, make_service) -> None:  # type: ignore[no-untyped-def]
        service, _plugin = make_service()
        message = service._handle_llm_error(provider.RequestTimeoutError("t"), "image generation")
        assert message == "Error: Image generation timed out. Try again or simplify your request."

    def test_rate_limit_says_wait(self, make_service) -> None:  # type: ignore[no-untyped-def]
        service, _plugin = make_service()
        message = service._handle_llm_error(provider.RateLimitError("x"), "chat")
        assert message == (
            "Error: API rate limit reached. Please wait a few minutes and try again."
        )

    def test_moderation_block_says_rephrase(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """A reworded prompt often gets through (ContentPolicyViolationError)."""
        service, _plugin = make_service()
        message = service._handle_llm_error(
            provider.ContentPolicyViolationError("x"), "image generation"
        )
        assert message == (
            "Error: Content violates AI safety policies. Please rephrase your request."
        )

    def test_unknown_exception(self, make_service) -> None:  # type: ignore[no-untyped-def]
        service, _plugin = make_service()
        message = service._handle_llm_error(ValueError("boom"), "chat")
        assert message == (
            "Error: Unable to complete chat. Check your configuration or try again later."
        )


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

    Deliberately a RefusalError, not a ContentPolicyViolationError: the latter
    arms the rewrite loop, and a prompt that tripped a hard safety check is the
    one thing the loop must not go back and reword. Classify for the message,
    not for the retry. (Telling the two apart from the body is provider.py's
    job; see test_provider.py.)
    """

    XAI_HARD_REFUSAL = (
        '{"code":"permission-denied",'
        '"error":"Content violates usage guidelines. Failed check: '
        'SAFETY_CHECK_TYPE_EXAMPLE"}'
    )

    def _refusal(self) -> provider.RefusalError:
        return provider.RefusalError(self.XAI_HARD_REFUSAL)

    def test_a_refusal_is_named_as_one(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """The one thing the user needs: it was the content, not the bot."""
        service, _plugin = make_service()

        message = service._handle_llm_error(self._refusal(), "chat")

        assert message in service_module._CHAT_REFUSED_LINES
        assert "check logs" not in message.lower()

    def test_a_refusal_does_not_repeat_the_providers_category(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """The category is an accusation, and the filter has false positives.

        It goes to the operator in the log, not to the channel, where naming it
        lands on whoever happened to be talking.
        """
        service, _plugin = make_service()

        message = service._handle_llm_error(self._refusal(), "chat")

        assert "SAFETY_CHECK" not in message
        assert "permission-denied" not in message

    def test_assistant_completion_stops_saying_something_went_wrong(
        self,
        make_service,  # type: ignore[no-untyped-def]
        mocker: MockerFixture,
    ) -> None:
        """End to end on the path that produced the line in the channel."""
        service, _plugin = make_service(assistantModel="xai/grok-4-1-fast-non-reasoning")
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=self._refusal(),
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
        mocker.patch("llm.service.provider.chat", side_effect=RuntimeError("disk on fire"))

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
        service, _plugin = make_service(
            assistantModel="xai/grok-4.3",
            drawAutoRewriteMax=1,
            httpUrlBase="https://example.com/llm",
        )
        refusal = provider.ContentPolicyViolationError(
            "{'code': 'imagine:content-moderated', "
            "'error': 'Generated image rejected by content moderation.'}",
        )
        mocker.patch("llm.service.provider.draw", side_effect=[refusal, refusal])
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("something milder"),
        )

        result = service.image_generation("bunga bunga party")

        assert "attempt(s)" not in result.content
        assert result.content in service_module._DRAW_REWORDED_LINES

    def test_more_than_one_reword_is_counted(self, make_service, mocker: MockerFixture) -> None:
        service, _plugin = make_service(
            assistantModel="xai/grok-4.3",
            drawAutoRewriteMax=2,
            httpUrlBase="https://example.com/llm",
        )
        refusal = provider.ContentPolicyViolationError(
            "{'code': 'imagine:content-moderated', "
            "'error': 'Generated image rejected by content moderation.'}",
        )
        mocker.patch("llm.service.provider.draw", side_effect=[refusal, refusal, refusal])
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("something milder"),
        )

        result = service.image_generation("bunga bunga party")

        assert "2" in result.content
        assert "attempt(s)" not in result.content


class TestRefusalCopyHasSomeLifeInIt:
    """A refusal is the bot's most-read line on a bad day. It can have a pulse.

    The constraint is that every variant keeps the actionable half. "Try a
    different subject" is the only part that changes what the user does next,
    so a joke that replaces it is a worse message than the flat one it replaced.
    """

    def test_the_line_varies(self, make_service, mocker: MockerFixture) -> None:
        """One line every time is the thing being fixed; pin that it rotates."""
        service, _plugin = make_service()
        chosen = mocker.patch("llm.service.random.choice", side_effect=lambda seq: seq[-1])

        message = service._handle_llm_error(
            provider.RefusalError('{"error":"Content violates usage guidelines."}'),
            "chat",
        )

        assert chosen.called
        assert message == service_module._CHAT_REFUSED_LINES[-1]
