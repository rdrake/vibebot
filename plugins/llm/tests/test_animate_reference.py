"""Tests for @animate with a reference image — image-to-video.

The video box serves a MiniMax-H3 ``FL2VA`` checkpoint whose partition takes
exactly two tasks: ``t2va`` (text only) and ``fl2va`` (first/last frame). Measured
against the box on 2026-08-21, ``t2va`` with a file attached fails outright —
"t2va does not accept an image condition" — so the task has to switch with the
attachment, not with the audio setting.

The bytes are the other half. A reference image is a file a stranger in a
channel chose, fetched by the bot from a URL of their choosing, so this file
covers the fetch guards as closely as the plumbing: no private hosts, no
redirects, no oversize payloads, and nothing forwarded that Pillow could not
decode and re-encode first.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from llm.service import split_reference_url

from .conftest import huge_png

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm.service import LLMService

_URL = "http://video.example.com:14205"
_KEY = "not-a-real-token-for-tests"
_IMG = "https://pics.example.com/cat.png"


@pytest.fixture
def animate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMATE_API_KEY", _KEY)


def _service(make_service: Callable[..., tuple[LLMService, Mock]], **overrides: Any):
    overrides.setdefault("animateApiUrl", _URL)
    return make_service(**overrides)


def _png_jpeg_bytes(width: int = 64, height: int = 48):
    """A real JPEG, the shape fetch_reference_image returns."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes(width: int = 64, height: int = 48, colour: tuple[int, int, int] = (10, 120, 200)):
    """A real PNG, built by Pillow so the decode path under test is honest."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


class _Resp:
    """Minimal urlopen context manager."""

    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        self._data = data
        self.headers = headers or {"Content-Type": "image/png"}
        self.status = 200

    def read(self, amount: int | None = None) -> bytes:
        return self._data[:amount] if amount is not None else self._data

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


class TestReferenceUrlExtraction:
    """The URL rides inside the prompt; the user should not have to flag it."""

    def test_url_anywhere_is_lifted_out_of_the_prompt(self) -> None:
        """GIVEN a prompt with a URL WHEN split THEN the URL leaves the text."""
        url, prompt = split_reference_url(f"make {_IMG} this cat dance")

        assert url == _IMG
        assert prompt == "make this cat dance"

    def test_no_url_leaves_the_prompt_untouched(self) -> None:
        """GIVEN a plain prompt WHEN split THEN nothing is claimed or removed."""
        url, prompt = split_reference_url("a corgi riding a unicorn")

        assert url is None
        assert prompt == "a corgi riding a unicorn"

    def test_non_image_url_is_not_a_reference(self) -> None:
        """GIVEN a link that is not an image WHEN split THEN it is left alone.

        Stripping it would silently delete a subject the user typed, which is
        the failure this whole path exists to stop.
        """
        text = "a clip in the style of https://youtube.com/watch?v=abc"

        url, prompt = split_reference_url(text)

        assert url is None
        assert prompt == text

    def test_second_image_url_does_not_survive_into_the_prompt(self) -> None:
        """GIVEN two image URLs WHEN split THEN the first conditions and neither is text.

        A URL left in the prompt reaches a video model that renders words on
        screen, so the leftover would appear in the clip.
        """
        other = "https://pics.example.com/dog.png"

        url, prompt = split_reference_url(f"{_IMG} and {other} boxing")

        assert url == _IMG
        assert "http" not in prompt
        assert prompt == "and boxing"


class TestReferenceImageFetch:
    """Someone else's file, fetched on their say-so. Guard it like one."""

    def test_unsafe_url_is_never_fetched(self, make_service, animate_env, mocker) -> None:
        """GIVEN a URL that fails validation WHEN fetched THEN no request is made."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "validate_image_url", return_value=False)
        opener = mocker.patch("urllib.request.build_opener")

        assert service.fetch_reference_image("http://127.0.0.1/x.png") is None
        opener.assert_not_called()

    def test_valid_png_is_decoded_and_re_encoded(self, make_service, animate_env, mocker) -> None:
        """GIVEN a real PNG WHEN fetched THEN the forwarded bytes are Pillow's, not the wire's.

        Re-encoding is the guard: a polyglot file or an EXIF payload does not
        survive a decode to pixels and a fresh encode.
        """
        service, _ = _service(make_service)
        mocker.patch.object(service, "validate_image_url", return_value=True)
        original = _png_bytes()
        mocker.patch(
            "urllib.request.OpenerDirector.open",
            return_value=_Resp(original + b"<?php trailing junk ?>"),
        )

        ref = service.fetch_reference_image(_IMG)

        assert ref is not None
        assert ref.extension == "jpg"
        assert b"php" not in ref.data
        from PIL import Image

        with Image.open(io.BytesIO(ref.data)) as img:
            assert img.size == (64, 48)

    def test_html_pretending_to_be_a_png_is_rejected(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a .png URL serving HTML WHEN fetched THEN it is refused."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "validate_image_url", return_value=True)
        mocker.patch(
            "urllib.request.OpenerDirector.open",
            return_value=_Resp(b"<!doctype html><html>not a cat</html>"),
        )

        assert service.fetch_reference_image(_IMG) is None

    def test_oversize_payload_is_refused(self, make_service, animate_env, mocker) -> None:
        """GIVEN a body past the cap WHEN fetched THEN it is refused, not truncated.

        A truncated image would still decode for some formats; the point is to
        refuse the download, not to salvage it.
        """
        service, _ = _service(make_service)
        mocker.patch.object(service, "validate_image_url", return_value=True)
        huge = _png_bytes() + b"\0" * (service._REFERENCE_MAX_BYTES + 1)
        mocker.patch("urllib.request.OpenerDirector.open", return_value=_Resp(huge))

        assert service.fetch_reference_image(_IMG) is None

    def test_oversized_dimensions_are_scaled_down(self, make_service, animate_env, mocker) -> None:
        """GIVEN a very large image WHEN fetched THEN its long edge is capped."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "validate_image_url", return_value=True)
        mocker.patch(
            "urllib.request.OpenerDirector.open",
            return_value=_Resp(_png_bytes(4000, 2000)),
        )

        ref = service.fetch_reference_image(_IMG)

        assert ref is not None
        from PIL import Image

        with Image.open(io.BytesIO(ref.data)) as img:
            assert max(img.size) == service._REFERENCE_MAX_EDGE
            assert img.size[0] > img.size[1]

    @pytest.mark.parametrize(
        "size",
        [
            pytest.param((13_000, 13_000), id="past-pillow-error"),
            # Between 1x and 2x Pillow's limit it only warns; the cap must still hold.
            pytest.param((5_000, 5_000), id="pillow-warns-only"),
        ],
    )
    @pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
    def test_decompression_bomb_is_refused(self, make_service, animate_env, mocker, size) -> None:
        """GIVEN a tiny PNG claiming huge dimensions WHEN fetched THEN it is
        refused before its pixels are decoded."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "validate_image_url", return_value=True)
        mocker.patch("urllib.request.OpenerDirector.open", return_value=_Resp(huge_png(*size)))
        from PIL import ImageFile

        decode = mocker.spy(ImageFile.ImageFile, "load")

        assert service.fetch_reference_image(_IMG) is None
        decode.assert_not_called()

    def test_network_failure_returns_none(self, make_service, animate_env, mocker) -> None:
        """GIVEN the host is down WHEN fetched THEN it returns None rather than raising."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "validate_image_url", return_value=True)
        mocker.patch("urllib.request.OpenerDirector.open", side_effect=OSError("no route"))

        assert service.fetch_reference_image(_IMG) is None

    def test_redirects_are_refused(self, make_service, animate_env, mocker) -> None:
        """GIVEN the fetcher WHEN it is built THEN redirects are disabled.

        A 3xx Location could point at a private host that validation rejected
        on the original URL — the same fail-closed rule the image downloader
        already applies.
        """
        service, _ = _service(make_service)
        mocker.patch.object(service, "validate_image_url", return_value=True)
        mocker.patch("urllib.request.OpenerDirector.open", return_value=_Resp(_png_bytes()))
        build = mocker.spy(__import__("urllib.request", fromlist=["x"]), "build_opener")

        service.fetch_reference_image(_IMG)

        handler = build.call_args.args[0]
        assert handler.redirect_request() is None


class TestAnimateFormWithReference:
    """t2va refuses an image condition; the task has to move with the file."""

    def test_reference_switches_the_task_to_fl2va(self, make_service, animate_env) -> None:
        """GIVEN a reference WHEN the form is built THEN the task is fl2va."""
        service, _ = _service(make_service, animateAudio=True)

        extra = json.loads(service._animate_form("a cat", None, has_reference=True)["extra_params"])

        assert extra["task"] == "fl2va"

    def test_reference_overrides_audio_off(self, make_service, animate_env) -> None:
        """GIVEN audio off and a reference WHEN built THEN the task is still fl2va.

        The served partition takes ['fl2va', 't2va'] and nothing else, so a
        t2v/fl2v combination is not a choice the box offers.
        """
        service, _ = _service(make_service, animateAudio=False)

        extra = json.loads(service._animate_form("a cat", None, has_reference=True)["extra_params"])

        assert extra["task"] == "fl2va"

    def test_no_reference_keeps_the_text_task(self, make_service, animate_env) -> None:
        """GIVEN no reference WHEN built THEN the text-only task is unchanged."""
        service, _ = _service(make_service, animateAudio=True)

        extra = json.loads(service._animate_form("a cat", None)["extra_params"])

        assert extra["task"] == "t2va"


class TestMultipartFiles:
    """The image rides as a file part next to the plain fields."""

    def test_file_part_carries_filename_and_type(self, make_service, animate_env) -> None:
        """GIVEN a file WHEN encoded THEN the part names it and types it."""
        service, _ = _service(make_service)

        body, boundary = service._multipart_body(
            {"prompt": "a cat"},
            files={"input_reference": ("reference.jpg", b"\xff\xd8\xffdata", "image/jpeg")},
        )

        assert b'name="input_reference"; filename="reference.jpg"' in body
        assert b"Content-Type: image/jpeg" in body
        assert b"\xff\xd8\xffdata" in body
        assert body.endswith(f"--{boundary}--\r\n".encode())

    def test_plain_fields_are_unchanged_without_files(self, make_service, animate_env) -> None:
        """GIVEN no files WHEN encoded THEN the body is the plain-field form."""
        service, _ = _service(make_service)

        body, _boundary = service._multipart_body({"prompt": "a cat"})

        assert b"filename=" not in body
        assert b'name="prompt"' in body


class TestVideoGenerationWithReference:
    """The submission carries the file and says so in the stashed row."""

    def test_reference_bytes_reach_the_server(self, make_service, animate_env, mocker) -> None:
        """GIVEN a reference WHEN submitting THEN the bytes are in the request body."""
        from llm.service import ReferenceImage

        service, _ = _service(make_service)
        request = mocker.patch.object(
            service, "_animate_request", return_value=(200, {"id": "video_gen_1"})
        )
        mocker.patch.object(service, "_stash_timeout", return_value=True)

        result = service.video_generation(
            "a cat dancing",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            reference=ReferenceImage(data=b"\xff\xd8\xffcatbytes", extension="jpg"),
        )

        assert result.queued is True
        body = request.call_args.kwargs["body"]
        assert b"\xff\xd8\xffcatbytes" in body
        assert b'name="input_reference"' in body
        assert b'"task": "fl2va"' in body


class TestReferenceVisionCopy:
    """The planner gets its own small copy; the box gets the full-size one."""

    def test_vision_copy_is_a_data_url(self, make_service, animate_env) -> None:
        """GIVEN a reference WHEN a vision copy is made THEN it is an inline data URL.

        Inline, not the user's URL: the bytes are already fetched, validated
        and re-encoded here, so handing the provider the original link would
        re-open a fetch this path deliberately closed.
        """
        from llm.service import ReferenceImage

        service, _ = _service(make_service)
        ref = ReferenceImage(data=_png_jpeg_bytes(), extension="jpg")

        url = service.reference_vision_url(ref)

        assert url is not None
        assert url.startswith("data:image/jpeg;base64,")

    def test_vision_copy_is_downscaled(self, make_service, animate_env) -> None:
        """GIVEN a large reference WHEN a vision copy is made THEN it is shrunk.

        The planner needs to see what is in frame, not read the pixels — a
        full-size still would be the dominant cost of a turn that exists to
        write 150 words.
        """
        import base64
        import io

        from llm.service import ReferenceImage
        from PIL import Image

        service, _ = _service(make_service)
        ref = ReferenceImage(data=_png_jpeg_bytes(2400, 1200), extension="jpg")

        url = service.reference_vision_url(ref)

        raw = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as img:
            assert max(img.size) == service._REFERENCE_VISION_EDGE

    def test_undecodable_reference_has_no_vision_copy(self, make_service, animate_env) -> None:
        """GIVEN bytes that will not decode WHEN a copy is asked for THEN None."""
        from llm.service import ReferenceImage

        service, _ = _service(make_service)

        assert service.reference_vision_url(ReferenceImage(data=b"junk", extension="jpg")) is None


class TestVisionImageFilter:
    """Our own inline copy has to survive the URL filter that guards vision."""

    def test_our_data_url_survives(self, make_service, animate_env) -> None:
        """GIVEN an inline image WHEN filtered THEN it is kept."""
        service, _ = _service(make_service)

        assert service._filter_images(["data:image/jpeg;base64,Y2F0"]) == [
            "data:image/jpeg;base64,Y2F0"
        ]

    def test_non_image_data_url_is_dropped(self, make_service, animate_env) -> None:
        """GIVEN a data URL that is not an image WHEN filtered THEN it is dropped."""
        service, _ = _service(make_service)

        assert service._filter_images(["data:text/html;base64,PHNjcmlwdD4="]) is None
