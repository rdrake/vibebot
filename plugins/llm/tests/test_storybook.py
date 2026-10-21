"""Tests for the verse storybook tool."""

from __future__ import annotations

from llm.identity import Caller


def test_storybook_illustration_rules_scale_with_budget():
    from llm.service import _storybook_illustration_rules as rules

    # Text-only: no illustrations at all.
    zero = rules(0)
    assert "PROSE page" in zero and "NO illustrations" in zero
    assert "empty" in zero
    # Prose-first: at most one, zero is fine, ~5 paragraphs.
    one = rules(1)
    assert "PROSE-FIRST" in one and "AT MOST ONE" in one
    assert "five paragraphs" in one
    # Illustrated: generous, up to the cap.
    many = rules(4)
    assert "GENEROUSLY" in many and "2 to 4" in many


def test_generate_storybook_honours_max_images_override(make_service, mocker, tmp_path):
    """An explicit max_images overrides verseStorybookMaxImages for both the
    prompt budget and the image cap (the ambient prose-first path uses this)."""
    service, plugin = make_service(
        httpRoot=str(tmp_path), httpUrlBase="http://h/llm", verseStorybookMaxImages=5
    )
    illos = [{"id": i, "caption": f"c{i}", "image_prompt": f"p{i}"} for i in range(1, 6)]
    markers = " ".join(f"[[illustration:{i}]]" for i in range(1, 6))
    struct = mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={"title": "T", "story_markdown": markers, "illustrations": illos},
    )
    from llm.service import ImageResult

    gen = mocker.patch.object(
        service,
        "_attempt_image_generation",
        return_value=ImageResult(content="ok", url="http://h/llm/i.jpg"),
    )
    service.generate_storybook("b", channel="#c", persona="v", max_images=1)
    # Only one image drawn despite five markers + a config cap of five.
    assert gen.call_count == 1
    # And the struct generation was told the same budget.
    assert struct.call_args.kwargs["max_images"] == 1


def test_generate_story_struct_prose_first_prompt_when_budget_low(make_service, mocker):
    service, plugin = make_service(verseStorybookMaxImages=5, verseStorybookMaxChars=6000)
    ask = mocker.patch.object(service, "_ask_completion", return_value="not json")
    service._generate_story_struct("the lads", channel="#c", persona="v", max_images=1)
    system_prompt = ask.call_args.args[0]
    assert "PROSE-FIRST" in system_prompt
    assert "GENEROUSLY" not in system_prompt


def test_extract_json_object():
    from llm.service import _extract_json_object as ex

    assert ex('{"a": 1}') == {"a": 1}
    assert ex('Here ye go! ```json\n{"a": 2}\n``` enjoy') == {"a": 2}
    assert ex('prose {"a": {"b": 3}} more prose') == {"a": {"b": 3}}
    assert ex("not json at all") is None
    assert ex("") is None


def test_embed_illustrations_basic():
    from llm.service import LLMService as S

    md = "Intro [[illustration:1]] middle [[illustration:11]] end"
    illos = {1: ("cat", "u1.jpg"), 11: ("dog", "u11.jpg")}
    out, used = S._embed_illustrations(md, illos)
    assert "![cat](u1.jpg)" in out and "*cat*" in out
    assert "![dog](u11.jpg)" in out
    assert used == {1, 11}


def test_embed_duplicate_marker_first_wins():
    from llm.service import LLMService as S

    md = "[[illustration:2]] x [[illustration:2]]"
    out, used = S._embed_illustrations(md, {2: ("c", "u.jpg")})
    assert out.count("![c](u.jpg)") == 1
    assert "[[illustration:2]]" not in out
    assert used == {2}


def test_embed_orphan_marker_removed():
    from llm.service import LLMService as S

    out, used = S._embed_illustrations("a [[illustration:9]] b", {1: ("c", "u")})
    assert "[[illustration:9]]" not in out and used == set()


def test_embed_strips_user_injected_image():
    from llm.service import LLMService as S

    md = "![evil](http://evil/p.png) tale [[illustration:1]]"
    out, used = S._embed_illustrations(S._strip_untrusted_markup(md), {1: ("c", "u.jpg")})
    assert "evil" not in out
    assert "![c](u.jpg)" in out


def test_generate_story_struct_parses_and_validates(make_service, mocker):
    service, plugin = make_service(verseStorybookMaxImages=3, verseStorybookMaxChars=6000)
    payload = (
        "Here is your tale! ```json\n"
        '{"title":"The Tin Fox","story_markdown":"Once [[illustration:1]] fin.",'
        '"illustrations":[{"id":1,"caption":"a fox","image_prompt":"a tin fox"}]}\n```'
    )
    mocker.patch.object(service, "_ask_completion", return_value=payload)
    out = service._generate_story_struct(
        "spin a tale",
        channel="#c",
        persona="voice",
    )
    assert out["title"] == "The Tin Fox"
    assert out["illustrations"][0]["id"] == 1
    assert "[[illustration:1]]" in out["story_markdown"]


def test_generate_story_struct_injects_world_context(make_service, mocker):
    """A non-empty world_context grounds the generator: the canon roster must
    reach the system prompt so the model uses established names."""
    service, plugin = make_service(verseStorybookMaxImages=3, verseStorybookMaxChars=6000)
    payload = (
        '{"title":"T","story_markdown":"a [[illustration:1]]",'
        '"illustrations":[{"id":1,"caption":"c","image_prompt":"p"}]}'
    )
    ask = mocker.patch.object(service, "_ask_completion", return_value=payload)
    service._generate_story_struct(
        "the stinky lads",
        channel="#c",
        persona="voice",
        world_context="- Assgas Archie: Y11 windbag",
    )
    system_prompt = ask.call_args.args[0]
    assert "WORLD & CAST" in system_prompt
    assert "Assgas Archie: Y11 windbag" in system_prompt


def test_generate_story_struct_no_world_block_when_empty(make_service, mocker):
    """Empty world_context (non-verse caller) leaves the prompt unchanged —
    no dangling WORLD header."""
    service, plugin = make_service(verseStorybookMaxImages=3, verseStorybookMaxChars=6000)
    ask = mocker.patch.object(service, "_ask_completion", return_value="not json")
    service._generate_story_struct(
        "x",
        channel="#c",
        persona="v",
    )
    assert "WORLD & CAST" not in ask.call_args.args[0]


def test_generate_story_struct_injects_scene_context(make_service, mocker):
    """A non-empty scene_context is folded into the user prompt so a thin brief
    draws on what actually happened. Empty leaves the brief untouched."""
    service, plugin = make_service(verseStorybookMaxImages=3, verseStorybookMaxChars=6000)
    ask = mocker.patch.object(service, "_ask_completion", return_value="not json")
    service._generate_story_struct(
        "recap today",
        channel="#c",
        persona="v",
        scene_context="alice: we won the quiz\nbob: barely",
    )
    user_prompt = ask.call_args.args[1]
    assert "RECENT SCENE" in user_prompt
    assert "alice: we won the quiz" in user_prompt

    ask.reset_mock()
    service._generate_story_struct("recap today", channel="#c", persona="v")
    assert "RECENT SCENE" not in ask.call_args.args[1]


def test_generate_story_struct_retries_then_fails(make_service, mocker):
    service, plugin = make_service(verseStorybookMaxImages=3, verseStorybookMaxChars=6000)
    m = mocker.patch.object(service, "_ask_completion", return_value="no json here")
    out = service._generate_story_struct(
        "x",
        channel="#c",
        persona="v",
    )
    assert out is None
    assert m.call_count >= 3  # initial + >=2 retries


def test_validate_story_obj_drops_bad_illustrations():
    from llm.service import LLMService as S

    obj = {
        "title": "T",
        "story_markdown": "body",
        "illustrations": [
            {"id": 1, "caption": "ok", "image_prompt": "p"},
            {"id": "bad", "caption": "x", "image_prompt": "y"},
            {"id": 2, "caption": "no prompt", "image_prompt": "  "},
        ],
    }
    out = S._validate_story_obj(obj)
    assert [i["id"] for i in out["illustrations"]] == [1]


def test_validate_story_obj_requires_title_and_story():
    from llm.service import LLMService as S

    assert S._validate_story_obj({"title": "", "story_markdown": "x"}) is None
    assert S._validate_story_obj({"title": "T", "story_markdown": "  "}) is None
    assert S._validate_story_obj("not a dict") is None


def test_validate_story_obj_captures_style():
    from llm.service import LLMService as S

    out = S._validate_story_obj(
        {"title": "T", "style": "  muted watercolour  ", "story_markdown": "x"}
    )
    assert out["style"] == "muted watercolour"
    # Missing / non-string style degrades to "" rather than failing.
    assert S._validate_story_obj({"title": "T", "story_markdown": "x"})["style"] == ""
    assert S._validate_story_obj({"title": "T", "style": 7, "story_markdown": "x"})["style"] == ""


def test_generate_storybook_applies_model_style_to_every_image(make_service, mocker, tmp_path):
    """The model-authored style is prepended to EVERY illustration prompt so the
    stateless image model stays consistent panel to panel."""
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={
            "title": "T",
            "style": "muted watercolour, the Lads: three grubby boys",
            "story_markdown": "a [[illustration:1]] b [[illustration:2]] c",
            "illustrations": [
                {"id": 1, "caption": "c1", "image_prompt": "kickoff"},
                {"id": 2, "caption": "c2", "image_prompt": "the winning goal"},
            ],
        },
    )
    from llm.service import ImageResult

    gen = mocker.patch.object(
        service,
        "_attempt_image_generation",
        return_value=ImageResult(content="ok", url="http://h/llm/img.jpg"),
    )
    service.generate_storybook(
        "b",
        channel="#c",
        persona="v",
    )
    prompts = [c.args[0] for c in gen.call_args_list]
    assert len(prompts) == 2
    assert all(p.startswith("muted watercolour, the Lads: three grubby boys: ") for p in prompts)
    assert "kickoff" in prompts[0] and "the winning goal" in prompts[1]


def test_generate_storybook_falls_back_to_default_style(make_service, mocker, tmp_path):
    """No model style → the generic fairytale prefix, preserving prior behaviour."""
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={
            "title": "T",
            "style": "",
            "story_markdown": "a [[illustration:1]] b",
            "illustrations": [{"id": 1, "caption": "c", "image_prompt": "p"}],
        },
    )
    from llm.service import ImageResult

    gen = mocker.patch.object(
        service,
        "_attempt_image_generation",
        return_value=ImageResult(content="ok", url="http://h/llm/img.jpg"),
    )
    service.generate_storybook(
        "b",
        channel="#c",
        persona="v",
    )
    assert gen.call_args.args[0] == "storybook illustration, painted fairytale style: p"


def test_generate_storybook_embeds_and_saves(make_service, mocker, tmp_path):
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={
            "title": "The Tin Fox",
            "story_markdown": "Once [[illustration:1]] upon a time.",
            "illustrations": [{"id": 1, "caption": "a fox", "image_prompt": "a tin fox"}],
        },
    )
    from llm.service import ImageResult

    mocker.patch.object(
        service,
        "_attempt_image_generation",
        return_value=ImageResult(content="ok", url="http://h/llm/img_fox.jpg"),
    )
    res = service.generate_storybook(
        "brief",
        channel="#c",
        persona="v",
    )
    assert res is not None and res.title == "The Tin Fox" and res.image_count == 1
    page = (tmp_path / res.url.split("/")[-1]).read_text()
    assert "<img" in page and 'src="img_fox.jpg"' in page  # bare filename embedded


def test_generate_storybook_image_failure_drops_marker(make_service, mocker, tmp_path):
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={
            "title": "T",
            "story_markdown": "a [[illustration:1]] b",
            "illustrations": [{"id": 1, "caption": "c", "image_prompt": "p"}],
        },
    )
    from llm.service import ImageResult

    mocker.patch.object(
        service,
        "_attempt_image_generation",
        return_value=ImageResult(content="blocked", url=None, error="safety"),
    )
    res = service.generate_storybook(
        "b",
        channel="#c",
        persona="v",
    )
    assert res is not None and res.image_count == 0
    page = (tmp_path / res.url.split("/")[-1]).read_text()
    assert "[[illustration:1]]" not in page


def test_generate_storybook_image_none_drops_marker(make_service, mocker, tmp_path):
    service, plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={
            "title": "T",
            "story_markdown": "a [[illustration:1]] b",
            "illustrations": [{"id": 1, "caption": "c", "image_prompt": "p"}],
        },
    )
    mocker.patch.object(service, "_attempt_image_generation", return_value=None)
    res = service.generate_storybook(
        "b",
        channel="#c",
        persona="v",
    )
    assert res is not None and res.image_count == 0


def test_generate_storybook_caps_images(make_service, mocker, tmp_path):
    service, plugin = make_service(
        httpRoot=str(tmp_path), httpUrlBase="http://h/llm", verseStorybookMaxImages=3
    )
    illos = [{"id": i, "caption": f"c{i}", "image_prompt": f"p{i}"} for i in range(1, 6)]
    markers = " ".join(f"[[illustration:{i}]]" for i in range(1, 6))
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={"title": "T", "story_markdown": markers, "illustrations": illos},
    )
    from llm.service import ImageResult

    gen = mocker.patch.object(
        service,
        "_attempt_image_generation",
        return_value=ImageResult(content="ok", url="http://h/llm/i.jpg"),
    )
    service.generate_storybook(
        "b",
        channel="#c",
        persona="v",
    )
    assert gen.call_count == 3  # capped at verseStorybookMaxImages


def test_generate_storybook_none_when_story_fails(make_service, mocker):
    service, plugin = make_service()
    mocker.patch.object(service, "_generate_story_struct", return_value=None)
    assert (
        service.generate_storybook(
            "b",
            channel="#c",
            persona="v",
        )
        is None
    )


def test_resolves_to_public(make_service, mocker):
    service, plugin = make_service()
    mocker.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 443))])
    assert service._resolves_to_public("http://rebind.example/x.png") is False
    mocker.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 443))])
    assert service._resolves_to_public("http://example.com/x.png") is True
    # no host / unresolvable → False
    assert service._resolves_to_public("not-a-url") is False


def test_resolves_to_public_false_on_unicode_encode_error(make_service, mocker):
    """getaddrinfo raises UnicodeEncodeError (not an OSError subclass) on a
    hostname that fails IDNA encoding; the guard must fail closed rather than
    let the exception escape."""
    service, plugin = make_service()
    mocker.patch(
        "socket.getaddrinfo",
        side_effect=UnicodeEncodeError("idna", "x", 0, 1, "label empty or too long"),
    )
    assert service._resolves_to_public("http://example.com/x.png") is False


def test_storybook_job_records_canon_event(plugin_env, tmp_path, mocker):
    from llm.verse.store import VerseStore

    from .conftest import make_registry_side_effect

    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = make_registry_side_effect({"verseEnabled": True})
    store = VerseStore(tmp_path / "verse", "#afnet")
    avatar_id = store.opt_in_avatar(nick="alice", account="alice-acct", instruct_text="").entity_id
    mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
    # Neutralize the async render/post job; the canon-record runs synchronously before it.
    mocker.patch.object(plugin, "_llm_executor")

    plugin._submit_storybook_job(
        caller=Caller.from_row("alice", "alice-acct", "#afnet"),
        persona="",
        brief="a tale of dragons",
    )

    events = store.recent_events(limit=10)
    assert len(events) == 1
    assert "dragons" in events[0].summary
    assert avatar_id in events[0].entity_ids


def test_storybook_job_no_avatar_skips_record(plugin_env, tmp_path, mocker):
    from llm.verse.store import VerseStore

    from .conftest import make_registry_side_effect

    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = make_registry_side_effect({"verseEnabled": True})
    store = VerseStore(tmp_path / "verse", "#afnet")  # nobody opted in
    mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
    mocker.patch.object(plugin, "_llm_executor")

    plugin._submit_storybook_job(
        caller=Caller.from_row("ghost", None, "#afnet"), persona="", brief="x"
    )

    assert store.recent_events(limit=10) == []


def test_storybook_job_non_verse_channel_skips_store(plugin_env, mocker):
    """@story in a non-verse channel (or PM) must not lazily create a verse DB."""
    plugin, irc, msg = plugin_env  # verseEnabled defaults to False
    get_store = mocker.patch.object(plugin, "_get_or_create_verse_store")
    mocker.patch.object(plugin, "_llm_executor")

    plugin._submit_storybook_job(
        caller=Caller.from_row("alice", "acct", "#test"), persona="", brief="a tale"
    )
    plugin._submit_storybook_job(
        caller=Caller.from_row("alice", "acct", "testbot"), persona="", brief="a tale"
    )

    get_store.assert_not_called()


def test_storybook_job_logs_usage_on_success(plugin_env, mocker):
    from llm.service import StorybookResult

    plugin, irc, msg = plugin_env
    plugin.llm_service.generate_storybook.return_value = StorybookResult(
        url="http://h/llm/story_x.html",
        title="A Tale",
        image_count=2,
        dropped=0,
        prompt_tokens=10,
        completion_tokens=20,
        cost=0.08,
        model="dall-e-3",
    )
    mocker.patch.object(plugin._llm_executor, "submit", side_effect=lambda label, fn, *a: fn(*a))
    mocker.patch("llm.plugin.world.ircs", [irc])
    mocker.patch.object(plugin, "_safe_queue", return_value=True)

    plugin._submit_storybook_job(
        caller=Caller.from_row("alice", "acct", "#test"), persona="", brief="a tale"
    )

    plugin.db.log_usage.assert_called_once()
    kwargs = plugin.db.log_usage.call_args.kwargs
    assert kwargs["command"] == "story"
    assert kwargs["cost"] == 0.08
    assert kwargs["model"] == "dall-e-3"


def test_storybook_job_pm_delivers_to_nick(plugin_env, mocker):
    """A PM @story delivers the link to the requesting nick, not the void."""
    from llm.service import StorybookResult

    plugin, irc, msg = plugin_env
    plugin.llm_service.generate_storybook.return_value = StorybookResult(
        url="http://h/llm/story_x.html", title="A Tale", image_count=0, dropped=0
    )
    mocker.patch.object(plugin._llm_executor, "submit", side_effect=lambda label, fn, *a: fn(*a))
    mocker.patch("llm.plugin.world.ircs", [irc])
    safe_queue = mocker.patch.object(plugin, "_safe_queue", return_value=True)
    privmsg = mocker.patch.object(plugin, "_safe_privmsg", side_effect=lambda t, m: (t, m))

    # A PM caller's reply target is their nick — never in state.channels.
    plugin._submit_storybook_job(
        caller=Caller.from_row("alice", "acct", "testbot"), persona="", brief="a tale"
    )

    assert safe_queue.called
    target, text = privmsg.call_args.args
    assert target == "alice"
    assert "story_x.html" in text


def test_generate_storybook_books_a_refused_draw_that_billed(make_service, mocker, tmp_path):
    """GIVEN one draw raises a refusal the provider still charged WHEN the
    page is built THEN that refusal's cost is in the page total, not lost."""
    service, _plugin = make_service(httpRoot=str(tmp_path), httpUrlBase="http://h/llm")
    mocker.patch.object(
        service,
        "_generate_story_struct",
        return_value={
            "title": "T",
            "story_markdown": "a [[illustration:1]] b [[illustration:2]] c",
            "illustrations": [
                {"id": 1, "caption": "c1", "image_prompt": "fine"},
                {"id": 2, "caption": "c2", "image_prompt": "refused"},
            ],
        },
    )
    from llm.service import ImageResult

    def draw(prompt, *_args, **_kwargs):
        if prompt.endswith("refused"):
            raise RuntimeError("moderation: billed")
        return ImageResult(content="ok", url="http://h/llm/img.jpg", cost=0.07)

    mocker.patch.object(service, "_attempt_image_generation", side_effect=draw)
    billed = mocker.patch.object(service, "_billed_failure_cost", return_value=0.05)

    result = service.generate_storybook("b", channel="#c", persona="v")

    assert result is not None
    assert result.image_count == 1
    assert abs(result.cost - 0.12) < 1e-9
    assert isinstance(billed.call_args.args[0], RuntimeError)
