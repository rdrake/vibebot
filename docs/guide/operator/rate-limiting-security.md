# Rate limiting and security

## Rate limiting

Rate limits are checked per user and per command. The `enforceRateLimits` setting (global) controls whether limits block requests or only observe them:

| Value | Behaviour |
|-------|-----------|
| `True` (default) | Requests over the limit are blocked |
| `False` | Limits are tracked and logged but not enforced (observe-only mode) |

### Tiers

Users fall into one of four tiers, checked from most to least privileged:

| Tier | How to qualify | Behaviour |
|------|----------------|-----------|
| Owner or admin | Limnoria `owner` or `admin` capability | Always exempt |
| Trusted | Has the `trusted` capability | Uses the `Trusted` settings |
| Registered | Authenticated with the network | Uses the base settings |
| Unregistered | Not identified | Uses the `Unreg` settings |

To grant the `trusted` capability, see the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html).

### Setting pattern

The rate-limited commands are `ask`, `draw`, `meme`, and `story`. Each has three setting pairs, one per non-exempt tier:

- `{cmd}RateLimitCount` and `{cmd}RateLimitWindow`: registered users.
- `{cmd}TrustedRateLimitCount` and `{cmd}TrustedRateLimitWindow`: trusted users.
- `{cmd}UnregRateLimitCount` and `{cmd}UnregRateLimitWindow`: unregistered users.

A count of `0` blocks nothing; it disables the limit for that command and tier. All rate-limit settings are global.

Buckets are keyed by account name for identified users and by nick for everyone else, so an unidentified user can reset an `ask` bucket by changing nick. `draw` and `story` are unaffected: they refuse to run without an account at all.

### Default limits

| Command | Registered | Trusted | Unregistered |
|---------|------------|---------|--------------|
| `ask` | 15 per 60 s | 15 per 60 s | 15 per 60 s |
| `draw` | 2 per 300 s | 5 per 60 s | 1 per 3600 s |
| `meme` | 6 per 60 s | 12 per 60 s | 2 per 600 s |
| `story` | 2 per 300 s | 5 per 60 s | 1 per 3600 s |

`draw`, `meme`, and `story` also require an authenticated account regardless of tier, so media generation is always attributable and the unregistered tier is unreachable in practice. It stays a small positive number rather than `0`, because `0` means "no limit" and would hand accountless users unlimited access to the most expensive commands.

`@rp` has no keys of its own; it draws on the `ask` bucket.

## Security

### Capability-based access control

The AI and verse command surfaces require a Limnoria capability:

| Capability | Gates |
|------------|-------|
| `llm.ask` | `@ask`, `@forget`, `@remind`, and general assistant tool access |
| `llm.draw` | `@draw`, `@meme`, `@story`, the image-generation tool, and the `make_meme` tool |
| `llm.verse` | Verse participation: `@verseopt`, `@rp`, `@verse`, `@look`, `@who` |
| `llm.verse.edit` | Canon editing: `@versedit`, `@canon`, the `verse_edit` tool |
| `llm.verse.gm` | Game-moderator (GM) operations: `@versedump`, `@versepurge`, `@versecompact` |

`@memories`, `@instruct`, `@avatar`, and `@usage` carry no plugin capability; anyone who can talk to the bot can use them on their own data. Their privileged forms — `@memories <nick>`, `@memories cleanup <nick>`, and the global `@usage` overview by PM — check Limnoria's stock `owner` or `admin` capability instead, as does `@remind admin`.

!!! warning "The default is allow, not deny"

    None of the `llm.*` capabilities is registered default-deny, and this deployment ships `supybot.capabilities.default` at its stock value of `True`. Limnoria grants any capability a user neither holds nor anti-holds, so on a fresh install every user — identified or not — passes all six checks above. That includes `llm.verse.gm`, which gates `@versepurge`, the command that deletes a channel's verse database file.

Lock the destructive surfaces down before you enable verse. Add the anti-capability globally, then hand it back per account:

```
@config supybot.capabilities [config supybot.capabilities] -llm.verse.gm -llm.verse.edit
@admin capability add <account> llm.verse.edit
```

The nested `[config supybot.capabilities]` reads the current list back, so the stock entries survive the write. A capability held explicitly on an account beats the global anti-capability, and the `owner` capability satisfies every check regardless of either. The same pattern restricts `llm.ask` and `llm.draw`. See the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html).

### URL validation

When users include image URLs in `@ask` prompts for vision, the bot validates them:

- **Scheme check.** Only `http://` and `https://` URLs are accepted. Schemes such as `javascript:`, `data:`, and `file:` are blocked.
- **Path traversal.** URLs containing `..` in the path are rejected.
- **Request-forgery protection.** URLs resolving to private, loopback, link-local, or reserved IP addresses are blocked, which prevents server-side request forgery (SSRF) against the bot's network. The check fails closed: if DNS resolution fails, the URL is rejected.
- **Extension check.** Only recognised image extensions are accepted: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`.

### Input and output sanitisation

Model output is sanitised before it reaches IRC, and incoming text before Limnoria parses it:

- **IRC command injection.** Lines starting with a configured command prefix get a leading space, so neither this bot nor another can be driven by model output. `commandPrefixes` defaults to `. @` — the conventional `.` and VibeBot's own `@`; add any others the network uses.
- **Wire-structural bytes.** NUL and the CTCP delimiter `\x01` are deleted from every outbound model line, the `ACTION` path included, so model text cannot forge a CTCP. Leaked model sentinels such as `<|eos|>` are stripped in the same pass.
- **API key scrubbing.** Error messages are scrubbed of API keys before display.
- **HTML sanitisation.** Every page the bot publishes — code pastes, long answers, storybooks, verse dumps — passes through `nh3`, an allowlist-based sanitiser restricted to `http`, `https`, and `mailto` URLs. A second pass drops any `<img>` whose source is neither a bare relative filename nor under `httpUrlBase` or `imageUploadUrl`, so model output cannot embed a tracking pixel or fire a request when the page is opened.
- **Inbound control characters.** Incoming PRIVMSG text has its C0 control characters removed, ESC among them, and unbalanced `[` or `]` replaced with full-width equivalents, before Limnoria's tokenizer sees it.

### Prompt injection defences

- **Structured system prompts.** Prompts separate instructions from context, with a preamble telling the model to ignore instructions embedded in user-provided content.
- **Channel topics as untrusted data.** Topics arrive as user messages, not system messages, because any channel member can set them. Line separators inside a topic are collapsed to spaces so it cannot open a new instruction line, and the text is trimmed to 300 characters. The anti-injection preamble calls the topic out explicitly as an attack surface.
- **Input length limit.** `maxPromptLength` (default 10,000 characters) is enforced on image prompts, not on the chat path — `@ask` and nick-addressed text reach the model unmeasured.
- **Tool-loop cap.** `metaMaxSteps` (default 12) bounds tool-call round trips per turn, so a prompt cannot drive an unbounded tool loop.

### Verse write safety

The model's in-band canon tool, `verse_edit`, is restricted to five operations: `add_entity`, `add_event`, `add_relation`, `set_attribute`, and `update_entity`. It can never delete an entity, retire one, or edit a recorded event. It can, however, rename an entity, rewrite its summary, and overwrite an attribute value, so a hijacked turn can still corrupt canon in place. `set_attribute` refuses reserved lifecycle keys, and both it and `add_relation` refuse retired entities.

Destructive operations live behind operator commands with the `llm.verse.edit` and `llm.verse.gm` capabilities, and `@versepurge` adds a two-step token confirmation. See [The verse](verse.md).
