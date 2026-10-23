# AI commands

You don't need commands to ask the bot questions. Mention it by name in a channel or send a PM:

```irc
<you>     VibeBot, what causes the northern lights?
<VibeBot> The northern lights happen when charged particles from the sun
          collide with gases in Earth's atmosphere...
```

Plain language covers questions, web lookups, images, code, service status, [aurora](aurora.md) forecasts, and saving a memory ("remember that I use Debian"). Reminders and scheduled tasks join that list where an operator has enabled `pendingTasksEnabled`. Managing memories, instructions, usage and context is command-only — use the commands below.

## Icons on a reply

Two icons can appear at the front of a reply. They are the bot telling you
something about how the answer was produced, and both can show at once.

| Icon | Meaning |
|------|---------|
| 🌐 | The answer used a live web lookup rather than the model's own knowledge |
| 🔁 | A content filter refused the image, so the bot reworded the prompt and tried again. You got a picture, but the wording behind it moved, so it may read a little differently from what you asked for |

## `ask`

Ask the AI a question. The bot remembers your recent conversation, so follow-up questions work naturally.

**Usage:** `@ask <question>`

```
@ask What is the capital of France?
@ask And what's its population?
@ask Explain quantum entanglement in plain terms
```

### Follow-up questions

The bot keeps conversation context per user, per channel. Your next `@ask` in the same channel carries the history forward:

```irc
<you>     @ask What's the fastest land animal?
<VibeBot> The cheetah, reaching speeds of 113 km/h.
<you>     @ask How does it compare to the fastest bird?
<VibeBot> The peregrine falcon dives at over 380 km/h...
```

Context expires after a few minutes of inactivity. Use `@forget` to clear it sooner.

### Vision

Include an image URL in your question and the bot analyses it:

```
@ask What's in this image? https://example.com/photo.jpg
@ask https://i.imgur.com/abc123.png Is this a bird or a plane?
```

More than one image URL in a question works too. The bot detects HTTP and HTTPS image links automatically.

### Custom instructions

Use `@instruct` to set persistent instructions that shape how `@ask` responds to you. See [Custom instructions](memory.md#custom-instructions).

## `draw`

Generate an image from a text description.

**Usage:** `@draw <prompt>`

```
@draw A sunset over mountains in watercolour style
@draw A golden retriever in a field of sunflowers, oil painting
```

`@draw` needs an authenticated account. The bot asks you to log in first if you aren't.

The shape follows the request: a poster, a book cover or a standing figure comes out portrait, a scene or a panorama landscape, an icon square. Say "portrait" or "wide" if you want to force it.

Name a real person, place or event and the bot looks up what they actually look like before it draws, because the image model knows nobody by name — "Churchill" on its own is a word, not a face. It keeps the name you wrote and adds the appearance beside it, so you get the man rather than a generic figure. This costs a few seconds on every `@draw`. In a channel with the verse enabled, characters the channel invented are described from canon instead, and canon wins where the two overlap.

Image generation applies content safety filters. If a filter blocks your prompt, the bot rewords it and tries again — once by default. The reworded version keeps your subject, what it is doing, the setting and the style you asked for, and changes only what the filter is likely to have objected to, so you should still recognise the picture you asked for. A reply carrying a reworded image is marked 🔁. If the reworded version is blocked too, the bot says so and tells you it already tried rewording, because rewording it again yourself is not the move — pick a different subject instead.

## `meme`

Caption a meme template. Name the template and write the captions, or just say what the meme is about and the bot picks a template and writes them.

**Usage:** `@meme <template> | <caption> [| <caption> ...]` or `@meme <what it's about>`

```
@meme drake | left on unread | left on read
@meme distracted boyfriend | me | a new side project | my actual job
@meme this is fine | | when the build is green on the third retry
@meme waiting for claude to finish
@meme spirit halloween costume of my dad
@meme list cat
```

It can go three ways, and they cost different amounts:

| You give it | What happens | Cost |
|-------------|--------------|------|
| A template name and the captions | The bot renders it itself. No model is called | Free |
| Just what the meme is about | A model picks a template from the catalog and writes the captions. Any captions you gave are kept as written | About a cent |
| Who or what should be *in* the picture | The bot also generates that picture, on top of the pick | About $0.02 per picture |

Every pick and picture shows up in `@usage` as a `meme` row at what it cost; a named template books a $0 row.

**Picking.** The picker reads names, example captions and topics, and real people and places in the request get the same look-up `@draw` does; on a verse channel, canon characters too. The reply is the image followed by `template | caption | caption`, so you can see what it chose and redo it by hand. If nothing fits it says so, with the closest names.

**Captions.** Separate them with `|`, in the order the template reads (top to bottom, left to right). An empty caption leaves that box blank; fewer captions than boxes is fine, more is an error that tells you how many the template takes.

**Finding a template.** `@meme <template>` alone shows its caption count and an example. `@meme list <word>` searches over a thousand templates by name, keyword, topic or example caption: `drake`, `db` (Distracted Boyfriend), `fry`, `doge`, `buzz`, `fine`, `trade-offer`, `absolute-cinema`. Words you share with a name match too, so `willy wonka` finds Condescending Wonka, and topics reach templates by situation: `monday` finds the Will Smith slap. When several match, the bot lists them rather than guessing. Your operator can add names the catalog lacks.

**Any image.** Paste an image URL in place of the name and give a top and a bottom caption:

```
@meme https://i.imgflip.com/1c1uej.jpg | y'all got any more of them | templates?
```

**Generated pictures.** Templates with a blank area, such as `spirit` (the Spirit Halloween costume), `same` (Pam's two sheets), `who-would-win`, `milk-carton` or `wanted-poster`, get a fresh picture cut to each blank, so `who would win, a goose or a sysadmin` gets one of each. On any other template an image-edit model changes who is in the picture. A refused picture leaves its area blank and the reply says so.

**Flags** go before the template. `--font <name>` changes the typeface (`thick`, the default; `thin`; `comic`). `--top` puts the captions in a white band above the picture, one row each.

```
@meme --font comic --top rollsafe | can't fail tests if you don't run them
```

`@meme` needs an authenticated account. Its rate limit is looser than `@draw`'s.

The same works in conversation: "vibebot make a drake meme: left on unread / left on read" uses the name and captions you gave, and "vibebot make a meme about waiting for CI" goes to the picker. If a name matches nothing and the picker declines, the bot lists the closest templates and asks which one you meant.

## `story`

Generate an illustrated page from a short brief and post a link when it's ready. The bot picks one of two modes from your brief:

- **Story:** an illustrated tale.
- **Explainer:** a concept explained with labelled diagrams, as a learning aid.

**Usage:** `@story <brief>`

```
@story an illustrated tale of the lads winning the pub quiz
@story explain how photosynthesis works, with diagrams
```

The page renders in the background; the bot posts the link when it finishes. `@story` needs an authenticated account, uses the same permission as `@draw`, and allows one page per account every five minutes. That limit covers everyone, admins and owners included, and it is shared with illustrated pages in [the verse](verse.md).

## `forget`

Clear your conversation context for the current channel, or a named one:

```
@forget
@forget #otherchannel
```

Run in a channel, `@forget` also clears that channel's shared recent history, so stale bot answers stop feeding everyone's follow-ups. Naming a different channel clears only your own thread there — you cannot wipe another channel's shared history from outside it.

## `usage`

See what the bot has spent on API calls this month, for you, another user, or a channel. Anyone can look up anyone.

**Usage:** `@usage [nick | #channel]`

```irc
<you>     @usage
<VibeBot> #lounge this month: $0.4210 (183 requests, rank 2/9 channels) |
          You: $0.0312 (14 requests, rank 5/31 users) | Context: 6/20 msgs,
          expires in 4m
<you>     @usage carol
<VibeBot> carol in #lounge this month: $0.1180 (40 requests, rank 1/31 users)
<you>     @usage #linux
<VibeBot> #linux this month: $1.0920 (512 requests, rank 1/9 channels)
```

In a channel, a nick shows that user's spending in that channel; by private message it covers every channel. A bare `@usage` by private message is the bot-wide overview, and only admins can run it. The month is a calendar month: the figures start over at midnight UTC on the 1st.

The bot carries no usage tool and no usage figures in its prompt, so asking in conversation can get you a made-up number.

## `model`

Show which model answered the bot's last reply in a channel, and which chat model the channel is set to use.

**Usage:** `@model [<channel>]`

```irc
<you>     @model
<VibeBot> Last reply in #linux: anthropic/claude-haiku-5-5, 2m ago. Set to:
          anthropic/claude-haiku-5-5.
```

`@model #tv` asks about another channel.

The two can differ: a runtime `@config` change is lost when the bot restarts, and the reply after that runs on whatever `bot.conf` holds. The bot forgets the last reply on restart too, and says so.

## `channels`

List the busiest public channels on the network, with user counts and topics.

**Usage:** `@channels [--min <n>] [<channel> | <glob>]`

```
@channels                  → the ten busiest channels
@channels #linux           → that channel's count and topic
@channels --min 5 #linux*  → matching channels with at least 5 users
```

You can also just ask: "vibebot, is `#help` active?" or "vibebot, who is eck?" The bot looks it up (`LIST`, `NAMES`, or `WHOIS`) rather than guessing.

## `names`

List who is in a channel at this moment — including channels the bot is not in.

**Usage:** `@names [<channel>]`

```
@names          → the current channel
@names #linux   → another channel
```

Secret and private channels come back empty, exactly as they would for `/names`.

## Rate limits

Each command family carries its own rate limit. Your tier is unregistered or registered depending on whether you are authenticated, or trusted if your operator has granted you the `trusted` capability. Media commands have the tightest limits: `@draw` and `@story` for images. If you reach a limit, the bot tells you; wait and retry. Admins and owners are exempt from these limits, but not from `@story`'s one page every five minutes. The numbers are in [Getting started](getting-started.md#rate-limits).

## The verse

Channels with the verse enabled host a shared story world where you play an avatar. Joining, `@rp`, tales and the rest are on [The verse](verse.md).
