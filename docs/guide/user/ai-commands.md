# AI commands

You don't need commands to ask the bot questions. Mention it by name in a channel or send a PM:

```
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

---

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

```
@ask What's the fastest land animal?
  → The cheetah, reaching speeds of 113 km/h.
@ask How does it compare to the fastest bird?
  → The peregrine falcon dives at over 380 km/h...
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

---

## `code`

Generate code, delivered as a syntax-highlighted web page. The bot can search the web for current documentation before generating.

**Usage:** `@code <request>`

```
@code Python function to calculate Fibonacci numbers
@code JavaScript async fetch with error handling
```

### Iterating on code

Conversation context carries over, so you can refine across requests:

```
@code Python function to sort a list of dictionaries by key
  → https://bot.example.com/llm/abc123.html
@code Now add type hints and handle empty lists
  → https://bot.example.com/llm/def456.html
```

If the page cannot be saved, the bot answers without a link — ask your operator to check `httpRoot` and `httpUrlBase`. A `@code` request that times out is retried in the background and delivered later, pasted inline if the page still cannot be saved.

---

## `draw`

Generate an image from a text description.

**Usage:** `@draw <prompt>`

```
@draw A sunset over mountains in watercolour style
@draw A golden retriever in a field of sunflowers, oil painting
```

`@draw` needs an authenticated account. The bot asks you to log in first if you aren't.

The shape follows the request: a poster, a book cover or a standing figure comes out portrait, a scene or a panorama landscape, an icon square. Say "portrait" or "wide" if you want to force it.

Name a real person, place or event and the bot looks up what they actually look like before it draws, because the image model knows nobody by name — "Churchill" on its own is a word, not a face. It keeps the name you wrote and adds the appearance beside it, so you get the man rather than a generic figure. This costs a few seconds on every `@draw`, and your operator can turn it off per channel. In a channel with the verse enabled, characters the channel invented are described from canon instead, and canon wins where the two overlap.

Image generation applies content safety filters. If a filter blocks your prompt, the bot rewords it and tries again — once by default (`drawAutoRewriteMax`). The reworded version keeps your subject, what it is doing, the setting and the style you asked for, and changes only what the filter is likely to have objected to, so you should still recognise the picture you asked for. A reply carrying a reworded image is marked 🔁. If the reworded version is blocked too, the bot says so and tells you it already tried rewording, because rewording it again yourself is not the move — pick a different subject instead.

---

## `meme`

Caption a meme template. Name the template and write the captions, or just say what the meme is about and the bot picks a template and writes them.

**Usage:** `@meme <template> | <caption> [| <caption> ...]` or `@meme <what it's about>`

```
@meme drake | left on unread | left on read
@meme distracted boyfriend | me | a new side project | my actual job
@meme this is fine | | when the build is green on the third retry
@meme waiting for claude to finish
@meme list cat
```

When the first part names no template the bot knows, a model reads the whole request against the catalog (names, example captions, topics), picks a template and writes the captions — any captions you gave after a `|` are kept verbatim. Real people and places named in the request get the same look-up `@draw` does (their appearance goes into the picture edit), and on a verse channel the canon block too. The reply is the image followed by `template | caption | caption`, so you can see what it chose and redo it by hand. If nothing in the catalog fits it says so, with the closest names. This costs about a cent per pick; naming a template costs nothing.

Separate the captions with `|`, in the order the template reads (top to bottom, left to right). An empty caption leaves that box blank; fewer captions than boxes is fine, more is an error that tells you how many the template takes. `@meme <template>` alone shows the template's caption count and an example, and `@meme list <word>` finds templates by name, keyword, topic, or example caption — there are over a thousand, drawn by the bot itself, with names like `drake`, `db` (Distracted Boyfriend), `fry`, `doge`, `buzz`, `fine`, `trade-offer`, and `absolute-cinema`.

Many templates are named by the quote, so the bot also matches on the words you share with a name: `willy wonka` finds Condescending Wonka, `elmo fire` finds Elmo. When two templates tie it lists both rather than guessing. Every template also carries topic tags — the characters people name it by and the situations it fits — so `monday` reaches the Will Smith slap and `harry potter` reaches Hagrid; a topic shared by several templates (`cat`, `star wars`) lists them instead. A topic nothing matches says so — paste an image URL instead. Your operator can add names the catalog lacks — Tyrone Biggums is filed under the quote, `yallgot`, and `tyrone` is aliased to it out of the box.

Any image works as a template: paste its URL in place of the name and give a top and a bottom caption.

Flags go before the template. `--font <name>` changes the typeface (`thick`, the default; `thin`; `comic`). `--top` ignores the template's boxes and puts the captions in a white band above the picture, one row each.

```
@meme --font comic --top rollsafe | can't fail tests if you don't run them
```

Say who or what should be in the picture and the bot draws it too. Templates with a blank picture area — `spirit` (the Spirit Halloween costume), `same` (Pam's two sheets), `who-would-win`, `milk-carton`, `wanted-poster`, `blank-t-shirt`, `youtube-video-template`, `blank-starter-pack`, `i-m-in-danger-blank`, `blank-wild-pokemon` — get a freshly generated picture per area, cut to the blank's exact outline (and warped onto Pam's tilted sheets) before the captions go on; for two areas the picker writes two pictures, so `who would win, a goose or a sysadmin` gets one of each. On any other template the captioned meme goes through an image-edit model that changes who is in the picture. About $0.02 per picture, booked in `@usage` beside the meme; a refused picture leaves its area blank and the reply says so. No flags to type: the `|` form with a template you name is the only path that never calls a model.

```
@meme spirit halloween costume of my dad
@meme distracted boyfriend where the boyfriend is a sysadmin and the girl in red is a glowing kubernetes logo
```

```
@meme https://i.imgflip.com/1c1uej.jpg | y'all got any more of them | templates?
```

`@meme` needs an authenticated account. Memes cost nothing to make, so the rate limit is looser than `@draw`'s; each one still shows in `@usage` as a $0 `meme` row.

The same thing works in conversation: "vibebot make a drake meme: left on unread / left on read" calls the `make_meme` tool with the name and captions you gave, and "vibebot make a meme about waiting for CI" hands the request to the same picker. If a name matches nothing and the picker declines, the bot lists the closest templates and asks which one you meant.

---

## `animate`

Generate a short video from a text description. Also available as `@video`.

**Usage:** `@animate <prompt>`

```
@animate A slow aerial shot over a pine forest at sunrise, mist in the valleys
@animate A neon sign flickering on a rainy street at night
```

`@animate` needs an authenticated account. The bot asks you to log in first if you aren't.

Rendering takes a minute or two — about 135 seconds for the default seven-second clip — and the box renders one clip at a time, so requests queue behind each other. The bot tells you where yours sits in the line and roughly how long the wait is, then posts the link as a reply to your original request when the clip is ready, so your client threads it back to what you asked for and there's no need to wait around. The estimate assumes every clip ahead of yours takes the full render time. Clips have sound by default.

**Making someone speak.** The video model writes the soundtrack from the prompt alone, so words only get spoken if you ask for them. Put them in quotes:

```
@animate a grizzled farmer on a porch shouts "get off my lawn!"
```

Those words go through to the video model verbatim. Keep a spoken line to about a dozen words — a longer one runs past the end of the clip and is cut off mid-word. A character you give no words to stays silent on purpose: speech the model has to invent comes out as babble rather than English.

While the clip renders the bot shows as typing continuously, not just for the first few seconds, so a working render still looks different from a failed one without another line appearing in the channel. Typing stops after six minutes even if the render is still going; the clip still arrives once it's ready. The delivered link also carries your nick and the prompt you gave it — `alice: your video is ready! "a corgi riding a unicorn" → https://…mp4` — so it's clear which request it answers even in clients that don't show reply threads.

Because each clip occupies the video hardware exclusively while it renders, `@animate` carries the tightest rate limit of any command, and the queue itself is capped: each person may have two clips waiting at once and the whole queue holds six, so a request past either limit is turned away rather than queued. Both limits are operator-configurable. Use [`@renders`](#renders) to see the queue or take your clip back out of it. If the bot is restarted mid-render it picks the job back up afterwards; a clip is only lost if the video server itself restarts.

`@animate` writes the video model's prompt from your request, the same way `@draw` does for pictures, and it writes a lot of it: this model follows detail, so a two-word ask renders generic while a shot description renders what you asked for. Your line becomes a paragraph covering the subject, what it does beat by beat, the camera, the setting and the light — the shot you asked for, spelled out, not a different one. The video model has never heard of your cast, so a name on its own would render as words on screen: the bot keeps the name you wrote and adds a description of what that subject looks like beside it. Where that description comes from depends on who you named: a character the channel invented is described from canon in a verse-enabled channel, and a real person, place or event is looked up before the clip is planned, so naming either one gets you the actual subject rather than a stand-in. Canon wins where the two overlap. Everything else you named — the people, the objects, the costumes — goes through as you wrote it, and the shot you asked for, the action, the camera move and the mood, is left alone.

**Animating a picture.** Put an image URL anywhere in the line and that picture becomes the clip's first frame:

```
@animate https://example.com/cat.png the cat turns its head and yawns
```

There's no flag to remember — a link ending in `.png`, `.jpg`, `.webp`, `.gif` or `.bmp` is taken as the reference, and the rest of the line is the prompt. It works in conversation too, so `vibebot animate <url> make it dance` does the same thing.

With a picture attached the bot still writes the video model's prompt, and it can see your picture while it does: it describes what's actually in frame and turns your words into the motion. So describe what should *happen* rather than what's in the shot — the picture already settled that. A URL on its own still works; you get gentle ambient movement. Clips built this way always carry sound, and quoted words are spoken here too.

The image has to be a public link the bot can fetch, under 10 MB. The bot downloads it, decodes it and re-encodes it before anything is sent to the video box, so files that are not really images are refused rather than forwarded; very large pictures are scaled down to 1920 px on the long edge. If the link can't be used the bot says so instead of quietly rendering without it.

You can also just ask for a video in conversation — the bot has a tool for it and will queue one the same way.

---

## `renders`

Shows the video render queue and lets you take a clip back out of it.

**Usage:** `@renders [cancel <id> | clear]`

```
@renders
@renders cancel 421
@renders clear
```

`@renders` on its own lists every clip waiting to render, oldest first, with its id, who asked for it, how long ago and its place in the line, all on one line; a queue too long for one line ends with `+N more`. When nothing is waiting it says so. `@renders cancel <id>` drops one of your own clips; admins can drop anyone's. `@renders clear` empties the queue and is admin-only. Cancelling asks the video server to stop the job as well; if the server declines, the clip still finishes on the box but is never posted.

`@renders` needs the same capability as `@animate`, so anyone who can queue a clip can see the queue.

---

## `story`

Generate an illustrated page from a short brief and post a link when it's ready. The bot picks one of two modes from your brief:

- **Story:** an illustrated tale.
- **Explainer:** a concept explained with labelled diagrams, as a learning aid.

**Usage:** `@story <brief>`

```
@story an illustrated tale of the lads winning the pub quiz
@story explain how photosynthesis works, with diagrams
```

The page renders in the background; the bot posts the link when it finishes. `@story` needs an authenticated account, uses the same permission as `@draw`, and allows one page per account every five minutes by default (`verseStorybookCooldownSeconds`).

---

## `forget`

Clear your conversation context for the current channel, or a named one:

```
@forget
@forget #otherchannel
```

Run in a channel, `@forget` also clears that channel's shared recent history, so stale bot answers stop feeding everyone's follow-ups. Naming a different channel clears only your own thread there — you cannot wipe another channel's shared history from outside it.

---

## `usage`

View API usage statistics for yourself, another user, or a channel.

**Usage:** `@usage [nick | #channel]`

```
@usage              → your stats plus the channel's, this month
@usage someone      → another user's stats, scoped to this channel (account-wide by PM)
@usage #somechannel → a channel's stats
```

The bot carries no usage tool and no usage figures in its prompt, so asking in conversation can get you a made-up number.

---

## `model`

Show which model answered the bot's last reply in a channel, and which chat model the channel is set to use.

**Usage:** `@model [<channel>]`

```
@model          → Last reply in #linux: anthropic/claude-haiku-5-5, 2m ago. Set to: anthropic/claude-haiku-5-5.
@model #tv      → another channel
```

The two can differ: a runtime `@config` change is lost when the bot restarts, and the reply after that runs on whatever `bot.conf` holds. The bot forgets the last reply on restart too, and says so.

---

## `channels`

List the busiest public channels on the network, with user counts and topics.

**Usage:** `@channels [--min <n>] [<channel> | <glob>]`

```
@channels                  → the ten busiest channels
@channels #linux           → that channel's count and topic
@channels --min 5 #linux*  → matching channels with at least 5 users
```

You can also just ask: "vibebot, is `#help` active?" or "vibebot, who is eck?" The bot looks it up (`LIST`, `NAMES`, or `WHOIS`) rather than guessing.

---

## `names`

List who is in a channel at this moment — including channels the bot is not in.

**Usage:** `@names [<channel>]`

```
@names          → the current channel
@names #linux   → another channel
```

Secret and private channels come back empty, exactly as they would for `/names`.

---

## Rate limits

Each command family carries its own rate limit. Your tier is unregistered or registered depending on whether you are authenticated, or trusted if your operator has granted you the `trusted` capability. Media commands have the tightest limits: `@draw` and `@story` for images, and `@animate` tighter still, since a clip holds the video hardware for over a minute. If you reach a limit, the bot tells you; wait and retry. Admins and owners are exempt.

---

## The verse

In channels with the verse enabled, you can join a persistent shared fiction as an avatar:

| Command | Description |
|---------|-------------|
| `@verseopt in` / `@verseopt out` | Opt your avatar in or out of the channel's verse |
| `@rp <text>` | Speak or act as your avatar for one turn |
| `@rp on` / `@rp off` | Stay in character without prefixing every line |
| `@avatar <persona>` | Set the persona that shapes your avatar's voice |
| `@verse` | Show where your avatar currently is |
| `@look [<target>]` | Show your scene, same as `@verse`, or describe any named person, place, faction, or item in the world |
| `@who` | List active avatars and their locations |

Once opted in, address the bot and mention someone or something in the world, or the channel's trigger word, and you get a tale: six or more paragraphs in your avatar's voice. The channel sees a one-line teaser and a link, because anything longer than a single IRC line is saved to the bot's web server rather than flooded into the channel. Where the operator has enabled `verseStorybookEnabled`, ask it to "illustrate" and you get an illustrated page instead; ask it to "draw" and you get a single image.

The bot stays in character for that turn only. `@rp` gives you a deliberate in-character turn without needing a mention, and `@rp on` holds character across a run of them. Without an avatar, a mention just keeps an ordinary answer true to the world.

Start a message with `//` or wrap it in `((...))` to speak out of character. Questions with nothing to do with the world get a straight answer, not a tall tale.
