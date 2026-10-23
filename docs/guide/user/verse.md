# The verse

Some channels host a verse: a shared story world that lasts between sessions. You play an avatar, the bot narrates, and the people, places and events that come up in play are written down as the channel's canon, so next week's scene remembers this week's.

The verse is off unless an operator has turned it on for the channel. To check, type `@who` in the channel: it answers "This channel doesn't have a verse" where there is none.

## A short session

```irc
<you>     @verseopt in
<VibeBot> You step into The Clearing. A quiet woodland clearing where new
          stories begin.
<you>     @avatar A retired lighthouse keeper who talks to gulls.
<VibeBot> Persona set.
<you>     @rp sits on a stump and asks the nearest gull for news
<VibeBot> The gull tilts its head, considers you, and drops a soggy
          scrap of map at your boots before wheeling off toward the
          ridge.
<you>     @rp on
<VibeBot> Roleplay mode ON — just talk and I'll answer in character.
          `@rp off` to stop; a leading `//` slips one message OOC.
<you>     VibeBot, I unfold the map and squint at it
<VibeBot> Salt has eaten most of it, but one mark survives: a tower,
          drawn in a hand you know, two days' walk north.
<you>     VibeBot, // what time is it in Tokyo?
<VibeBot> 10:05 Saturday morning in Tokyo.
<you>     @rp off
<VibeBot> Roleplay mode off — back to normal chat.
```

Act with `@rp`, not `@ask`: `@ask` always takes the ordinary chat path and never plays your avatar.

## Joining and leaving

<div class="vb-steps" markdown>

1. **Join.** `@verseopt in`, in the channel. You get an avatar and a starting place.
2. **Give it a voice.** `@avatar <persona>` describes who your avatar is. It is optional, and it needs a logged-in network account. The same persona voices your avatar in every verse channel and colours your [`@story`](ai-commands.md#story) pages. See [Avatar persona](memory.md#avatar-persona).
3. **Look around.** `@verse` says where your avatar is, `@look <name>` describes any person, place, faction or item in the world, and `@who` lists everyone playing and where they are.

</div>

`@verseopt out` retires your avatar. The story keeps what it did; opting back in later gives you a fresh avatar.

If the bot says you lack a capability, the operator has limited who can play. Ask them for `llm.verse`.

## Playing

| You type | You get |
|----------|---------|
| `@rp <text>` | One turn as your avatar. The bot answers in character |
| `@rp on` | Every message you address to the bot is an in-character turn, until `@rp off` |
| Address the bot and name someone or something in the world | A tale: six or more paragraphs in your avatar's voice. The channel sees a one-line teaser and a link to the full text |
| The same, asking it to *illustrate* (or "comic", "storybook", "with pictures") | An illustrated page, where the operator has turned storybooks on |
| The same, asking it to *draw*, *sketch* or *paint* | A single picture, drawn from what the world says about its subject |
| A question that has nothing to do with the world | A straight answer, not a tall tale |

`@rp on` still needs you to address the bot ("VibeBot, …"); it saves you typing `@rp` each time. It lasts until `@rp off`, until you go 15 minutes without an in-character turn, or until the bot restarts. It holds in one channel only.

Naming something the world knows, or the channel's trigger word, is what turns a message into a tale. The trigger word is "verse" unless the operator picked another. A tale needs an avatar: without one, naming a character just gets you an ordinary answer that keeps to the world's facts.

An illustrated page needs a logged-in account and the right to draw, and it shares `@story`'s limit of one page every five minutes per account. With storybooks off, an illustrate request gets an ordinary answer instead.

`@rp` counts against your `@ask` rate limit.

### Out of character

Start a message with `//`, or wrap the whole thing in `((...))`, and the bot answers it as ordinary chat. That works mid-`@rp on`, and it never starts a tale or an illustrated page:

```irc
<you>     VibeBot, // is the bot lagging for anyone else?
<you>     VibeBot, ((ignore that last bit, I fat-fingered it))
```

## What becomes canon

During in-character turns and tales the bot writes down what happened: events, who was there, and any new character someone names. Narrate something about someone else ("VibeBot, stinky dan threw a guff grenade at Andrew") and both of them end up in the record, added as new characters if the world didn't know them. Ordinary chat doesn't add to canon unless the operator has turned that on.

The record isn't permanent for everything:

- **Invented characters and places fade** after 14 days with nobody mentioning them. They are retired, not deleted: name one in play again and it comes back with its history.
- **Pinned and locked entries never fade.** A canon editor pins or locks the ones that matter. Pinned characters are in every scene.
- **Old events get condensed.** Once a world has 20 or more events, the bot folds those older than 30 days into a short summary, a batch at a time.

Canon editors can also fix mistakes by hand. If the story recorded something wrong, ask one. The commands they use are on the operator's [verse page](../operator/verse.md#editing-canon).

## Commands

| Command | Does |
|---------|------|
| `@verseopt in` / `@verseopt out` | Join with a new avatar, or retire yours |
| `@avatar <persona>` / `@avatar clear` | Set or clear your avatar's persona; bare `@avatar` shows it |
| `@rp <text>` | One in-character turn |
| `@rp on` / `@rp off` | Stay in character without typing `@rp` |
| `@verse` | Where your avatar is |
| `@look [<name>]` | Your surroundings, or any named thing in the world |
| `@who` | Everyone playing, and where they are |

Running a verse, curating canon and the GM tools are on the operator's [verse page](../operator/verse.md).
