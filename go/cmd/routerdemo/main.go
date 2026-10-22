// Command routerdemo wires the routing/intent layer with in-memory fakes
// and runs one chat turn end-to-end.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/rdrake/vibebot/v9/exec"
	ircfake "github.com/rdrake/vibebot/v9/ircout/fake"
	"github.com/rdrake/vibebot/v9/llmcore"
	llmfake "github.com/rdrake/vibebot/v9/llmcore/fake"
	"github.com/rdrake/vibebot/v9/overlay"
	overlayfake "github.com/rdrake/vibebot/v9/overlay/fake"
	persistfake "github.com/rdrake/vibebot/v9/persist/fake"
	"github.com/rdrake/vibebot/v9/router"
	"github.com/rdrake/vibebot/v9/router/profile"
	toolfake "github.com/rdrake/vibebot/v9/tooling/fake"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	reg := profile.NewRegistry()
	profile.RegisterBuiltins(reg)

	llm := llmfake.New()
	llm.CacheHandleOut = llmcore.CacheHandle{Name: "fake-cache-1", Provider: "gemini", ExpiresAt: time.Now().Add(time.Hour)}
	llm.Script = []llmcore.Completion{{Text: "hi alice, how can i help?"}}

	td := toolfake.New()
	for _, n := range []string{
		"clear_instruction", "delete_memory", "fetch_url", "generate_image",
		"get_instruction", "list_memories", "save_memory", "search_web",
		"set_instruction", "update_memory",
	} {
		td.Schemas[n] = []byte(`{"name":"` + n + `"}`)
	}

	res := overlayfake.New()
	overlayText := "be friendly"
	res.Texts[overlay.Scope{Network: "afternet", Channel: "#demo", Profile: "chat"}] = overlayText

	st := persistfake.New()
	sender := ircfake.New()

	e := exec.New(exec.Config{
		LLM: llm, Tools: td, Overlay: res, Store: st, IRC: sender, Log: log, Registry: reg,
	})

	evt := router.IRCEvent{
		Network: "afternet", Channel: "#demo", Nick: "alice",
		Text: "vibebot: hi", MessageID: "m-1", ReceivedAt: time.Now(),
	}
	state := router.ChannelState{Profile: "chat", Overlay: overlayText}
	bot := router.BotState{SelfNick: "vibebot", Now: time.Now()}

	d := router.Route(evt, state, bot, reg)
	fmt.Printf("Decision: action=%s profile=%s tools=%d model=%s overlay_hash=%s\n",
		d.Action, d.Profile, len(d.Tools), d.Model, d.CacheScope.OverlayHash)

	if err := e.Run(context.Background(), d); err != nil {
		log.Error("run failed", "err", err)
		os.Exit(1)
	}

	fmt.Printf("Sent %d msg(s); first: %q\n", len(sender.Sent), sender.Sent[0].Text)
	fmt.Printf("Typing events: %d\n", len(sender.TypingStates))
	fmt.Printf("Usage rows: %d (cached_tokens=%d, status=%s)\n",
		len(st.Usage), st.Usage[0].CachedTokens, st.Usage[0].Status)
}
