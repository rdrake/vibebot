package exec_test

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/rdrake/vibebot/v9/exec"
	"github.com/rdrake/vibebot/v9/ircout"
	ircfake "github.com/rdrake/vibebot/v9/ircout/fake"
	"github.com/rdrake/vibebot/v9/llmcore"
	llmfake "github.com/rdrake/vibebot/v9/llmcore/fake"
	overlayfake "github.com/rdrake/vibebot/v9/overlay/fake"
	persistfake "github.com/rdrake/vibebot/v9/persist/fake"
	"github.com/rdrake/vibebot/v9/router"
	"github.com/rdrake/vibebot/v9/router/profile"
	toolfake "github.com/rdrake/vibebot/v9/tooling/fake"
)

type harness struct {
	e      *exec.Executor
	llm    *llmfake.Client
	sender *ircfake.Sender
	tools  *toolfake.Dispatcher
	store  *persistfake.Store
	res    *overlayfake.Resolver
}

func newHarness(t *testing.T) *harness {
	t.Helper()
	td := toolfake.New()
	for _, n := range []string{
		"clear_instruction", "delete_memory", "fetch_url", "generate_image",
		"get_instruction", "list_memories", "save_memory", "search_web",
		"set_instruction", "update_memory",
	} {
		td.Schemas[n] = []byte(`{"name":"` + n + `"}`)
	}
	llm := llmfake.New()
	llm.CacheHandleOut = llmcore.CacheHandle{Name: "cache-1", Provider: "gemini", ExpiresAt: time.Now().Add(time.Hour)}
	sender := ircfake.New()
	st := persistfake.New()
	res := overlayfake.New()
	reg := profile.NewRegistry()
	profile.RegisterBuiltins(reg)

	e := exec.New(exec.Config{
		LLM: llm, Tools: td, Overlay: res, Store: st, IRC: sender,
		Log:      slog.New(slog.NewTextHandler(os.Stderr, nil)),
		Registry: reg,
	})
	return &harness{e, llm, sender, td, st, res}
}

func sampleDecision() router.RouteDecision {
	return router.RouteDecision{
		Action:     router.RespondChat,
		Profile:    "chat",
		Tools:      []string{"clear_instruction", "delete_memory", "fetch_url", "generate_image", "get_instruction", "list_memories", "save_memory", "search_web", "set_instruction", "update_memory"},
		CacheScope: router.CacheScope{Network: "afternet", Channel: "#x", Profile: "chat", OverlayHash: "abc"},
		Model:      "gemini-2.5-flash",
		Prompt:     router.PromptSpec{HistoryLimit: 5, UserText: "hi", UserNick: "alice"},
		Delivery:   router.DeliverySpec{Typing: true, ChunkSize: 380, MaxIters: 2},
		Event:      router.IRCEvent{Network: "afternet", Channel: "#x", Nick: "alice", Text: "hi", ReceivedAt: time.Now()},
	}
}

func TestRunSendsResponse(t *testing.T) {
	h := newHarness(t)
	h.llm.Script = []llmcore.Completion{{Text: "hello alice"}}
	if err := h.e.Run(context.Background(), sampleDecision()); err != nil {
		t.Fatal(err)
	}
	if len(h.sender.Sent) == 0 || h.sender.Sent[0].Text != "hello alice" {
		t.Fatalf("sent: %+v", h.sender.Sent)
	}
}

func TestRunSendsTypingActiveThenDone(t *testing.T) {
	h := newHarness(t)
	h.llm.Script = []llmcore.Completion{{Text: "hi"}}
	if err := h.e.Run(context.Background(), sampleDecision()); err != nil {
		t.Fatal(err)
	}
	if len(h.sender.TypingStates) < 2 {
		t.Fatalf("typing: %v", h.sender.TypingStates)
	}
	if h.sender.TypingStates[0].State != ircout.TypingActive {
		t.Errorf("first: %v", h.sender.TypingStates[0].State)
	}
	if h.sender.TypingStates[len(h.sender.TypingStates)-1].State != ircout.TypingDone {
		t.Errorf("last: %v", h.sender.TypingStates[len(h.sender.TypingStates)-1].State)
	}
}

func TestRunTypingDoneFiresOnError(t *testing.T) {
	h := newHarness(t)
	d := sampleDecision()
	d.Tools = []string{"nonexistent_tool"}
	if err := h.e.Run(context.Background(), d); err == nil {
		t.Fatal("expected error")
	}
	if len(h.sender.TypingStates) == 0 || h.sender.TypingStates[len(h.sender.TypingStates)-1].State != ircout.TypingDone {
		t.Fatalf("typing done missing: %v", h.sender.TypingStates)
	}
}

func TestRunTypingDoneFiresOnDispatcherPanic(t *testing.T) {
	h := newHarness(t)
	h.llm.Script = []llmcore.Completion{
		{ToolCalls: []llmcore.ToolCall{{ID: "1", Name: "save_memory", Arguments: "{}"}}},
	}
	h.tools.PanicOnDispatch = true
	h.tools.PanicMessage = "boom"

	defer func() {
		if got := recover(); got != "boom" {
			t.Fatalf("expected panic %q, got %v", "boom", got)
		}
		if len(h.sender.TypingStates) == 0 || h.sender.TypingStates[len(h.sender.TypingStates)-1].State != ircout.TypingDone {
			t.Fatalf("typing done missing after panic: %v", h.sender.TypingStates)
		}
	}()
	_ = h.e.Run(context.Background(), sampleDecision())
	t.Fatal("expected panic to propagate")
}

func TestRunRecordsCachedTokens(t *testing.T) {
	h := newHarness(t)
	h.llm.Script = []llmcore.Completion{{Text: "hi", CachedTokens: 100}}
	if err := h.e.Run(context.Background(), sampleDecision()); err != nil {
		t.Fatal(err)
	}
	if len(h.store.Usage) != 1 {
		t.Fatalf("usage: %d", len(h.store.Usage))
	}
	if h.store.Usage[0].CachedTokens != 100 {
		t.Errorf("cached: %d", h.store.Usage[0].CachedTokens)
	}
}

func TestRunUsesReplyToWhenCAPAvailable(t *testing.T) {
	h := newHarness(t)
	h.sender.ReplyCAP = map[string]bool{"afternet": true}
	h.llm.Script = []llmcore.Completion{{Text: "hi"}}
	d := sampleDecision()
	d.Delivery.ReplyToID = "m-1"
	d.Delivery.NickPrefixOnFallback = true
	if err := h.e.Run(context.Background(), d); err != nil {
		t.Fatal(err)
	}
	first := h.sender.Sent[0]
	if first.ReplyTo != "m-1" {
		t.Errorf("ReplyTo: %q", first.ReplyTo)
	}
	if first.NickPrefix != "" {
		t.Errorf("NickPrefix should be empty when ReplyTo set: %q", first.NickPrefix)
	}
}

func TestRunUsesNickPrefixWhenNoReplyCAP(t *testing.T) {
	h := newHarness(t)
	// ReplyCAP empty by default
	h.llm.Script = []llmcore.Completion{{Text: "hi"}}
	d := sampleDecision()
	d.Delivery.ReplyToID = "m-1"
	d.Delivery.NickPrefixOnFallback = true
	if err := h.e.Run(context.Background(), d); err != nil {
		t.Fatal(err)
	}
	first := h.sender.Sent[0]
	if first.ReplyTo != "" {
		t.Errorf("ReplyTo should be empty without CAP: %q", first.ReplyTo)
	}
	if first.NickPrefix != "alice: " {
		t.Errorf("NickPrefix: %q", first.NickPrefix)
	}
}

func TestRunChunksLongResponse(t *testing.T) {
	h := newHarness(t)
	long := strings.Repeat("a", 1000)
	h.llm.Script = []llmcore.Completion{{Text: long}}
	d := sampleDecision()
	d.Delivery.ChunkSize = 200
	if err := h.e.Run(context.Background(), d); err != nil {
		t.Fatal(err)
	}
	if len(h.sender.Sent) != 5 {
		t.Fatalf("expected 5 chunks, got %d", len(h.sender.Sent))
	}
}

func TestRunPropagatesLLMError(t *testing.T) {
	h := newHarness(t)
	h.llm.CompleteErr = errors.New("provider down")
	if err := h.e.Run(context.Background(), sampleDecision()); err == nil {
		t.Fatal("expected error")
	}
	// Failure should still record usage with status != success
	if len(h.store.Usage) != 1 {
		t.Fatalf("usage rows: %d", len(h.store.Usage))
	}
	if h.store.Usage[0].Status == "success" {
		t.Errorf("status: %q", h.store.Usage[0].Status)
	}
}
