package fake_test

import (
	"context"
	"testing"
	"time"

	"github.com/rdrake/vibebot/v9/llmcore"
	"github.com/rdrake/vibebot/v9/llmcore/fake"
)

func TestScriptedCompletions(t *testing.T) {
	c := fake.New()
	c.Script = []llmcore.Completion{
		{Text: "hello"},
		{ToolCalls: []llmcore.ToolCall{{ID: "1", Name: "save_memory"}}},
		{Text: "done"},
	}
	for i, want := range c.Script {
		got, err := c.Complete(context.Background(), llmcore.CachedPrefix{}, nil, nil, "m", llmcore.CacheHandle{})
		if err != nil {
			t.Fatalf("iter %d: %v", i, err)
		}
		if got.Text != want.Text {
			t.Fatalf("iter %d text: %q", i, got.Text)
		}
	}
}

func TestEnsureCacheReturnsHandle(t *testing.T) {
	c := fake.New()
	c.CacheHandleOut = llmcore.CacheHandle{Name: "fake-1", Provider: "gemini", ExpiresAt: time.Now().Add(time.Hour)}
	h, err := c.EnsureCache(context.Background(), llmcore.Scope{Channel: "#x"}, llmcore.CachedPrefix{})
	if err != nil {
		t.Fatal(err)
	}
	if h.Name != "fake-1" || !h.IsCached() {
		t.Fatalf("handle: %+v", h)
	}
}
