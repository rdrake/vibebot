package exec

import (
	"context"
	"errors"
	"testing"

	"github.com/rdrake/vibebot/v9/llmcore"
	llmfake "github.com/rdrake/vibebot/v9/llmcore/fake"
	"github.com/rdrake/vibebot/v9/tooling"
	toolfake "github.com/rdrake/vibebot/v9/tooling/fake"
)

func TestRunLoopNoToolCallsReturns(t *testing.T) {
	llm := llmfake.New()
	llm.Script = []llmcore.Completion{{Text: "hello"}}
	td := toolfake.New()
	text, err := runLoop(context.Background(), runLoopArgs{
		LLM: llm, Tools: td, MaxIters: 3,
		Messages: []llmcore.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("%q", text)
	}
	if len(llm.CompleteCalls) != 1 {
		t.Fatalf("calls: %d", len(llm.CompleteCalls))
	}
}

func TestRunLoopOneToolThenReturn(t *testing.T) {
	llm := llmfake.New()
	llm.Script = []llmcore.Completion{
		{ToolCalls: []llmcore.ToolCall{{ID: "1", Name: "save_memory", Arguments: "{}"}}},
		{Text: "saved!"},
	}
	td := toolfake.New()
	td.Results = map[string]tooling.ToolResult{"save_memory": {Content: `{"ok":true}`}}

	text, err := runLoop(context.Background(), runLoopArgs{
		LLM: llm, Tools: td, MaxIters: 3,
		Messages: []llmcore.Message{{Role: "user", Content: "remember"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "saved!" {
		t.Fatalf("%q", text)
	}
	if len(llm.CompleteCalls) != 2 {
		t.Fatalf("calls: %d", len(llm.CompleteCalls))
	}
	last := llm.CompleteCalls[1].Messages
	if last[len(last)-1].Role != "tool" {
		t.Fatalf("last msg: %+v", last)
	}
}

func TestRunLoopHitsMaxIters(t *testing.T) {
	llm := llmfake.New()
	llm.Script = []llmcore.Completion{
		{ToolCalls: []llmcore.ToolCall{{ID: "1", Name: "a"}}},
		{ToolCalls: []llmcore.ToolCall{{ID: "2", Name: "a"}}},
	}
	td := toolfake.New()
	td.Results = map[string]tooling.ToolResult{"a": {Content: "{}"}}

	text, err := runLoop(context.Background(), runLoopArgs{
		LLM: llm, Tools: td, MaxIters: 2, FallbackText: "looping",
		Messages: []llmcore.Message{{Role: "user", Content: "go"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "looping" {
		t.Fatalf("%q", text)
	}
}

func TestRunLoopPropagatesLLMError(t *testing.T) {
	llm := llmfake.New()
	llm.CompleteErr = errors.New("boom")
	td := toolfake.New()
	if _, err := runLoop(context.Background(), runLoopArgs{
		LLM: llm, Tools: td, MaxIters: 2,
		Messages: []llmcore.Message{{Role: "user", Content: "hi"}},
	}); err == nil {
		t.Fatal("expected error")
	}
}
