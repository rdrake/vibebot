package exec

import (
	"context"
	"encoding/json"

	"github.com/rdrake/vibebot/v9/llmcore"
	"github.com/rdrake/vibebot/v9/tooling"
)

type runLoopArgs struct {
	LLM          llmcore.Client
	Tools        tooling.Dispatcher
	MaxIters     int
	Prefix       llmcore.CachedPrefix
	Messages     []llmcore.Message
	ToolsList    []llmcore.ToolSchema
	Model        string
	Cache        llmcore.CacheHandle
	FallbackText string
}

func runLoop(ctx context.Context, args runLoopArgs) (string, error) {
	messages := append([]llmcore.Message(nil), args.Messages...)
	for i := 1; i <= args.MaxIters; i++ {
		completion, err := args.LLM.Complete(ctx, args.Prefix, messages, args.ToolsList, args.Model, args.Cache)
		if err != nil {
			return "", err
		}
		if len(completion.ToolCalls) == 0 {
			return completion.Text, nil
		}
		for _, tc := range completion.ToolCalls {
			result := args.Tools.Dispatch(ctx, tooling.ToolCall{
				ID: tc.ID, Name: tc.Name, Arguments: tc.Arguments,
			})
			messages = append(messages, llmcore.Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Content:    toolResultPayload(result),
			})
		}
	}
	fallback := args.FallbackText
	if fallback == "" {
		fallback = "I appear to be thinking in circles. Try rephrasing?"
	}
	return fallback, nil
}

func toolResultPayload(r tooling.ToolResult) string {
	type wire struct {
		Content string `json:"content"`
		Error   string `json:"error,omitempty"`
		Denied  bool   `json:"denied,omitempty"`
	}
	w := wire{Content: r.Content}
	if r.Err != nil {
		w.Error = r.Err.Error()
	}
	w.Denied = r.Denied
	b, _ := json.Marshal(w)
	return string(b)
}
