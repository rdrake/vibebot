package exec

import (
	"context"
	"log/slog"
	"time"

	"github.com/rdrake/vibebot/v9/ircout"
	"github.com/rdrake/vibebot/v9/llmcore"
	"github.com/rdrake/vibebot/v9/overlay"
	"github.com/rdrake/vibebot/v9/persist"
	"github.com/rdrake/vibebot/v9/router"
	"github.com/rdrake/vibebot/v9/router/profile"
	"github.com/rdrake/vibebot/v9/tooling"
)

// Config bundles the executor's dependencies.
type Config struct {
	LLM      llmcore.Client
	Tools    tooling.Dispatcher
	Overlay  overlay.Resolver
	Store    persist.Store
	IRC      ircout.Sender
	Log      *slog.Logger
	Registry *profile.Registry
}

// Executor runs RouteDecisions.
type Executor struct {
	cfg Config
}

// New constructs an Executor.
func New(cfg Config) *Executor { return &Executor{cfg: cfg} }

// Run executes one routing decision end-to-end.
func (e *Executor) Run(ctx context.Context, d router.RouteDecision) error {
	if d.Action == router.Ignore {
		return nil
	}

	target := deliveryTarget(d)
	if d.Delivery.Typing {
		e.cfg.IRC.SendTyping(ctx, d.Event.Network, target, ircout.TypingActive)
		defer e.cfg.IRC.SendTyping(context.Background(), d.Event.Network, target, ircout.TypingDone)
	}

	return e.runInner(ctx, d)
}

func (e *Executor) runInner(ctx context.Context, d router.RouteDecision) error {
	prof, _ := e.cfg.Registry.Get(d.Profile)

	ovText, err := e.cfg.Overlay.Get(ctx, overlay.Scope{
		Network: d.CacheScope.Network, Channel: d.CacheScope.Channel, Profile: d.CacheScope.Profile,
	})
	if err != nil {
		e.recordFailure(ctx, d, "overlay_get", err)
		return err
	}

	tools, err := ResolveSchemas(e.cfg.Tools, d.Tools)
	if err != nil {
		e.recordFailure(ctx, d, "resolve_schemas", err)
		return err
	}

	prefix := BuildCachedPrefix(BuildPrefixArgs{
		Framework: prof.FrameworkPrompt,
		Overlay:   ovText,
		Tools:     tools,
		Network:   d.CacheScope.Network,
		Channel:   d.CacheScope.Channel,
		Profile:   d.CacheScope.Profile,
	})

	cache, err := e.cfg.LLM.EnsureCache(ctx, llmcore.Scope{
		Network: d.CacheScope.Network, Channel: d.CacheScope.Channel,
		Profile: d.CacheScope.Profile, OverlayHash: d.CacheScope.OverlayHash,
	}, prefix)
	if err != nil {
		e.recordFailure(ctx, d, "ensure_cache", err)
		return err
	}

	return e.runWithCache(ctx, d, prefix, tools, cache)
}

func (e *Executor) runWithCache(ctx context.Context, d router.RouteDecision, prefix llmcore.CachedPrefix, tools []llmcore.ToolSchema, cache llmcore.CacheHandle) error {
	history, err := e.cfg.Store.History(ctx, d.Event.Network, d.Event.Channel, d.Prompt.HistoryLimit)
	if err != nil {
		e.recordFailure(ctx, d, "history", err)
		return err
	}
	memories, err := e.cfg.Store.Memories(ctx, d.Event.Network, d.Event.Channel, d.Prompt.UserNick)
	if err != nil {
		e.recordFailure(ctx, d, "memories", err)
		return err
	}
	messages := buildTail(history, memories, d.Prompt.UserText, d.Prompt.UserNick)

	var lastCached int
	wrapped := &cacheCounter{Client: e.cfg.LLM, last: &lastCached}

	text, err := runLoop(ctx, runLoopArgs{
		LLM: wrapped, Tools: e.cfg.Tools, MaxIters: d.Delivery.MaxIters,
		Prefix: prefix, Messages: messages, ToolsList: tools, Model: d.Model, Cache: cache,
	})
	if err != nil {
		e.recordFailure(ctx, d, "loop", err)
		return err
	}

	if err := e.deliver(ctx, d, text); err != nil {
		e.recordFailure(ctx, d, "deliver", err)
		return err
	}

	if err := e.cfg.Store.AppendTurn(ctx, d.Event.Network, d.Event.Channel, d.Event.Nick, text); err != nil {
		e.cfg.Log.Warn("append turn failed", "err", err)
	}
	_ = e.cfg.Store.RecordUsage(ctx, persist.UsageRow{
		Timestamp:    timeNow(),
		Network:      d.Event.Network,
		Channel:      d.Event.Channel,
		Nick:         d.Event.Nick,
		Profile:      d.Profile,
		Model:        d.Model,
		CachedTokens: lastCached,
		Status:       "success",
	})
	return nil
}

func buildTail(history []persist.HistoryEntry, memories []persist.Memory, userText, userNick string) []llmcore.Message {
	msgs := make([]llmcore.Message, 0, len(history)+len(memories)+1)
	for _, h := range history {
		msgs = append(msgs, llmcore.Message{Role: h.Role, Content: h.Content})
	}
	if len(memories) > 0 {
		var content string
		for _, m := range memories {
			content += "- " + m.Fact + "\n"
		}
		msgs = append(msgs, llmcore.Message{Role: "system", Content: "Memories about " + userNick + ":\n" + content})
	}
	msgs = append(msgs, llmcore.Message{Role: "user", Content: userText})
	return msgs
}

func (e *Executor) deliver(ctx context.Context, d router.RouteDecision, text string) error {
	chunks := chunkAt(text, d.Delivery.ChunkSize)
	for i, c := range chunks {
		opts := ircout.SendOpts{
			Network: d.Event.Network,
			Target:  deliveryTarget(d),
			Text:    c,
		}
		if i == 0 {
			if e.cfg.IRC.HasReplyCAP(d.Event.Network) && d.Delivery.ReplyToID != "" {
				opts.ReplyTo = d.Delivery.ReplyToID
			} else if d.Delivery.NickPrefixOnFallback && d.Event.Channel != "" {
				opts.NickPrefix = d.Event.Nick + ": "
			}
		}
		if err := e.cfg.IRC.Send(ctx, opts); err != nil {
			return err
		}
	}
	return nil
}

func (e *Executor) recordFailure(ctx context.Context, d router.RouteDecision, stage string, cause error) {
	_ = e.cfg.Store.RecordUsage(ctx, persist.UsageRow{
		Timestamp:   timeNow(),
		Network:     d.Event.Network,
		Channel:     d.Event.Channel,
		Nick:        d.Event.Nick,
		Profile:     d.Profile,
		Model:       d.Model,
		Status:      "fatal_fail",
		ErrorDetail: stage + ": " + cause.Error(),
	})
}

func deliveryTarget(d router.RouteDecision) string {
	if d.Event.Channel != "" {
		return d.Event.Channel
	}
	return d.Event.Nick
}

// cacheCounter wraps a Client to record the most recent CachedTokens.
type cacheCounter struct {
	llmcore.Client
	last *int
}

func (c *cacheCounter) Complete(ctx context.Context, prefix llmcore.CachedPrefix, messages []llmcore.Message, tools []llmcore.ToolSchema, model string, cache llmcore.CacheHandle) (llmcore.Completion, error) {
	out, err := c.Client.Complete(ctx, prefix, messages, tools, model, cache)
	if err == nil {
		*c.last = out.CachedTokens
	}
	return out, err
}

// timeNow is a package var so tests can override if needed.
var timeNow = time.Now
