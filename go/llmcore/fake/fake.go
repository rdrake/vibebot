// Package fake provides an in-memory llmcore.Client for tests.
package fake

import (
	"context"
	"errors"
	"sync"

	"github.com/rdrake/vibebot/v9/llmcore"
)

type Client struct {
	Script         []llmcore.Completion
	CacheHandleOut llmcore.CacheHandle
	EnsureCacheErr error
	CompleteErr    error

	mu sync.Mutex
	i  int

	EnsureCacheCalls []EnsureCacheCall
	CompleteCalls    []CompleteCall
}

type EnsureCacheCall struct {
	Scope  llmcore.Scope
	Prefix llmcore.CachedPrefix
}

type CompleteCall struct {
	Prefix   llmcore.CachedPrefix
	Messages []llmcore.Message
	Tools    []llmcore.ToolSchema
	Model    string
	Cache    llmcore.CacheHandle
}

func New() *Client { return &Client{} }

func (c *Client) EnsureCache(ctx context.Context, scope llmcore.Scope, prefix llmcore.CachedPrefix) (llmcore.CacheHandle, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.EnsureCacheCalls = append(c.EnsureCacheCalls, EnsureCacheCall{scope, prefix})
	if c.EnsureCacheErr != nil {
		return llmcore.CacheHandle{}, c.EnsureCacheErr
	}
	return c.CacheHandleOut, nil
}

func (c *Client) Complete(ctx context.Context, prefix llmcore.CachedPrefix, messages []llmcore.Message, tools []llmcore.ToolSchema, model string, cache llmcore.CacheHandle) (llmcore.Completion, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.CompleteCalls = append(c.CompleteCalls, CompleteCall{prefix, messages, tools, model, cache})
	if c.CompleteErr != nil {
		return llmcore.Completion{}, c.CompleteErr
	}
	if c.i >= len(c.Script) {
		return llmcore.Completion{}, errors.New("fake: script exhausted")
	}
	out := c.Script[c.i]
	c.i++
	return out, nil
}
