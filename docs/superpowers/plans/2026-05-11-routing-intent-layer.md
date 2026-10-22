# Routing/Intent Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sub-project E of the vibebot Go rewrite: the routing/intent layer (pure `router.Route` + stateful `exec.Executor`), with interface contracts and test fakes for sub-projects A/B/C/D/F so this slice is independently testable.

**Architecture:** Pure decision function emits a `RouteDecision` struct → stateful executor consumes it, resolves tools to schemas, builds a canonical cache prefix, drives an LLM tool-call loop, delivers via IRC. Cache discipline is enforced by `llmcore.Canonical(CachedPrefix) []byte` — a single byte serializer with documented ordering and delimiter. Test fakes for every external interface keep the slice runnable in isolation.

**Tech Stack:** Go 1.23 (pinned floor; `log/slog`, `crypto/sha256`, `encoding/json`, `sync`). No external runtime dependencies in this slice (real `openai-go`, `ergochat/irc-go`, Gemini SDK enter in sub-projects A/B). Module path: `github.com/rdrake/vibebot/v9`. Code lives in `go/` subdirectory of the existing repo.

**Spec:** `docs/superpowers/specs/2026-05-11-routing-intent-layer-design.md` (commit `2b07346`).

**Revision history:**
- v1: initial draft
- v2: revised after codex + code-reviewer pass — fixes cache-prefix-bytes serialization, overlay scope circular dep, panic-test fidelity, `CacheHandle`/`ToolSchema` types, task granularity, missing test coverage.

---

## File structure

All paths relative to repo root `/Users/rdrake/workspace/afternet/vibebot-v8/`.

```
go/
├── go.mod                          # module github.com/rdrake/vibebot/v9; go 1.23
├── Makefile                        # build/test/lint targets
├── llmcore/
│   ├── client.go                   # Client iface, CachedPrefix, ToolSchema, CacheHandle, Canonical()
│   ├── canonical_test.go
│   └── fake/fake.go                # in-memory fake LLM client
├── tooling/
│   ├── dispatcher.go               # Dispatcher iface, ToolCall, ToolResult
│   └── fake/fake.go                # in-memory fake (with PanicOnDispatch)
├── overlay/
│   ├── resolver.go                 # Resolver iface (Scope has no OverlayHash)
│   └── fake/fake.go
├── persist/
│   ├── store.go                    # Store iface
│   └── fake/fake.go
├── ircout/
│   ├── sender.go                   # Sender iface
│   └── fake/fake.go
├── router/
│   ├── doc.go
│   ├── types.go                    # IRCEvent, ChannelState, BotState, RouteDecision, ...
│   ├── addressed.go                # addressed() helper
│   ├── addressed_test.go
│   ├── scope.go                    # OverlayHash, CacheScope construction
│   ├── scope_test.go
│   ├── route.go                    # Route() pure function
│   ├── route_test.go
│   └── profile/
│       ├── profile.go              # Profile struct, Registry (defensive-copy Get)
│       ├── builtin.go              # quiet, chat, scene, loom, admin
│       ├── profile_test.go
│       └── builtin_test.go
├── exec/
│   ├── doc.go
│   ├── errors.go                   # typed error categories
│   ├── errors_test.go
│   ├── resolve.go                  # ResolveSchemas helper
│   ├── resolve_test.go
│   ├── prefix.go                   # BuildCachedPrefix
│   ├── prefix_test.go
│   ├── chunk.go                    # chunkAt helper
│   ├── chunk_test.go
│   ├── loop.go                     # tool-call loop
│   ├── loop_test.go
│   ├── executor.go                 # Executor struct, Run()
│   └── executor_test.go
└── cmd/
    └── routerdemo/
        └── main.go                 # wires fakes, sends a sample event
```

Total: 26 tasks across 7 phases. Each task is one logical unit; each step is 2–5 minutes of work.

---

## Phase 0 — Project skeleton

### Task 1: Initialize Go module and Makefile

**Files:**
- Create: `go/go.mod`
- Create: `go/Makefile`
- Create: `go/.gitignore`

- [ ] **Step 1: Verify Go is installed (1.23+)**

Run: `go version`
Expected: `go version go1.23.x ...` or newer. If older or missing, install via `brew install go` (macOS) or follow https://go.dev/doc/install.

- [ ] **Step 2: Verify golangci-lint is installed**

Run: `golangci-lint --version`
Expected: a version string. If missing, install via `brew install golangci-lint` before continuing — the Makefile treats it as a hard dependency.

- [ ] **Step 3: Create the module**

```bash
mkdir -p /Users/rdrake/workspace/afternet/vibebot-v8/go
cd /Users/rdrake/workspace/afternet/vibebot-v8/go
go mod init github.com/rdrake/vibebot/v9
```

Then edit `go/go.mod` so the `go` directive pins to `1.23` (not "or newer"):

```
module github.com/rdrake/vibebot/v9

go 1.23
```

- [ ] **Step 4: Write `go/Makefile`**

```make
.PHONY: build test lint vet fmt all check-go check-lint

all: check-go check-lint fmt vet lint test build

check-go:
	@v=$$(go version | awk '{print $$3}' | sed 's/^go//'); \
	  case "$$v" in \
	    1.23*|1.24*|1.25*|1.26*|1.27*|1.28*|1.29*) ;; \
	    *) echo "go 1.23+ required, found $$v"; exit 1 ;; \
	  esac

check-lint:
	@command -v golangci-lint >/dev/null 2>&1 || { \
	  echo "golangci-lint required; install: brew install golangci-lint"; exit 1; }

build:
	go build ./...

test:
	go test -race -count=1 ./...

vet:
	go vet ./...

fmt:
	gofmt -l -w .

lint:
	golangci-lint run ./...
```

- [ ] **Step 5: Write `go/.gitignore`**

```
# Go build artifacts
*.test
*.out
/bin/
/dist/
coverage.txt
```

- [ ] **Step 6: Smoke-build**

Run from `go/`: `make check-go check-lint build`
Expected: succeeds with no output (no packages yet but the module is valid; lint+go checks pass).

- [ ] **Step 7: Commit**

```bash
cd /Users/rdrake/workspace/afternet/vibebot-v8
git add go/go.mod go/Makefile go/.gitignore
git commit -m "feat(go): initialize v9 Go module skeleton"
```

---

## Phase 1 — External interface contracts

### Task 2: Define `llmcore` interface (sub-project B contract)

This task defines the types `CachedPrefix`, `ToolSchema`, `CacheHandle`, `Canonical()`, plus the `Client` interface. `Canonical(CachedPrefix) []byte` is the single byte serializer that pins the cache-discipline contract.

**Files:**
- Create: `go/llmcore/client.go`
- Create: `go/llmcore/canonical_test.go`

- [ ] **Step 1: Write the failing test**

```go
package llmcore_test

import (
	"bytes"
	"testing"

	"github.com/rdrake/vibebot/v9/llmcore"
)

func TestCanonicalIsDelimited(t *testing.T) {
	cp := llmcore.CachedPrefix{
		FrameworkPrompt: "F",
		Overlay:         "O",
		ToolSchemasJSON: []byte(`[]`),
		ChannelContext:  "C",
	}
	got := string(llmcore.Canonical(cp))
	want := "F\n\n---\n\nO\n\n---\n\n[]\n\n---\n\nC"
	if got != want {
		t.Fatalf("\ngot  %q\nwant %q", got, want)
	}
}

func TestCanonicalByteIdenticalAcrossEqualInputs(t *testing.T) {
	cp1 := llmcore.CachedPrefix{
		FrameworkPrompt: "F", Overlay: "O",
		ToolSchemasJSON: []byte(`[{"name":"a"},{"name":"b"}]`),
		ChannelContext:  "Network: n\nChannel: c\nProfile: p\n",
	}
	cp2 := cp1
	a := llmcore.Canonical(cp1)
	b := llmcore.Canonical(cp2)
	if !bytes.Equal(a, b) {
		t.Fatalf("not byte-identical:\nA=%s\nB=%s", a, b)
	}
}

func TestCanonicalDifferentInputsDifferBytes(t *testing.T) {
	a := llmcore.Canonical(llmcore.CachedPrefix{FrameworkPrompt: "F1", Overlay: "O", ToolSchemasJSON: []byte(`[]`), ChannelContext: "C"})
	b := llmcore.Canonical(llmcore.CachedPrefix{FrameworkPrompt: "F2", Overlay: "O", ToolSchemasJSON: []byte(`[]`), ChannelContext: "C"})
	if bytes.Equal(a, b) {
		t.Fatal("framework change must change canonical bytes")
	}
}

func TestCacheHandleZeroIsUncached(t *testing.T) {
	var h llmcore.CacheHandle
	if h.IsCached() {
		t.Fatal("zero CacheHandle should be uncached")
	}
	h.Name = "x"
	if !h.IsCached() {
		t.Fatal("non-empty Name should be cached")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run from `go/`: `go test ./llmcore/...`
Expected: FAIL — `llmcore` package types and `Canonical` not defined.

- [ ] **Step 3: Write `go/llmcore/client.go`**

```go
// Package llmcore defines the LLM client contract consumed by the executor.
// The real implementation lives in sub-project B.
package llmcore

import (
	"bytes"
	"context"
	"time"
)

// CachedPrefix is the deterministic content that goes into the LLM cache.
// The four fields are serialized by Canonical into a single byte block.
// E populates this struct; B passes Canonical(cp) to the provider.
//
// Schema canonicalization is sub-project C's responsibility: the bytes in
// ToolSchemasJSON MUST already be canonical (sorted keys, no insignificant
// whitespace). E does not re-canonicalize.
type CachedPrefix struct {
	FrameworkPrompt string
	Overlay         string
	ToolSchemasJSON []byte
	ChannelContext  string
}

// Canonical returns the byte representation of cp that the cache key
// must agree with byte-for-byte. The exact ordering and `\n\n---\n\n`
// delimiter are part of the contract with sub-project B.
func Canonical(cp CachedPrefix) []byte {
	const sep = "\n\n---\n\n"
	var buf bytes.Buffer
	buf.Grow(len(cp.FrameworkPrompt) + len(cp.Overlay) + len(cp.ToolSchemasJSON) + len(cp.ChannelContext) + 3*len(sep))
	buf.WriteString(cp.FrameworkPrompt)
	buf.WriteString(sep)
	buf.WriteString(cp.Overlay)
	buf.WriteString(sep)
	buf.Write(cp.ToolSchemasJSON)
	buf.WriteString(sep)
	buf.WriteString(cp.ChannelContext)
	return buf.Bytes()
}

// ToolSchema is one tool's canonical schema, ready for both cache prefix
// composition and Complete calls. Pre-resolved by exec.ResolveSchemas so
// B never needs a back-channel to C.
type ToolSchema struct {
	Name       string
	SchemaJSON []byte
}

// CacheHandle is the opaque reference to a CachedContent (or equivalent).
// Zero value means uncached path (e.g. xAI, or provider without cache).
type CacheHandle struct {
	Name      string
	Provider  string    // "gemini", "xai", ""
	ExpiresAt time.Time // zero when no cache
}

// IsCached reports whether the handle refers to a live cache entry.
func (h CacheHandle) IsCached() bool { return h.Name != "" }

// Message is one entry in the uncached tail (history, memories, user msg,
// or tool result). Role is "system", "user", "assistant", or "tool".
type Message struct {
	Role       string
	Content    string
	ToolCallID string // populated for role=="tool"
}

// ToolCall is what the model emitted.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // raw JSON from the model
}

// Completion is one LLM response.
type Completion struct {
	Text         string
	ToolCalls    []ToolCall
	CachedTokens int
}

// Scope identifies a cache lane. Mirror of router.CacheScope; lives here
// to avoid import cycles (llmcore must not import router).
type Scope struct {
	Network     string
	Channel     string
	Profile     string
	OverlayHash string
}

// Client is the contract.
type Client interface {
	// EnsureCache creates or refreshes a CachedContent for scope+prefix.
	// Returns a CacheHandle; zero value indicates no cache (provider does
	// not support, or scope already has live coverage cheaper than refresh).
	EnsureCache(ctx context.Context, scope Scope, prefix CachedPrefix) (CacheHandle, error)

	// Complete runs one LLM completion. The prefix is passed every call so
	// B can choose the right mode:
	//   - if !cache.IsCached(): B MUST inline Canonical(prefix) ahead of
	//     messages (uncached path; e.g. xAI, or Gemini with no live cache).
	//   - if cache.IsCached(): B MUST reference cache by handle and ignore
	//     the prefix bytes (they were already cached when EnsureCache ran).
	// E never branches on provider; B is the only layer that knows mode.
	Complete(ctx context.Context, prefix CachedPrefix, messages []Message, tools []ToolSchema, model string, cache CacheHandle) (Completion, error)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./llmcore/...`
Expected: PASS all four `TestCanonical*` and `TestCacheHandleZeroIsUncached`.

- [ ] **Step 5: Commit**

```bash
git add go/llmcore/client.go go/llmcore/canonical_test.go
git commit -m "feat(go): define llmcore.Client interface with Canonical serializer"
```

---

### Task 3: Define `tooling` interface (sub-project C contract)

**Files:**
- Create: `go/tooling/dispatcher.go`

- [ ] **Step 1: Write the file**

```go
// Package tooling defines the tool-dispatch contract consumed by the executor.
// The real implementation lives in sub-project C.
package tooling

import "context"

// ToolCall is what the LLM asked the bot to run.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// ToolResult is what to feed back to the LLM.
type ToolResult struct {
	Content string
	Err     error
	Denied  bool // true if profile/rate disallowed; Err must also be non-nil
}

// Dispatcher resolves a tool name to a handler.
type Dispatcher interface {
	// SchemaJSON returns the canonical JSON schema for a tool name.
	// MUST be byte-identical across calls (sorted keys, no insignificant
	// whitespace). Used by exec.ResolveSchemas to populate llmcore.ToolSchema.
	SchemaJSON(name string) ([]byte, error)

	// Dispatch runs a tool call. Honors ctx cancellation. The per-tool
	// timeout is enforced inside Dispatch (sub-project C's responsibility).
	Dispatch(ctx context.Context, call ToolCall) ToolResult
}
```

- [ ] **Step 2: Build**

Run: `go build ./tooling/...`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add go/tooling/dispatcher.go
git commit -m "feat(go): define tooling.Dispatcher interface"
```

---

### Task 4: Define `overlay` interface (sub-project D contract)

**Files:**
- Create: `go/overlay/resolver.go`

- [ ] **Step 1: Write the file**

```go
// Package overlay defines the persona/overlay resolution contract consumed
// by the executor. The real implementation lives in sub-project D.
package overlay

import "context"

// Scope identifies an overlay lookup. NO OverlayHash — the hash is derived
// FROM the returned text, computed by the caller. Including it here would
// be circular.
type Scope struct {
	Network string
	Channel string
	Profile string
}

// Resolver returns the overlay text for a scope. MUST be a pure function:
// same Scope → byte-identical text, across goroutines and calls.
//
// MAY NOT inject user-specific text (nick, account, timestamps). User-
// specific layering happens in the uncached tail.
type Resolver interface {
	Get(ctx context.Context, scope Scope) (text string, err error)
}
```

- [ ] **Step 2: Build**

Run: `go build ./overlay/...`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add go/overlay/resolver.go
git commit -m "feat(go): define overlay.Resolver interface (Scope has no hash)"
```

---

### Task 5: Define `persist` interface (sub-project F contract)

**Files:**
- Create: `go/persist/store.go`

- [ ] **Step 1: Write the file**

```go
// Package persist defines the persistence contract consumed by the executor.
// The real implementation lives in sub-project F.
package persist

import (
	"context"
	"time"
)

type HistoryEntry struct {
	Role      string // "user" or "assistant"
	Nick      string
	Content   string
	Timestamp time.Time
}

type Memory struct {
	ID      int64
	Nick    string
	Fact    string
	Channel string
}

type UsageRow struct {
	Timestamp        time.Time
	Network          string
	Channel          string
	Nick             string
	Profile          string
	Model            string
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
	Cost             float64
	Status           string
	ErrorDetail      string
}

type Store interface {
	History(ctx context.Context, network, channel string, limit int) ([]HistoryEntry, error)
	Memories(ctx context.Context, network, channel, nick string) ([]Memory, error)
	AppendTurn(ctx context.Context, network, channel, nick, assistantText string) error
	RecordUsage(ctx context.Context, row UsageRow) error
}
```

- [ ] **Step 2: Build**

Run: `go build ./persist/...`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add go/persist/store.go
git commit -m "feat(go): define persist.Store interface"
```

---

### Task 6: Define `ircout` interface (sub-project A output contract)

**Files:**
- Create: `go/ircout/sender.go`

- [ ] **Step 1: Write the file**

```go
// Package ircout defines the IRC output contract consumed by the executor.
// The real implementation lives in sub-project A.
package ircout

import "context"

type SendOpts struct {
	Network    string
	Target     string
	Text       string
	ReplyTo    string // IRCv3 +draft/reply target message-id; "" to omit
	NickPrefix string // fallback prefix when ReplyTo CAP unavailable; "" to omit
}

type TypingState string

const (
	TypingActive TypingState = "active"
	TypingDone   TypingState = "done"
)

type Sender interface {
	Send(ctx context.Context, opts SendOpts) error

	// SendTyping is best-effort: no error returned, failures logged at sink.
	SendTyping(ctx context.Context, network, target string, state TypingState)

	// HasReplyCAP reports whether IRCv3 draft/reply was negotiated.
	HasReplyCAP(network string) bool
}
```

- [ ] **Step 2: Build**

Run: `go build ./ircout/...`
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add go/ircout/sender.go
git commit -m "feat(go): define ircout.Sender interface"
```

---

## Phase 2 — Test fakes

### Task 7: `llmcore/fake`

**Files:**
- Create: `go/llmcore/fake/fake.go`
- Create: `go/llmcore/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
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
```

- [ ] **Step 2: Run to fail**

Run: `go test ./llmcore/fake/...`
Expected: FAIL.

- [ ] **Step 3: Write the fake**

```go
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
```

- [ ] **Step 4: Run to pass**

Run: `go test ./llmcore/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/llmcore/fake/
git commit -m "feat(go): add llmcore fake client for tests"
```

---

### Task 8: `tooling/fake` (includes `PanicOnDispatch`)

**Files:**
- Create: `go/tooling/fake/fake.go`
- Create: `go/tooling/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
package fake_test

import (
	"context"
	"errors"
	"testing"

	"github.com/rdrake/vibebot/v9/tooling"
	"github.com/rdrake/vibebot/v9/tooling/fake"
)

func TestDispatchReturnsScripted(t *testing.T) {
	d := fake.New()
	d.Results = map[string]tooling.ToolResult{
		"save_memory": {Content: `{"ok":true}`},
		"search_web":  {Err: errors.New("rate"), Denied: true},
	}
	r := d.Dispatch(context.Background(), tooling.ToolCall{Name: "save_memory"})
	if r.Content != `{"ok":true}` {
		t.Fatal(r)
	}
	r = d.Dispatch(context.Background(), tooling.ToolCall{Name: "search_web"})
	if !r.Denied || r.Err == nil {
		t.Fatal(r)
	}
}

func TestSchemaJSONStable(t *testing.T) {
	d := fake.New()
	d.Schemas = map[string][]byte{"save_memory": []byte(`{"name":"save_memory"}`)}
	a, _ := d.SchemaJSON("save_memory")
	b, _ := d.SchemaJSON("save_memory")
	if string(a) != string(b) {
		t.Fatal("unstable")
	}
}

func TestPanicOnDispatch(t *testing.T) {
	d := fake.New()
	d.PanicOnDispatch = true
	d.PanicMessage = "boom"
	defer func() {
		got := recover()
		if got != "boom" {
			t.Fatalf("recover got %v", got)
		}
	}()
	d.Dispatch(context.Background(), tooling.ToolCall{Name: "x"})
	t.Fatal("expected panic")
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./tooling/fake/...`
Expected: FAIL.

- [ ] **Step 3: Write the fake**

```go
// Package fake provides an in-memory tooling.Dispatcher for tests.
package fake

import (
	"context"
	"errors"
	"sync"

	"github.com/rdrake/vibebot/v9/tooling"
)

type Dispatcher struct {
	Results         map[string]tooling.ToolResult
	Schemas         map[string][]byte
	PanicOnDispatch bool
	PanicMessage    string

	mu    sync.Mutex
	Calls []tooling.ToolCall
}

func New() *Dispatcher {
	return &Dispatcher{
		Results: map[string]tooling.ToolResult{},
		Schemas: map[string][]byte{},
	}
}

func (d *Dispatcher) SchemaJSON(name string) ([]byte, error) {
	s, ok := d.Schemas[name]
	if !ok {
		return nil, errors.New("fake: no schema for " + name)
	}
	return s, nil
}

func (d *Dispatcher) Dispatch(ctx context.Context, call tooling.ToolCall) tooling.ToolResult {
	if d.PanicOnDispatch {
		msg := d.PanicMessage
		if msg == "" {
			msg = "fake panic"
		}
		panic(msg)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.Calls = append(d.Calls, call)
	r, ok := d.Results[call.Name]
	if !ok {
		return tooling.ToolResult{Err: errors.New("fake: no result for " + call.Name)}
	}
	return r
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./tooling/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/tooling/fake/
git commit -m "feat(go): add tooling fake dispatcher with PanicOnDispatch"
```

---

### Task 9: `overlay/fake`

**Files:**
- Create: `go/overlay/fake/fake.go`
- Create: `go/overlay/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
package fake_test

import (
	"context"
	"testing"

	"github.com/rdrake/vibebot/v9/overlay"
	"github.com/rdrake/vibebot/v9/overlay/fake"
)

func TestGetReturnsCanned(t *testing.T) {
	r := fake.New()
	r.Texts = map[overlay.Scope]string{
		{Network: "afternet", Channel: "#x", Profile: "chat"}: "hello overlay",
	}
	got, err := r.Get(context.Background(), overlay.Scope{Network: "afternet", Channel: "#x", Profile: "chat"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello overlay" {
		t.Fatalf("got %q", got)
	}
}

func TestGetIsPureForSameScope(t *testing.T) {
	r := fake.New()
	r.Texts = map[overlay.Scope]string{{Channel: "#x"}: "stable"}
	a, _ := r.Get(context.Background(), overlay.Scope{Channel: "#x"})
	b, _ := r.Get(context.Background(), overlay.Scope{Channel: "#x"})
	if a != b {
		t.Fatalf("not pure: %q vs %q", a, b)
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./overlay/fake/...`
Expected: FAIL.

- [ ] **Step 3: Write the fake**

```go
package fake

import (
	"context"

	"github.com/rdrake/vibebot/v9/overlay"
)

type Resolver struct {
	Texts map[overlay.Scope]string
	Err   error
}

func New() *Resolver { return &Resolver{Texts: map[overlay.Scope]string{}} }

func (r *Resolver) Get(ctx context.Context, scope overlay.Scope) (string, error) {
	if r.Err != nil {
		return "", r.Err
	}
	return r.Texts[scope], nil
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./overlay/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/overlay/fake/
git commit -m "feat(go): add overlay fake resolver for tests"
```

---

### Task 10: `persist/fake`

**Files:**
- Create: `go/persist/fake/fake.go`
- Create: `go/persist/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
package fake_test

import (
	"context"
	"testing"
	"time"

	"github.com/rdrake/vibebot/v9/persist"
	"github.com/rdrake/vibebot/v9/persist/fake"
)

func TestAppendAndHistory(t *testing.T) {
	s := fake.New()
	ctx := context.Background()
	if err := s.AppendTurn(ctx, "n", "#x", "alice", "hi alice"); err != nil {
		t.Fatal(err)
	}
	got, err := s.History(ctx, "n", "#x", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("len: %d", len(got))
	}
}

func TestRecordUsage(t *testing.T) {
	s := fake.New()
	if err := s.RecordUsage(context.Background(), persist.UsageRow{Timestamp: time.Now(), Status: "success"}); err != nil {
		t.Fatal(err)
	}
	if len(s.Usage) != 1 {
		t.Fatal("not recorded")
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./persist/fake/...`
Expected: FAIL.

- [ ] **Step 3: Write the fake**

```go
package fake

import (
	"context"
	"sync"
	"time"

	"github.com/rdrake/vibebot/v9/persist"
)

type Store struct {
	mu             sync.Mutex
	HistoryEntries []persist.HistoryEntry
	MemoryRows     map[memoryKey][]persist.Memory
	Usage          []persist.UsageRow
}

type memoryKey struct{ network, channel, nick string }

func New() *Store { return &Store{MemoryRows: map[memoryKey][]persist.Memory{}} }

func (s *Store) History(ctx context.Context, network, channel string, limit int) ([]persist.HistoryEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := append([]persist.HistoryEntry(nil), s.HistoryEntries...)
	if limit > 0 && len(out) > limit {
		out = out[len(out)-limit:]
	}
	return out, nil
}

func (s *Store) Memories(ctx context.Context, network, channel, nick string) ([]persist.Memory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.MemoryRows[memoryKey{network, channel, nick}], nil
}

func (s *Store) AppendTurn(ctx context.Context, network, channel, nick, assistantText string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.HistoryEntries = append(s.HistoryEntries,
		persist.HistoryEntry{Role: "user", Nick: nick, Content: "(user msg)", Timestamp: time.Now()},
		persist.HistoryEntry{Role: "assistant", Content: assistantText, Timestamp: time.Now()},
	)
	return nil
}

func (s *Store) RecordUsage(ctx context.Context, row persist.UsageRow) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Usage = append(s.Usage, row)
	return nil
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./persist/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/persist/fake/
git commit -m "feat(go): add persist fake store"
```

---

### Task 11: `ircout/fake`

**Files:**
- Create: `go/ircout/fake/fake.go`
- Create: `go/ircout/fake/fake_test.go`

- [ ] **Step 1: Write the failing test**

```go
package fake_test

import (
	"context"
	"testing"

	"github.com/rdrake/vibebot/v9/ircout"
	"github.com/rdrake/vibebot/v9/ircout/fake"
)

func TestSendRecords(t *testing.T) {
	s := fake.New()
	if err := s.Send(context.Background(), ircout.SendOpts{Target: "#x", Text: "hi"}); err != nil {
		t.Fatal(err)
	}
	if len(s.Sent) != 1 || s.Sent[0].Text != "hi" {
		t.Fatal(s.Sent)
	}
}

func TestSendTypingRecords(t *testing.T) {
	s := fake.New()
	s.SendTyping(context.Background(), "n", "#x", ircout.TypingActive)
	s.SendTyping(context.Background(), "n", "#x", ircout.TypingDone)
	if len(s.TypingStates) != 2 {
		t.Fatal(s.TypingStates)
	}
}

func TestHasReplyCAP(t *testing.T) {
	s := fake.New()
	s.ReplyCAP = map[string]bool{"n": true}
	if !s.HasReplyCAP("n") || s.HasReplyCAP("other") {
		t.Fatal("cap mismatch")
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./ircout/fake/...`
Expected: FAIL.

- [ ] **Step 3: Write the fake**

```go
package fake

import (
	"context"
	"sync"

	"github.com/rdrake/vibebot/v9/ircout"
)

type Sender struct {
	ReplyCAP map[string]bool

	mu           sync.Mutex
	Sent         []ircout.SendOpts
	TypingStates []TypingRecord
	SendErr      error
}

type TypingRecord struct {
	Network string
	Target  string
	State   ircout.TypingState
}

func New() *Sender { return &Sender{ReplyCAP: map[string]bool{}} }

func (s *Sender) Send(ctx context.Context, opts ircout.SendOpts) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.SendErr != nil {
		return s.SendErr
	}
	s.Sent = append(s.Sent, opts)
	return nil
}

func (s *Sender) SendTyping(ctx context.Context, network, target string, state ircout.TypingState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.TypingStates = append(s.TypingStates, TypingRecord{network, target, state})
}

func (s *Sender) HasReplyCAP(network string) bool { return s.ReplyCAP[network] }
```

- [ ] **Step 4: Run to pass**

Run: `go test ./ircout/fake/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/ircout/fake/
git commit -m "feat(go): add ircout fake sender for tests"
```

---

## Phase 3 — Profile registry

### Task 12: `router/profile` — Profile struct + Registry (defensive-copy Get)

**Files:**
- Create: `go/router/profile/profile.go`
- Create: `go/router/profile/profile_test.go`

- [ ] **Step 1: Write the failing test**

```go
package profile_test

import (
	"testing"

	"github.com/rdrake/vibebot/v9/router/profile"
)

func TestRegistryLookup(t *testing.T) {
	r := profile.NewRegistry()
	r.Register(profile.Profile{Name: "test", Tools: []string{"a", "b"}, MaxIters: 2})
	got, ok := r.Get("test")
	if !ok || got.MaxIters != 2 {
		t.Fatalf("got %+v ok=%v", got, ok)
	}
}

func TestRegistryUnknownReturnsFalse(t *testing.T) {
	r := profile.NewRegistry()
	if _, ok := r.Get("missing"); ok {
		t.Fatal("should not find")
	}
}

func TestToolsSortedOnRegister(t *testing.T) {
	r := profile.NewRegistry()
	r.Register(profile.Profile{Name: "x", Tools: []string{"c", "a", "b"}})
	p, _ := r.Get("x")
	if p.Tools[0] != "a" || p.Tools[1] != "b" || p.Tools[2] != "c" {
		t.Fatalf("not sorted: %v", p.Tools)
	}
}

func TestRegisterOverwrites(t *testing.T) {
	r := profile.NewRegistry()
	r.Register(profile.Profile{Name: "x", Tools: []string{"a"}, MaxIters: 1})
	r.Register(profile.Profile{Name: "x", Tools: []string{"b"}, MaxIters: 5})
	p, _ := r.Get("x")
	if len(p.Tools) != 1 || p.Tools[0] != "b" || p.MaxIters != 5 {
		t.Fatalf("not overwritten: %+v", p)
	}
}

func TestGetReturnsDefensiveCopy(t *testing.T) {
	r := profile.NewRegistry()
	r.Register(profile.Profile{Name: "x", Tools: []string{"a", "b"}})
	p, _ := r.Get("x")
	p.Tools[0] = "MUTATED"
	p2, _ := r.Get("x")
	if p2.Tools[0] != "a" {
		t.Fatalf("registry mutated through Get: %v", p2.Tools)
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/profile/...`
Expected: FAIL.

- [ ] **Step 3: Write the package**

```go
// Package profile defines the v9 engagement profiles and the lookup registry.
package profile

import "sort"

type Profile struct {
	Name            string
	Tools           []string
	Model           string
	MaxIters        int
	FrameworkPrompt string
	AllowAmbient    bool
}

type Registry struct {
	m map[string]Profile
}

func NewRegistry() *Registry { return &Registry{m: map[string]Profile{}} }

// Register inserts (or overwrites) a profile. Tool list is sorted on insert.
func (r *Registry) Register(p Profile) {
	tools := append([]string(nil), p.Tools...)
	sort.Strings(tools)
	p.Tools = tools
	r.m[p.Name] = p
}

// Get returns a deep copy of the profile, so callers cannot mutate the
// registry's Tools slice through the returned value.
func (r *Registry) Get(name string) (Profile, bool) {
	p, ok := r.m[name]
	if !ok {
		return Profile{}, false
	}
	out := p
	out.Tools = append([]string(nil), p.Tools...)
	return out, true
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./router/profile/...`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add go/router/profile/profile.go go/router/profile/profile_test.go
git commit -m "feat(go): add router/profile registry with defensive-copy Get"
```

---

### Task 13: `router/profile/builtin.go` — v9 profile definitions

**Files:**
- Create: `go/router/profile/builtin.go`
- Create: `go/router/profile/builtin_test.go`

- [ ] **Step 1: Write the failing test**

```go
package profile_test

import (
	"testing"

	"github.com/rdrake/vibebot/v9/router/profile"
)

func TestBuiltinsRegistered(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	for _, name := range []string{"quiet", "chat", "scene", "loom", "admin"} {
		if _, ok := r.Get(name); !ok {
			t.Errorf("missing: %s", name)
		}
	}
}

func TestQuietIsMinimal(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	p, _ := r.Get("quiet")
	for _, banned := range []string{"delete_memory", "update_memory", "generate_image", "search_web", "fetch_url"} {
		for _, t1 := range p.Tools {
			if t1 == banned {
				t.Errorf("quiet contains banned %s", banned)
			}
		}
	}
}

func TestSceneExcludesDestructiveMemoryAndInstruction(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	p, _ := r.Get("scene")
	for _, banned := range []string{"delete_memory", "update_memory", "set_instruction", "clear_instruction"} {
		for _, t1 := range p.Tools {
			if t1 == banned {
				t.Errorf("scene contains banned %s", banned)
			}
		}
	}
}

func TestChatHasDrawAndSearch(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	p, _ := r.Get("chat")
	gotDraw, gotSearch := false, false
	for _, t1 := range p.Tools {
		if t1 == "generate_image" {
			gotDraw = true
		}
		if t1 == "search_web" {
			gotSearch = true
		}
	}
	if !gotDraw || !gotSearch {
		t.Fatalf("missing draw/search: %v", p.Tools)
	}
}

func TestNoRemindersAnywhere(t *testing.T) {
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	for _, name := range []string{"quiet", "chat", "scene", "loom", "admin"} {
		p, _ := r.Get(name)
		for _, banned := range []string{"set_reminder", "cancel_pending_task", "cancel_all_pending_tasks", "schedule_llm_task", "list_pending_tasks"} {
			for _, t1 := range p.Tools {
				if t1 == banned {
					t.Errorf("%s contains %s", name, banned)
				}
			}
		}
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/profile/...`
Expected: FAIL — `RegisterBuiltins` undefined.

- [ ] **Step 3: Write the builtins**

```go
package profile

func RegisterBuiltins(r *Registry) {
	r.Register(Profile{
		Name:            "quiet",
		Tools:           []string{"clear_instruction", "get_instruction", "list_memories", "save_memory", "set_instruction"},
		Model:           "gemini-2.5-flash",
		MaxIters:        2,
		FrameworkPrompt: "You are a quiet, helpful IRC assistant. Speak only when addressed.",
		AllowAmbient:    false,
	})
	r.Register(Profile{
		Name: "chat",
		Tools: []string{
			"clear_instruction", "delete_memory", "fetch_url", "generate_image",
			"get_instruction", "list_memories", "save_memory", "search_web",
			"set_instruction", "update_memory",
		},
		Model:           "gemini-2.5-flash",
		MaxIters:        2,
		FrameworkPrompt: "You are a conversational IRC assistant.",
		AllowAmbient:    true,
	})
	r.Register(Profile{
		Name: "scene",
		Tools: []string{
			"fetch_url", "generate_image", "get_instruction",
			"list_memories", "save_memory", "search_web",
		},
		Model:           "gemini-2.5-pro",
		MaxIters:        3,
		FrameworkPrompt: "You are participating in an ongoing narrative scene. Stay in character.",
		AllowAmbient:    false,
	})
	r.Register(Profile{
		Name: "loom",
		Tools: []string{
			"fetch_url", "generate_image", "get_instruction",
			"list_memories", "loom_propose", "loom_seed", "loom_yield",
			"save_memory", "search_web",
		},
		Model:           "gemini-2.5-pro",
		MaxIters:        3,
		FrameworkPrompt: "You are one of several voices weaving a collaborative narrative.",
		AllowAmbient:    false,
	})
	r.Register(Profile{
		Name: "admin",
		Tools: []string{
			"clear_instruction", "delete_memory", "fetch_url",
			"generate_code", "generate_image", "get_instruction",
			"list_memories", "save_memory", "search_web",
			"set_instruction", "update_memory",
		},
		Model:           "gemini-2.5-pro",
		MaxIters:        5,
		FrameworkPrompt: "You are the assistant. The user has admin privileges.",
		AllowAmbient:    false,
	})
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./router/profile/...`
Expected: PASS (all builtin tests green).

- [ ] **Step 5: Commit**

```bash
git add go/router/profile/builtin.go go/router/profile/builtin_test.go
git commit -m "feat(go): add v9 builtin profile definitions"
```

---

## Phase 4 — Router types and helpers

### Task 14: `router/types.go`

**Files:**
- Create: `go/router/doc.go`
- Create: `go/router/types.go`

- [ ] **Step 1: Write doc.go**

```go
// Package router converts inbound IRC events into RouteDecision values.
// Route is a pure function; the stateful executor lives in package exec.
package router
```

- [ ] **Step 2: Write types.go**

```go
package router

import "time"

type IRCEvent struct {
	Network    string
	Channel    string
	Nick       string
	Account    string
	Text       string
	IsAction   bool
	Tags       map[string]string
	MessageID  string
	ReceivedAt time.Time
}

type SceneRef struct{ ID string }
type LoomRef struct{ ID string }

type ChannelKey struct {
	Network string
	Channel string
}

// ChannelState is hydrated by the dispatcher BEFORE Route is called. The
// Overlay field is the pre-resolved overlay text from D's Resolver.Get —
// Route hashes it to build CacheScope.OverlayHash. Executor will re-resolve
// via D for determinism, but D's purity contract guarantees the same text.
type ChannelState struct {
	Profile         string
	Overlay         string
	AmbientEnabled  bool
	AmbientCooldown time.Duration
	SceneActive     *SceneRef
	LoomActive      *LoomRef
}

type BotState struct {
	SelfNick      string
	LastAmbientAt map[ChannelKey]time.Time
	Now           time.Time
	RecentSentIDs []string
}

type Action int

const (
	Ignore Action = iota
	RespondChat
	RespondScene
	RespondLoom
)

func (a Action) String() string {
	switch a {
	case Ignore:
		return "Ignore"
	case RespondChat:
		return "RespondChat"
	case RespondScene:
		return "RespondScene"
	case RespondLoom:
		return "RespondLoom"
	default:
		return "Unknown"
	}
}

type CacheScope struct {
	Network     string
	Channel     string
	Profile     string
	OverlayHash string
}

type PromptSpec struct {
	HistoryLimit int
	UserText     string
	UserNick     string
}

type DeliverySpec struct {
	Typing               bool
	ChunkSize            int
	ReplyToID            string
	NickPrefixOnFallback bool
	MaxIters             int
}

type RouteDecision struct {
	Action     Action
	Profile    string
	Tools      []string
	CacheScope CacheScope
	Model      string
	Prompt     PromptSpec
	Delivery   DeliverySpec
	Event      IRCEvent
}
```

- [ ] **Step 3: Build**

Run: `go build ./router/...`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add go/router/doc.go go/router/types.go
git commit -m "feat(go): add router data types"
```

---

### Task 15: `router/addressed.go`

**Files:**
- Create: `go/router/addressed.go`
- Create: `go/router/addressed_test.go`

- [ ] **Step 1: Write the failing tests**

```go
package router

import "testing"

func TestAddressedDM(t *testing.T) {
	if !addressed(IRCEvent{Channel: "", Text: "anything"}, "vibebot", nil) {
		t.Fatal("DM should be addressed")
	}
}

func TestAddressedFirstTokenMatch(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"vibebot: hello", true},
		{"vibebot, hi", true},
		{"vibebot; what?", true},
		{"VIBEBOT hi", true},
		{"  vibebot   hi", true},
		{"vibebot:\nhello", true},     // newline whitespace
		{"vibebot:\thi", true},        // tab
		{"hi vibebot", false},         // mid-message — out of scope for v1
		{"vibebotsomething hi", false},
		{"", false},
	}
	for _, c := range cases {
		got := addressed(IRCEvent{Channel: "#x", Text: c.text}, "vibebot", nil)
		if got != c.want {
			t.Errorf("addressed(%q) = %v want %v", c.text, got, c.want)
		}
	}
}

func TestAddressedReplyTag(t *testing.T) {
	recent := []string{"msg-1", "msg-2"}
	evt := IRCEvent{Channel: "#x", Text: "yes", Tags: map[string]string{"+draft/reply": "msg-1"}}
	if !addressed(evt, "vibebot", recent) {
		t.Fatal("reply to recent bot msg should be addressed")
	}
	evt.Tags["+draft/reply"] = "unknown-id"
	if addressed(evt, "vibebot", recent) {
		t.Fatal("reply to non-bot msg should not be addressed")
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/ -run TestAddressed`
Expected: FAIL.

- [ ] **Step 3: Write the function**

```go
package router

import "strings"

// addressed reports whether evt is addressed to selfNick.
//
// True iff ANY of:
//   - evt.Channel == "" (DM)
//   - evt.Tags["+draft/reply"] is in recentSentIDs
//   - first whitespace-delimited token of evt.Text, lowercased and stripped
//     of trailing punctuation in [:,;.!?], equals strings.ToLower(selfNick)
//
// Whitespace is any unicode whitespace (via strings.Fields).
// Mid-message mention is intentionally NOT addressed for v1.
func addressed(evt IRCEvent, selfNick string, recentSentIDs []string) bool {
	if evt.Channel == "" {
		return true
	}
	if replyID := evt.Tags["+draft/reply"]; replyID != "" {
		for _, id := range recentSentIDs {
			if id == replyID {
				return true
			}
		}
	}
	fields := strings.Fields(evt.Text)
	if len(fields) == 0 {
		return false
	}
	return strings.EqualFold(stripTrailingPunct(fields[0]), selfNick)
}

func stripTrailingPunct(s string) string {
	return strings.TrimRight(s, ":,;.!?")
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./router/ -run TestAddressed -v`
Expected: PASS for all cases including the newline case.

- [ ] **Step 5: Commit**

```bash
git add go/router/addressed.go go/router/addressed_test.go
git commit -m "feat(go): add router addressed() with unicode-aware tokenization"
```

---

### Task 16: `router/scope.go` — OverlayHash + CacheScope

**Files:**
- Create: `go/router/scope.go`
- Create: `go/router/scope_test.go`

- [ ] **Step 1: Write the failing tests**

```go
package router

import "testing"

func TestOverlayHashStable(t *testing.T) {
	a := overlayHash("hello world")
	b := overlayHash("hello world")
	if a != b {
		t.Fatalf("not stable")
	}
	if len(a) != 32 {
		t.Fatalf("len: %d", len(a))
	}
}

func TestOverlayHashDifferentInputs(t *testing.T) {
	if overlayHash("a") == overlayHash("b") {
		t.Fatal("collision on trivial inputs")
	}
}

func TestBuildCacheScope(t *testing.T) {
	got := buildCacheScope("afternet", "#x", "chat", "overlay text")
	if got.Network != "afternet" || got.Channel != "#x" || got.Profile != "chat" {
		t.Fatalf("%+v", got)
	}
	if len(got.OverlayHash) != 32 {
		t.Fatalf("hash: %q", got.OverlayHash)
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/ -run "TestOverlay|TestBuildCache"`
Expected: FAIL.

- [ ] **Step 3: Write the helpers**

```go
package router

import (
	"crypto/sha256"
	"encoding/hex"
)

// overlayHash is hex sha256 truncated to 16 bytes (32 hex chars).
// Same input → same hash, byte-identical.
func overlayHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:16])
}

func buildCacheScope(network, channel, profile, overlay string) CacheScope {
	return CacheScope{
		Network:     network,
		Channel:     channel,
		Profile:     profile,
		OverlayHash: overlayHash(overlay),
	}
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./router/ -run "TestOverlay|TestBuildCache" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/router/scope.go go/router/scope_test.go
git commit -m "feat(go): add router OverlayHash and CacheScope builder"
```

---

## Phase 5 — Route() decision function

### Task 17: `router/route.go`

This task is split into two implementation steps to keep each one under ~50 lines.

**Files:**
- Create: `go/router/route.go`
- Create: `go/router/route_test.go`

- [ ] **Step 1: Write the failing tests**

```go
package router

import (
	"sync"
	"testing"
	"time"

	"github.com/rdrake/vibebot/v9/router/profile"
)

func newTestRegistry(t *testing.T) *profile.Registry {
	t.Helper()
	r := profile.NewRegistry()
	profile.RegisterBuiltins(r)
	return r
}

func TestRouteIgnoresUnaddressedNoAmbient(t *testing.T) {
	r := newTestRegistry(t)
	evt := IRCEvent{Channel: "#x", Text: "random", Network: "afternet", ReceivedAt: time.Now()}
	state := ChannelState{Profile: "chat", AmbientEnabled: false}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	if d := Route(evt, state, bot, r); d.Action != Ignore {
		t.Fatalf("got %v", d.Action)
	}
}

func TestRouteAddressedChat(t *testing.T) {
	r := newTestRegistry(t)
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", ReceivedAt: time.Now()}
	state := ChannelState{Profile: "chat", Overlay: "be friendly"}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	d := Route(evt, state, bot, r)
	if d.Action != RespondChat || d.Profile != "chat" {
		t.Fatalf("%+v", d)
	}
	if d.CacheScope.OverlayHash == "" {
		t.Fatal("hash empty")
	}
}

func TestRouteSceneTakesPriorityOverChatWhenAddressed(t *testing.T) {
	r := newTestRegistry(t)
	state := ChannelState{Profile: "chat", SceneActive: &SceneRef{ID: "s"}}
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", ReceivedAt: time.Now()}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	if d := Route(evt, state, bot, r); d.Action != RespondScene {
		t.Fatalf("got %v", d.Action)
	}
}

func TestRouteLoomTakesPriorityOverScene(t *testing.T) {
	r := newTestRegistry(t)
	state := ChannelState{Profile: "chat", SceneActive: &SceneRef{ID: "s"}, LoomActive: &LoomRef{ID: "l"}}
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", ReceivedAt: time.Now()}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	if d := Route(evt, state, bot, r); d.Action != RespondLoom {
		t.Fatalf("got %v", d.Action)
	}
}

func TestRouteAmbientPassesCooldown(t *testing.T) {
	r := newTestRegistry(t)
	now := time.Now()
	state := ChannelState{Profile: "chat", AmbientEnabled: true, AmbientCooldown: 60 * time.Second}
	evt := IRCEvent{Channel: "#x", Text: "general", Network: "afternet", ReceivedAt: now}
	bot := BotState{
		SelfNick:      "vibebot",
		Now:           now,
		LastAmbientAt: map[ChannelKey]time.Time{{Network: "afternet", Channel: "#x"}: now.Add(-90 * time.Second)},
	}
	if d := Route(evt, state, bot, r); d.Action != RespondChat {
		t.Fatalf("got %v", d.Action)
	}
}

func TestRouteAmbientBlockedByCooldown(t *testing.T) {
	r := newTestRegistry(t)
	now := time.Now()
	state := ChannelState{Profile: "chat", AmbientEnabled: true, AmbientCooldown: 60 * time.Second}
	evt := IRCEvent{Channel: "#x", Text: "general", Network: "afternet", ReceivedAt: now}
	bot := BotState{
		SelfNick:      "vibebot",
		Now:           now,
		LastAmbientAt: map[ChannelKey]time.Time{{Network: "afternet", Channel: "#x"}: now.Add(-30 * time.Second)},
	}
	if d := Route(evt, state, bot, r); d.Action != Ignore {
		t.Fatalf("got %v", d.Action)
	}
}

func TestRouteAmbientNeverTriggersSceneOrLoom(t *testing.T) {
	r := newTestRegistry(t)
	now := time.Now()
	state := ChannelState{
		Profile:         "chat",
		SceneActive:     &SceneRef{ID: "s"},
		AmbientEnabled:  true,
		AmbientCooldown: time.Second,
	}
	evt := IRCEvent{Channel: "#x", Text: "general", Network: "afternet", ReceivedAt: now}
	bot := BotState{
		SelfNick:      "vibebot",
		Now:           now,
		LastAmbientAt: map[ChannelKey]time.Time{{Network: "afternet", Channel: "#x"}: now.Add(-1 * time.Hour)},
	}
	if d := Route(evt, state, bot, r); d.Action != RespondChat {
		t.Fatalf("got %v", d.Action)
	}
}

func TestRouteUnknownProfileFallsBackToQuiet(t *testing.T) {
	r := newTestRegistry(t)
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", ReceivedAt: time.Now()}
	state := ChannelState{Profile: "doesnotexist"}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	d := Route(evt, state, bot, r)
	if d.Profile != "quiet" {
		t.Fatalf("fallback: %q", d.Profile)
	}
}

func TestRouteCarriesEvent(t *testing.T) {
	r := newTestRegistry(t)
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", MessageID: "m-1", ReceivedAt: time.Now()}
	state := ChannelState{Profile: "chat"}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	if d := Route(evt, state, bot, r); d.Event.MessageID != "m-1" {
		t.Fatalf("not carried: %+v", d.Event)
	}
}

func TestRouteIsPureUnderConcurrentCalls(t *testing.T) {
	r := newTestRegistry(t)
	evt := IRCEvent{Channel: "#x", Text: "vibebot: hi", Network: "afternet", ReceivedAt: time.Now()}
	state := ChannelState{Profile: "chat", Overlay: "be friendly"}
	bot := BotState{SelfNick: "vibebot", Now: time.Now()}
	want := Route(evt, state, bot, r)

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := Route(evt, state, bot, r)
			if got.Action != want.Action || got.Profile != want.Profile || got.CacheScope.OverlayHash != want.CacheScope.OverlayHash {
				t.Errorf("non-deterministic")
			}
		}()
	}
	wg.Wait()
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./router/ -run TestRoute`
Expected: FAIL — `Route` undefined.

- [ ] **Step 3: Write `decideAction` (decision-tree only)**

```go
package router

func decideAction(evt IRCEvent, state ChannelState, bot BotState) Action {
	if addressed(evt, bot.SelfNick, bot.RecentSentIDs) {
		switch {
		case state.LoomActive != nil:
			return RespondLoom
		case state.SceneActive != nil:
			return RespondScene
		default:
			return RespondChat
		}
	}
	if !state.AmbientEnabled {
		return Ignore
	}
	key := ChannelKey{Network: evt.Network, Channel: evt.Channel}
	if evt.ReceivedAt.Sub(bot.LastAmbientAt[key]) < state.AmbientCooldown {
		return Ignore
	}
	return RespondChat
}
```

- [ ] **Step 4: Write `Route` (wires registry + scope + decision)**

```go
package router

import "github.com/rdrake/vibebot/v9/router/profile"

// Route is the pure decision function. Same inputs → same RouteDecision.
//
// The dispatcher is responsible for ambient claim atomicity (turn lock +
// compare-and-set on LastAmbientAt) before invoking Executor.Run. Route
// reads the snapshot it was given and does not mutate anything.
func Route(evt IRCEvent, state ChannelState, bot BotState, reg *profile.Registry) RouteDecision {
	profileName := state.Profile
	p, ok := reg.Get(profileName)
	if !ok {
		profileName = "quiet"
		p, ok = reg.Get(profileName)
		if !ok {
			return RouteDecision{Action: Ignore, Event: evt}
		}
	}

	action := decideAction(evt, state, bot)
	if action == Ignore {
		return RouteDecision{Action: Ignore, Event: evt, Profile: profileName}
	}

	scope := buildCacheScope(evt.Network, evt.Channel, profileName, state.Overlay)
	return RouteDecision{
		Action:     action,
		Profile:    profileName,
		Tools:      p.Tools, // already a defensive copy from registry.Get
		CacheScope: scope,
		Model:      p.Model,
		Prompt: PromptSpec{
			HistoryLimit: 20,
			UserText:     evt.Text,
			UserNick:     evt.Nick,
		},
		Delivery: DeliverySpec{
			Typing:               true,
			ChunkSize:            380,
			ReplyToID:            evt.MessageID,
			NickPrefixOnFallback: evt.Channel != "",
			MaxIters:             p.MaxIters,
		},
		Event: evt,
	}
}
```

- [ ] **Step 5: Run to pass**

Run: `go test ./router/... -race`
Expected: PASS for all router tests including the concurrent purity test.

- [ ] **Step 6: Commit**

```bash
git add go/router/route.go go/router/route_test.go
git commit -m "feat(go): add router.Route() pure decision function"
```

---

## Phase 6 — Executor

Phase 6 is split into smaller tasks: errors, ResolveSchemas, BuildCachedPrefix, chunking, loop, then the full Executor.Run wiring.

### Task 18: `exec/errors.go` — typed error categories

**Files:**
- Create: `go/exec/doc.go`
- Create: `go/exec/errors.go`
- Create: `go/exec/errors_test.go`

- [ ] **Step 1: Write doc.go**

```go
// Package exec runs the LLM tool-call loop and delivers responses to IRC.
// Constructed once at startup with concrete A/B/C/D/F implementations.
package exec
```

- [ ] **Step 2: Write the failing test**

```go
package exec

import (
	"errors"
	"testing"
)

func TestErrCategoriesDistinct(t *testing.T) {
	cats := []error{
		ErrLLMTransient, ErrLLMFatal, ErrToolDenied, ErrToolFailed,
		ErrIRCSend, ErrCacheStale, ErrBudgetExceeded,
	}
	for i, a := range cats {
		if a == nil {
			t.Errorf("cat %d nil", i)
		}
		for j, b := range cats {
			if i == j {
				continue
			}
			if errors.Is(a, b) {
				t.Errorf("%v incorrectly Is(%v)", a, b)
			}
		}
	}
}
```

- [ ] **Step 3: Run to fail**

Run: `go test ./exec/ -run TestErr`
Expected: FAIL — error variables undefined.

- [ ] **Step 4: Write `errors.go`**

```go
package exec

import "errors"

var (
	ErrLLMTransient   = errors.New("exec: llm transient failure")
	ErrLLMFatal       = errors.New("exec: llm fatal failure")
	ErrToolDenied     = errors.New("exec: tool denied")
	ErrToolFailed     = errors.New("exec: tool execution failed")
	ErrIRCSend        = errors.New("exec: irc send failed")
	ErrCacheStale     = errors.New("exec: cache stale")
	ErrBudgetExceeded = errors.New("exec: budget exceeded")
)
```

- [ ] **Step 5: Run to pass**

Run: `go test ./exec/ -run TestErr`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add go/exec/doc.go go/exec/errors.go go/exec/errors_test.go
git commit -m "feat(go): add exec typed errors with distinctness test"
```

---

### Task 19: `exec/resolve.go` — ResolveSchemas helper

**Files:**
- Create: `go/exec/resolve.go`
- Create: `go/exec/resolve_test.go`

- [ ] **Step 1: Write the failing test**

```go
package exec_test

import (
	"testing"

	"github.com/rdrake/vibebot/v9/exec"
	toolfake "github.com/rdrake/vibebot/v9/tooling/fake"
)

func TestResolveSchemasReturnsInOrder(t *testing.T) {
	td := toolfake.New()
	td.Schemas = map[string][]byte{
		"a": []byte(`{"name":"a"}`),
		"b": []byte(`{"name":"b"}`),
	}
	got, err := exec.ResolveSchemas(td, []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Name != "a" || got[1].Name != "b" {
		t.Fatalf("got %+v", got)
	}
}

func TestResolveSchemasMissingErrors(t *testing.T) {
	td := toolfake.New()
	if _, err := exec.ResolveSchemas(td, []string{"missing"}); err == nil {
		t.Fatal("expected error")
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./exec/ -run TestResolveSchemas`
Expected: FAIL.

- [ ] **Step 3: Write `resolve.go`**

```go
package exec

import (
	"fmt"

	"github.com/rdrake/vibebot/v9/llmcore"
	"github.com/rdrake/vibebot/v9/tooling"
)

// ResolveSchemas converts tool names into ToolSchema via the dispatcher.
// Caller-side resolution means B never needs a back-channel to C.
func ResolveSchemas(d tooling.Dispatcher, names []string) ([]llmcore.ToolSchema, error) {
	out := make([]llmcore.ToolSchema, 0, len(names))
	for _, n := range names {
		b, err := d.SchemaJSON(n)
		if err != nil {
			return nil, fmt.Errorf("schema for %q: %w", n, err)
		}
		out = append(out, llmcore.ToolSchema{Name: n, SchemaJSON: b})
	}
	return out, nil
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./exec/ -run TestResolveSchemas -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/exec/resolve.go go/exec/resolve_test.go
git commit -m "feat(go): add exec.ResolveSchemas to pre-resolve tools"
```

---

### Task 20: `exec/prefix.go` — BuildCachedPrefix

**Files:**
- Create: `go/exec/prefix.go`
- Create: `go/exec/prefix_test.go`

- [ ] **Step 1: Write the failing test**

```go
package exec_test

import (
	"bytes"
	"testing"

	"github.com/rdrake/vibebot/v9/exec"
	"github.com/rdrake/vibebot/v9/llmcore"
)

func TestBuildCachedPrefixFields(t *testing.T) {
	cp := exec.BuildCachedPrefix(exec.BuildPrefixArgs{
		Framework: "fp", Overlay: "ov",
		Tools: []llmcore.ToolSchema{
			{Name: "a", SchemaJSON: []byte(`{"name":"a"}`)},
			{Name: "b", SchemaJSON: []byte(`{"name":"b"}`)},
		},
		Network: "afternet", Channel: "#x", Profile: "chat",
	})
	if cp.FrameworkPrompt != "fp" || cp.Overlay != "ov" {
		t.Fatalf("%+v", cp)
	}
	if cp.ChannelContext != "Network: afternet\nChannel: #x\nProfile: chat\n" {
		t.Fatalf("ctx: %q", cp.ChannelContext)
	}
	if !bytes.Contains(cp.ToolSchemasJSON, []byte(`"name":"a"`)) ||
		!bytes.Contains(cp.ToolSchemasJSON, []byte(`"name":"b"`)) {
		t.Fatalf("schemas: %s", cp.ToolSchemasJSON)
	}
}

func TestBuildCachedPrefixToolOrderStable(t *testing.T) {
	a := exec.BuildCachedPrefix(exec.BuildPrefixArgs{
		Framework: "fp", Overlay: "ov",
		Tools: []llmcore.ToolSchema{
			{Name: "b", SchemaJSON: []byte(`{"name":"b"}`)},
			{Name: "a", SchemaJSON: []byte(`{"name":"a"}`)},
		},
		Network: "n", Channel: "c", Profile: "p",
	})
	b := exec.BuildCachedPrefix(exec.BuildPrefixArgs{
		Framework: "fp", Overlay: "ov",
		Tools: []llmcore.ToolSchema{
			{Name: "a", SchemaJSON: []byte(`{"name":"a"}`)},
			{Name: "b", SchemaJSON: []byte(`{"name":"b"}`)},
		},
		Network: "n", Channel: "c", Profile: "p",
	})
	if !bytes.Equal(llmcore.Canonical(a), llmcore.Canonical(b)) {
		t.Fatalf("canonical bytes differ across input order:\nA=%s\nB=%s", llmcore.Canonical(a), llmcore.Canonical(b))
	}
}

func TestBuildCachedPrefixDifferentSchemasDiffer(t *testing.T) {
	a := exec.BuildCachedPrefix(exec.BuildPrefixArgs{
		Framework: "fp", Overlay: "ov",
		Tools:   []llmcore.ToolSchema{{Name: "x", SchemaJSON: []byte(`{"v":1}`)}},
		Network: "n", Channel: "c", Profile: "p",
	})
	b := exec.BuildCachedPrefix(exec.BuildPrefixArgs{
		Framework: "fp", Overlay: "ov",
		Tools:   []llmcore.ToolSchema{{Name: "x", SchemaJSON: []byte(`{"v":2}`)}},
		Network: "n", Channel: "c", Profile: "p",
	})
	if bytes.Equal(llmcore.Canonical(a), llmcore.Canonical(b)) {
		t.Fatal("different schemas must produce different canonical bytes")
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./exec/ -run TestBuildCached`
Expected: FAIL.

- [ ] **Step 3: Implement `BuildCachedPrefix`**

```go
package exec

import (
	"bytes"
	"fmt"
	"sort"

	"github.com/rdrake/vibebot/v9/llmcore"
)

// BuildPrefixArgs is the input to BuildCachedPrefix.
// Tools must already be resolved schemas (see ResolveSchemas). The schemas'
// bytes must be canonical per sub-project C's contract; this function does
// NOT re-canonicalize JSON keys or whitespace.
type BuildPrefixArgs struct {
	Framework string
	Overlay   string
	Tools     []llmcore.ToolSchema
	Network   string
	Channel   string
	Profile   string
}

// BuildCachedPrefix assembles the four-block cacheable prefix per the spec
// section "Cache prefix composition". Tools are sorted by name before
// serialization to make output insensitive to caller-side ordering.
//
// Use llmcore.Canonical(cp) to get the cacheable byte block.
func BuildCachedPrefix(args BuildPrefixArgs) llmcore.CachedPrefix {
	tools := append([]llmcore.ToolSchema(nil), args.Tools...)
	sort.Slice(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })

	var buf bytes.Buffer
	buf.WriteByte('[')
	for i, t := range tools {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.Write(t.SchemaJSON)
	}
	buf.WriteByte(']')

	return llmcore.CachedPrefix{
		FrameworkPrompt: args.Framework,
		Overlay:         args.Overlay,
		ToolSchemasJSON: buf.Bytes(),
		ChannelContext:  fmt.Sprintf("Network: %s\nChannel: %s\nProfile: %s\n", args.Network, args.Channel, args.Profile),
	}
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./exec/ -run TestBuildCached -v`
Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
git add go/exec/prefix.go go/exec/prefix_test.go
git commit -m "feat(go): add exec.BuildCachedPrefix backed by llmcore.Canonical"
```

---

### Task 21: `exec/chunk.go` — chunkAt helper

**Files:**
- Create: `go/exec/chunk.go`
- Create: `go/exec/chunk_test.go`

- [ ] **Step 1: Write the failing tests**

```go
package exec

import "testing"

func TestChunkAtSplits(t *testing.T) {
	got := chunkAt("aaaaaaaaaaaaaaa", 5)
	if len(got) != 3 {
		t.Fatalf("len: %d", len(got))
	}
	for _, c := range got {
		if len(c) > 5 {
			t.Errorf("chunk too big: %q", c)
		}
	}
}

func TestChunkAtPreservesShort(t *testing.T) {
	got := chunkAt("hi", 380)
	if len(got) != 1 || got[0] != "hi" {
		t.Fatal(got)
	}
}

func TestChunkAtZeroSizeReturnsSingle(t *testing.T) {
	got := chunkAt("hello", 0)
	if len(got) != 1 || got[0] != "hello" {
		t.Fatal(got)
	}
}

func TestChunkAtExactBoundary(t *testing.T) {
	got := chunkAt("hello", 5)
	if len(got) != 1 || got[0] != "hello" {
		t.Fatalf("got %v", got)
	}
}
```

- [ ] **Step 2: Run to fail**

Run: `go test ./exec/ -run TestChunkAt`
Expected: FAIL.

- [ ] **Step 3: Implement chunkAt**

```go
package exec

// chunkAt splits s into pieces of at most n bytes. n<=0 returns [s].
// Byte-naive on purpose; sub-project A is responsible for wire-level
// validation (UTF-8 safety, IRC line length).
func chunkAt(s string, n int) []string {
	if n <= 0 || len(s) <= n {
		return []string{s}
	}
	var out []string
	for len(s) > n {
		out = append(out, s[:n])
		s = s[n:]
	}
	if len(s) > 0 {
		out = append(out, s)
	}
	return out
}
```

- [ ] **Step 4: Run to pass**

Run: `go test ./exec/ -run TestChunkAt -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/exec/chunk.go go/exec/chunk_test.go
git commit -m "feat(go): add exec.chunkAt"
```

---

### Task 22: `exec/loop.go` — tool-call loop

**Files:**
- Create: `go/exec/loop.go`
- Create: `go/exec/loop_test.go`

- [ ] **Step 1: Write the failing test**

```go
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
```

- [ ] **Step 2: Run to fail**

Run: `go test ./exec/ -run TestRunLoop`
Expected: FAIL.

- [ ] **Step 3: Implement runLoop**

```go
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
```

- [ ] **Step 4: Run to pass**

Run: `go test ./exec/ -run TestRunLoop -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add go/exec/loop.go go/exec/loop_test.go
git commit -m "feat(go): add exec.runLoop tool-call loop"
```

---

### Task 23: `exec/executor.go` — Executor wiring (split into 4 implementation steps)

> **Note:** Implementation steps 3, 4, and 5 compose into the final file. Steps 3 and 4 leave `executor.go` in an incomplete state (Step 3's `runInner` is a stub; Step 4 references `runWithCache` which Step 5 defines). Do not run executor tests until after Step 5; the failing-tests assertion in Step 2 is the only test step until then. Steps 3–5 together replace the file's content; final compile/test happens in Step 6.

**Files:**
- Create: `go/exec/executor.go`
- Create: `go/exec/executor_test.go`

- [ ] **Step 1: Write the failing tests**

```go
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
```

- [ ] **Step 2: Run to fail**

Run: `go test ./exec/ -run TestRun -v`
Expected: FAIL — `exec.New`/`exec.Executor`/`exec.Config` undefined.

- [ ] **Step 3: Write `Executor` struct + constructor + skeleton `Run` (typing on/off, fast-return on Ignore)**

```go
package exec

import (
	"context"
	"log/slog"

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

// Run executes one routing decision end-to-end. Sub-steps below add the
// overlay/prefix/loop/deliver/persist phases.
func (e *Executor) Run(ctx context.Context, d router.RouteDecision) (err error) {
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
	// Filled in by subsequent steps.
	return nil
}

func deliveryTarget(d router.RouteDecision) string {
	if d.Event.Channel != "" {
		return d.Event.Channel
	}
	return d.Event.Nick
}
```

- [ ] **Step 4: Add overlay resolve, schema resolve, prefix build, cache ensure to `runInner`**

Replace `runInner` with:

```go
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
```

Add `"time"` to the imports if not already present, and add `var timeNow = time.Now` at the bottom of the file (separate so tests can override later). `recordFailure` uses `timeNow()`.

- [ ] **Step 5: Add `runWithCache` (hydrate tail + run loop + deliver + persist)**

```go
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
```

- [ ] **Step 6: Run all executor tests**

Run: `go test ./exec/... -race -v`
Expected: PASS for all `TestRun*` tests including the panic test.

- [ ] **Step 7: Commit**

```bash
git add go/exec/executor.go go/exec/executor_test.go
git commit -m "feat(go): add exec.Executor.Run with overlay/cache/loop/deliver/persist"
```

---

## Phase 7 — Demo

### Task 24: `cmd/routerdemo/main.go`

**Files:**
- Create: `go/cmd/routerdemo/main.go`

- [ ] **Step 1: Write the demo**

```go
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
```

- [ ] **Step 2: Build the demo**

Run: `go build ./cmd/routerdemo/`
Expected: success.

- [ ] **Step 3: Run the demo**

Run: `go run ./cmd/routerdemo/`
Expected output (approximately):
```
Decision: action=RespondChat profile=chat tools=10 model=gemini-2.5-flash overlay_hash=<32-hex>
Sent 1 msg(s); first: "hi alice, how can i help?"
Typing events: 2
Usage rows: 1 (cached_tokens=0, status=success)
```

- [ ] **Step 4: Commit**

```bash
git add go/cmd/routerdemo/main.go
git commit -m "feat(go): add routerdemo wiring sample"
```

---

### Task 25: Full sweep

- [ ] **Step 1: Run all Go tests with race detector**

Run from `go/`: `make test`
Expected: ALL packages PASS, race detector clean.

- [ ] **Step 2: Vet**

Run: `make vet`
Expected: no warnings.

- [ ] **Step 3: Format check**

Run: `gofmt -l .`
Expected: empty output.

- [ ] **Step 4: Lint**

Run: `make lint`
Expected: no findings (or only minor style nits depending on linter config).

- [ ] **Step 5: Commit (any auto-format fixes if needed)**

```bash
git status
# if anything is modified by gofmt:
git add -A && git commit -m "style(go): apply gofmt"
```

---

### Task 26: Repo-level docs note

**Files:**
- Modify: `AGENTS.md` (at repo root)

- [ ] **Step 1: Append a new section to `AGENTS.md`**

Add (or create) this section at the bottom of `AGENTS.md`:

```markdown
## Go rewrite (v9)

The Go rewrite lives in `go/`. Sub-project E (routing/intent layer) is the first slice — see `docs/superpowers/plans/2026-05-11-routing-intent-layer.md` for the implementation plan and `docs/superpowers/specs/2026-05-11-routing-intent-layer-design.md` for the design. All other sub-projects (A IRC, B LLM, C tools, D overlay, F persistence, G deploy) ship as separate specs/plans.

Build/test: `cd go && make all`.
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: note Go rewrite v9 sub-project E"
```

---

## Out of scope for this plan (deferred to sibling sub-projects)

- **Sub-project A** — Real IRC client (ergochat/irc-go), CAP negotiation, SASL, multi-network, reconnect. Includes the **per-channel turn-lock dispatcher** that takes the `LastAmbientAt` snapshot, calls `router.Route`, atomically commits the ambient timestamp, and invokes `Executor.Run`. The spec's atomicity requirement lives at the dispatch boundary.
- **Sub-project B** — Real LLM client (openai-go + Gemini OpenAI-compat for the hot path; `google.golang.org/genai` for `CachedContent` lifecycle). Provider abstraction for tool-call gaps. Emits typed errors (`ErrLLMTransient`, `ErrLLMFatal`, `ErrCacheStale`, `ErrBudgetExceeded`) that the executor can categorize for recovery.
- **Sub-project C** — Real tool implementations and rate-bucket dispatcher with per-tool timeout enforcement. C is responsible for canonical schema bytes.
- **Sub-project D** — Real overlay resolver. Includes the property test asserting `Get(scope)` is byte-stable across N calls and N goroutines.
- **Sub-project F** — Real persistence (SQLite or similar) for history, memories, usage.
- **Sub-project G** — Single-binary build, config loader, logging sink, optional Prometheus metrics, deploy scripts.
- **Error-category recovery in executor** — Currently every step returns wrapped errors with `status=fatal_fail` recorded. Once B emits typed errors, a follow-up adds: retry-with-backoff for transient (3 tries, 1s/3s/9s), inline cache rebuild for stale, user-visible apology for fatal. Deferred because it can't be meaningfully tested against the current fakes.
- **Tier 3 integration** (real ergochat against local Ergo IRCd) — sub-project A.
- **Tier 4 live-Gemini smoke** — sub-project B.
- **Admin `@cmd` dispatcher** — separate spec.
- **Mid-message mention detection** — future iteration per spec.
