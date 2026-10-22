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
