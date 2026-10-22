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
