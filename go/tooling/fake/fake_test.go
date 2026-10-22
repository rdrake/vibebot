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
