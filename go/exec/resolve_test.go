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
