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
