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
