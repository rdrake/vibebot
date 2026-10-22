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
