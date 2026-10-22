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
