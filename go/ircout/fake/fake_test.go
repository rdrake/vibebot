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
