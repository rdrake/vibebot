// Package fake provides an in-memory ircout.Sender for tests.
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
