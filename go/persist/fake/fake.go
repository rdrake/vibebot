// Package fake provides an in-memory persist.Store for tests.
package fake

import (
	"context"
	"sync"
	"time"

	"github.com/rdrake/vibebot/v9/persist"
)

type Store struct {
	mu             sync.Mutex
	HistoryEntries []persist.HistoryEntry
	MemoryRows     map[memoryKey][]persist.Memory
	Usage          []persist.UsageRow
}

type memoryKey struct{ network, channel, nick string }

func New() *Store { return &Store{MemoryRows: map[memoryKey][]persist.Memory{}} }

func (s *Store) History(ctx context.Context, network, channel string, limit int) ([]persist.HistoryEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := append([]persist.HistoryEntry(nil), s.HistoryEntries...)
	if limit > 0 && len(out) > limit {
		out = out[len(out)-limit:]
	}
	return out, nil
}

func (s *Store) Memories(ctx context.Context, network, channel, nick string) ([]persist.Memory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.MemoryRows[memoryKey{network, channel, nick}], nil
}

func (s *Store) AppendTurn(ctx context.Context, network, channel, nick, assistantText string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.HistoryEntries = append(s.HistoryEntries,
		persist.HistoryEntry{Role: "user", Nick: nick, Content: "(user msg)", Timestamp: time.Now()},
		persist.HistoryEntry{Role: "assistant", Content: assistantText, Timestamp: time.Now()},
	)
	return nil
}

func (s *Store) RecordUsage(ctx context.Context, row persist.UsageRow) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Usage = append(s.Usage, row)
	return nil
}
