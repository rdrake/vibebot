package fake_test

import (
	"context"
	"testing"
	"time"

	"github.com/rdrake/vibebot/v9/persist"
	"github.com/rdrake/vibebot/v9/persist/fake"
)

func TestAppendAndHistory(t *testing.T) {
	s := fake.New()
	ctx := context.Background()
	if err := s.AppendTurn(ctx, "n", "#x", "alice", "hi alice"); err != nil {
		t.Fatal(err)
	}
	got, err := s.History(ctx, "n", "#x", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("len: %d", len(got))
	}
}

func TestRecordUsage(t *testing.T) {
	s := fake.New()
	if err := s.RecordUsage(context.Background(), persist.UsageRow{Timestamp: time.Now(), Status: "success"}); err != nil {
		t.Fatal(err)
	}
	if len(s.Usage) != 1 {
		t.Fatal("not recorded")
	}
}
