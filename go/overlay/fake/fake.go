// Package fake provides an in-memory overlay.Resolver for tests.
package fake

import (
	"context"

	"github.com/rdrake/vibebot/v9/overlay"
)

type Resolver struct {
	Texts map[overlay.Scope]string
	Err   error
}

func New() *Resolver { return &Resolver{Texts: map[overlay.Scope]string{}} }

func (r *Resolver) Get(ctx context.Context, scope overlay.Scope) (string, error) {
	if r.Err != nil {
		return "", r.Err
	}
	return r.Texts[scope], nil
}
