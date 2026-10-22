// Package fake provides an in-memory tooling.Dispatcher for tests.
package fake

import (
	"context"
	"errors"
	"sync"

	"github.com/rdrake/vibebot/v9/tooling"
)

type Dispatcher struct {
	Results         map[string]tooling.ToolResult
	Schemas         map[string][]byte
	PanicOnDispatch bool
	PanicMessage    string

	mu    sync.Mutex
	Calls []tooling.ToolCall
}

func New() *Dispatcher {
	return &Dispatcher{
		Results: map[string]tooling.ToolResult{},
		Schemas: map[string][]byte{},
	}
}

func (d *Dispatcher) SchemaJSON(name string) ([]byte, error) {
	s, ok := d.Schemas[name]
	if !ok {
		return nil, errors.New("fake: no schema for " + name)
	}
	return s, nil
}

func (d *Dispatcher) Dispatch(ctx context.Context, call tooling.ToolCall) tooling.ToolResult {
	if d.PanicOnDispatch {
		msg := d.PanicMessage
		if msg == "" {
			msg = "fake panic"
		}
		panic(msg)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.Calls = append(d.Calls, call)
	r, ok := d.Results[call.Name]
	if !ok {
		return tooling.ToolResult{Err: errors.New("fake: no result for " + call.Name)}
	}
	return r
}
