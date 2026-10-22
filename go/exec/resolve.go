package exec

import (
	"fmt"

	"github.com/rdrake/vibebot/v9/llmcore"
	"github.com/rdrake/vibebot/v9/tooling"
)

// ResolveSchemas converts tool names into ToolSchema via the dispatcher.
// Caller-side resolution means B never needs a back-channel to C.
func ResolveSchemas(d tooling.Dispatcher, names []string) ([]llmcore.ToolSchema, error) {
	out := make([]llmcore.ToolSchema, 0, len(names))
	for _, n := range names {
		b, err := d.SchemaJSON(n)
		if err != nil {
			return nil, fmt.Errorf("schema for %q: %w", n, err)
		}
		out = append(out, llmcore.ToolSchema{Name: n, SchemaJSON: b})
	}
	return out, nil
}
