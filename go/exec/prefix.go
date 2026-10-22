package exec

import (
	"bytes"
	"fmt"
	"sort"

	"github.com/rdrake/vibebot/v9/llmcore"
)

// BuildPrefixArgs is the input to BuildCachedPrefix.
// Tools must already be resolved schemas (see ResolveSchemas). The schemas'
// bytes must be canonical per sub-project C's contract; this function does
// NOT re-canonicalize JSON keys or whitespace.
type BuildPrefixArgs struct {
	Framework string
	Overlay   string
	Tools     []llmcore.ToolSchema
	Network   string
	Channel   string
	Profile   string
}

// BuildCachedPrefix assembles the four-block cacheable prefix per the spec
// section "Cache prefix composition". Tools are sorted by name before
// serialization to make output insensitive to caller-side ordering.
//
// Use llmcore.Canonical(cp) to get the cacheable byte block.
func BuildCachedPrefix(args BuildPrefixArgs) llmcore.CachedPrefix {
	tools := append([]llmcore.ToolSchema(nil), args.Tools...)
	sort.Slice(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })

	var buf bytes.Buffer
	buf.WriteByte('[')
	for i, t := range tools {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.Write(t.SchemaJSON)
	}
	buf.WriteByte(']')

	return llmcore.CachedPrefix{
		FrameworkPrompt: args.Framework,
		Overlay:         args.Overlay,
		ToolSchemasJSON: buf.Bytes(),
		ChannelContext:  fmt.Sprintf("Network: %s\nChannel: %s\nProfile: %s\n", args.Network, args.Channel, args.Profile),
	}
}
