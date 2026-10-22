package router

import "github.com/rdrake/vibebot/v9/router/profile"

func decideAction(evt IRCEvent, state ChannelState, bot BotState) Action {
	if addressed(evt, bot.SelfNick, bot.RecentSentIDs) {
		switch {
		case state.LoomActive != nil:
			return RespondLoom
		case state.SceneActive != nil:
			return RespondScene
		default:
			return RespondChat
		}
	}
	if !state.AmbientEnabled {
		return Ignore
	}
	key := ChannelKey{Network: evt.Network, Channel: evt.Channel}
	if evt.ReceivedAt.Sub(bot.LastAmbientAt[key]) < state.AmbientCooldown {
		return Ignore
	}
	return RespondChat
}

// Route is the pure decision function. Same inputs → same RouteDecision.
//
// The dispatcher is responsible for ambient claim atomicity (turn lock +
// compare-and-set on LastAmbientAt) before invoking Executor.Run. Route
// reads the snapshot it was given and does not mutate anything.
func Route(evt IRCEvent, state ChannelState, bot BotState, reg *profile.Registry) RouteDecision {
	profileName := state.Profile
	p, ok := reg.Get(profileName)
	if !ok {
		profileName = "quiet"
		p, ok = reg.Get(profileName)
		if !ok {
			return RouteDecision{Action: Ignore, Event: evt}
		}
	}

	action := decideAction(evt, state, bot)
	if action == Ignore {
		return RouteDecision{Action: Ignore, Event: evt, Profile: profileName}
	}

	scope := buildCacheScope(evt.Network, evt.Channel, profileName, state.Overlay)
	return RouteDecision{
		Action:     action,
		Profile:    profileName,
		Tools:      p.Tools, // already a defensive copy from registry.Get
		CacheScope: scope,
		Model:      p.Model,
		Prompt: PromptSpec{
			HistoryLimit: 20,
			UserText:     evt.Text,
			UserNick:     evt.Nick,
		},
		Delivery: DeliverySpec{
			Typing:               true,
			ChunkSize:            380,
			ReplyToID:            evt.MessageID,
			NickPrefixOnFallback: evt.Channel != "",
			MaxIters:             p.MaxIters,
		},
		Event: evt,
	}
}
