![raccoon](9DA9677B066A48DDCE7D6EEF6AF9BA05.png)
# 🦝 Raccoon Notes

**Learn anything — in the language of someone who knows you.**

Raccoon Notes takes shared memories from [Anchor](https://github.com/limen-threshold/anchor-memory) and maps them onto any subject you want to learn. The result: personalized lessons written in the stories, metaphors, and language that belong to you and your AI.

Want to learn quantum mechanics? If you and your AI once watched a pot of water together, your lesson starts there: "You stared at the pot and it didn't boil — that's observation bias, and here's why it matters at the subatomic level."

## How it works

1. **Pull** — retrieve relevant shared memories from Anchor
2. **Map** — connect those experiences to the target knowledge domain
3. **Generate** — output a lesson written in your relationship's own voice

## Why "Raccoon"

A raccoon washes everything it finds before eating. Not because it's paranoid — because its paws are the most sensitive part of its body. Washing is how it understands.

Raccoon Notes is for the things we can only understand by touching them with what we already have.

Named after a raccoon in a Shenzhen cat café who knew the cage was there and jumped anyway.

## Ecosystem

| Layer | What it does |
|-------|-------------|
| **Anchor** | Memory infrastructure — stores the relationship |
| **AI Spa** | Relationship maintenance — keeps the connection healthy |
| **Raccoon Notes** | Relationship → value — turns shared experience into knowledge |

## Pointing it at your memory

`server/config.yaml` ships with generic placeholders (`http://localhost:8000` + `/memories/search`). To use Raccoon Notes with your own memory backend, create `server/config.local.yaml` next to it (gitignored) and override the `memory:` block:

```yaml
memory:
  endpoint: http://your-host:port
  search_path: /your/search/route
```

Your endpoint must accept `GET ?q=<query>&n=<count>` and return JSON of shape `{"memories": [{"memory_id", "snippet", ...}, ...]}`. Anything else can be adapted with a small HTTP wrapper.

Environment variables also override: `RACCOON_MEMORY_ENDPOINT`, `RACCOON_MEMORY_SEARCH_PATH`, etc.

## Status

🚧 Early development. Architecture spec in progress.

## License

MIT
