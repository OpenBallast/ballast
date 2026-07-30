# Demo endpoint: mcp.openballast.org

A live, free-to-use instance of the Ballast T0 corpus (levels L0–L5: 4.4M entities,
62.8M triples, 6.7M name aliases) speaking **MCP over streamable HTTP** plus plain
GET endpoints. It exists to demonstrate that the knowledge layer costs nothing to
run — the whole thing is Cloudflare free tier. It is a *demo*: no auth, no SLA,
fair-use rate expectations (~100k requests/day pool). For serious or offline use,
[download the artifact](https://huggingface.co/datasets/OpenBallast/ballast-t0)
and serve it yourself.

## MCP

Endpoint: `https://mcp.openballast.org/mcp`

Works with any MCP client (streamable HTTP transport, no auth). Example client
config:

```json
{
  "mcpServers": {
    "ballast": { "url": "https://mcp.openballast.org/mcp" }
  }
}
```

### Tools

| tool | args | returns |
|---|---|---|
| `resolve` | `name`, `limit=5` | candidate Q-ids for an entity name (normalized label/alias match, most-notable first) |
| `evidence` | `id` (Q-id or name), `level=5`, `max_triples=32` | one entity's facts as a compact evidence block |
| `lookup` | `question`, `level=5`, `max_triples=24` | one-shot: mines entity mentions from the question, resolves each, returns evidence blocks — feed them to your model before answering |

### Level semantics

`level` (0–5) is the corpus quantization knob. Lower levels are smaller corpora
holding only more notable entities. Semantics match the parquet artifact exactly:
an entity outside the level has no evidence; a fact whose *object* entity falls
outside the level is dropped, not rendered as a bare Q-id.

```
level 5:  Facts about Berlin: … head of government: Kai Wegner … country: Germany …
level 0:  Facts about Berlin: … country: Germany …        (Wegner is outside L0)
```

## Plain HTTP

```bash
# whole-question grounding
curl "https://mcp.openballast.org/lookup?question=Who+composed+the+opera+Turandot%3F&level=5"

# one entity, by Q-id or name
curl "https://mcp.openballast.org/evidence?id=Q42&level=3"
curl "https://mcp.openballast.org/evidence?id=Douglas%20Adams&level=5"
```

## Raw JSON-RPC (what MCP does under the hood)

```bash
curl -X POST https://mcp.openballast.org/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"lookup","arguments":{"question":"Where was Douglas Adams born?","level":5}}}'
```

## Recommended prompt pattern

Prepend the evidence blocks, then ask — completion-style for base models:

```
Facts about Douglas Adams:
- place of birth: Cambridge
- occupation: science fiction writer
…

Q: Where was Douglas Adams born?
A:
```

This is the exact prompt shape the published numbers were measured under.
