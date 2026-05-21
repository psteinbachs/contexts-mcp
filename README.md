# contexts-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Multi-environment session state management for AI coding assistants.

Tested in production with **Claude Code** and **opencode**; works with
any assistant that reads a project-instructions file and can shell out
to `curl`.

## What it does

Save your working context, close the assistant, come back days later,
and pick up exactly where you left off — with full semantic search
across your session history.

- **Session memory** - `ss` saves task, context, blockers, next_steps to qdrant
- **Session restore** - `rs <env>` gets most recent, or `rs <env> "query"` searches semantically
- **Multi-environment** - Route to different MCP relays (dev, prod, staging, etc.)
- **Concurrent sessions** - Token-based routing so multiple assistant instances can run independently
- **Bootstrap context** - `/bootstrap/{env}` provides environment-specific context for new sessions

## Quick Start

### One-liner install

```bash
curl -fsSL https://raw.githubusercontent.com/psteinbachs/contexts-mcp/main/setup/install.sh | bash
```

### Or manual deploy

```bash
git clone https://github.com/psteinbachs/contexts-mcp.git
cd contexts-mcp
cp config.example.yaml config.yaml
# Edit config.yaml with your environments
docker compose up -d
```

### 2. Configure your environments

Each environment points to an MCP relay (or any MCP server) and can have custom context:

```yaml
environments:
  dev:
    url: http://relay-mcp:8000        # Your MCP server
    description: Development environment
    context:
      networks:
        allowed: [192.168.0.0/16]
      omega: false                     # No special warnings

  prod:
    url: https://mcp.prod.example.com
    description: Production - be careful!
    context:
      networks:
        allowed: [10.0.0.0/8]
        forbidden: [192.168.0.0/16]
      omega: true                      # Surface a critical-directive guard at session start
```

### 3. Add session commands to your assistant's instructions

The runtime contract is identical across assistants — they just read
the recipe out of different files:

| Assistant      | Instructions file           |
|----------------|-----------------------------|
| Claude Code    | `CLAUDE.md` (project) / `~/.claude/CLAUDE.md` (global) |
| opencode       | `AGENTS.md`                 |
| Codex / Cursor | `AGENTS.md`                 |

Drop this block into whichever file your assistant reads:

```markdown
## Session Restore (rs)

**`rs`** - Prompts for environment, then restores most recent session
**`rs <env>`** - Loads environment and restores most recent session
**`rs <env> "<query>"`** - Loads environment and searches for specific session

### When user types `rs <env>`:
1. Restore most recent session:
   ```bash
   curl -s -X POST http://localhost:8100/rs \
     -H "Content-Type: application/json" \
     -d '{"environment": "<env>", "limit": 1}'
   ```
2. Display session context and confirm ready to continue

## Session Save (ss)

**`ss`** - Save current session

### When user types `ss`:
1. Save session:
   ```bash
   curl -s -X POST http://localhost:8100/ss \
     -H "Content-Type: application/json" \
     -d '{"environment": "<env>", "task": "<current task>", "context": "<relevant details>", "next_steps": "<what comes next>"}'
   ```
```

Any assistant that can execute shell commands and pattern-match on
user input can drive `ss` / `rs` from this contract.

### 4. First session - nothing to restore yet

```bash
# First time? There's no session to restore, just start working
# When done, save your first session:

curl -X POST http://localhost:8100/ss \
  -H "Content-Type: application/json" \
  -d '{
    "environment": "dev",
    "task": "Setting up the new auth service",
    "context": "Created user model, added JWT middleware",
    "next_steps": "Wire up login endpoint, add tests"
  }'
```

## The Workflow

Once you have sessions saved:

```bash
# Start a new assistant session
rs dev                      # Restore most recent dev session
                            # The assistant now knows what you were doing

# ... do work ...

# Before closing the assistant
ss                          # Saves current task/context/next_steps

# Days later, can't remember where you left off?
rs dev "auth bug"           # Semantic search across all dev sessions
```

## API Reference

### Session Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ss` | POST | Save session state |
| `/rs` | POST | Restore session (most recent or semantic search) |
| `/sessions` | GET | List recent sessions |

**Save session:**
```bash
curl -X POST http://localhost:8100/ss \
  -H "Content-Type: application/json" \
  -d '{
    "environment": "dev",
    "task": "What you were working on",
    "context": "Relevant details, decisions made",
    "blockers": "Optional - what was blocking you",
    "next_steps": "What needs to happen next"
  }'
```

**Restore most recent:**
```bash
curl -X POST http://localhost:8100/rs \
  -H "Content-Type: application/json" \
  -d '{"environment": "dev", "limit": 1}'
```

**Semantic search:**
```bash
curl -X POST http://localhost:8100/rs \
  -H "Content-Type: application/json" \
  -d '{"environment": "dev", "query": "database migration issue", "limit": 5}'
```

### Context & Knowledge

Long-form context entries (knowledge that outlives an individual
session) live in the same Qdrant collection as before, but `contexts-mcp`
no longer exposes write or arbitrary-query HTTP endpoints for them.
Clients write and search the collection directly with the Qdrant SDK,
and `contexts-mcp` reads the collection on the bootstrap path to render
priorities into session-start responses.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/bootstrap/{env}` | GET | Environment context (config + priorities surfaced from the collection) |
| `/full-context/{env}` | POST | One-shot bootstrap: issues a token and fans out MCP servers, priorities, and last session in parallel |

The previous `POST /context`, `GET /context`, and `DELETE /context/{id}`
endpoints were thin HTTP wrappers over basic Qdrant CRUD. Clients are
expected to embed (e.g. with `fastembed`) and upsert directly against
their Qdrant instance — the collection name lives in
`qdrant.collections.context` in `config.yaml` (default `global-context`,
384-dim Cosine to match `sentence-transformers/all-MiniLM-L6-v2`).

**Bootstrap a new session (one-shot):**
```bash
curl -X POST http://localhost:8100/full-context/dev
```
Response includes a fresh session token, the environment's MCP servers,
config-level critical directive, high/critical-urgency priorities, the most
recent saved session, and a pre-rendered `summary` object for easy display.

### MCP Proxy

Routes MCP calls to the active environment's relay:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/session/{env}` | POST | Create environment-bound token |
| `/session/{token}` | GET / DELETE | Inspect or revoke a token |
| `/sessions/tokens` | GET | List active tokens |
| `/env` / `/env/{name}` | GET / POST | Read or set the active environment |
| `/mcp/sse` | GET | SSE proxy to environment's relay |
| `/mcp/messages` | POST | MCP message proxy |
| `/health` | GET | Health check |

## Configuration Reference

```yaml
# config.yaml
default_environment: dev

environments:
  dev:
    url: http://relay-mcp:8000
    description: Development environment
    auth:                              # optional — per-env credential profile
      type: oauth
      profile: personal                # credential profile loaded before the assistant starts
    statusline:                        # optional — colors used by hooks
      bg_rgb: "76;86;106"
      icon: ""
    context:
      networks:
        allowed: [192.168.0.0/16]      # Networks the assistant is expected to reach
        forbidden: []                   # Networks to warn about
      omega: false                      # true = surface a critical-directive guard at session start

  prod:
    url: https://mcp.prod.example.com
    description: Production infrastructure
    auth:
      type: oauth
      profile: work
    statusline:
      bg_rgb: "191;97;106"             # red for danger
      icon: ""
    context:
      networks:
        allowed: [10.0.0.0/8]
        forbidden: [192.168.0.0/16]
      omega: true
      # critical_directive surfaces prominently in /full-context responses
      # and is intended for "pain of death" rules that must be followed.
      critical_directive: "PRODUCTION DATA MUST BE PRESERVED. Before any operation: identify affected data, verify backups, know rollback path."

qdrant:
  url: http://qdrant:6333
  collections:
    sessions: session-memory    # Session storage
    context: global-context     # Long-term knowledge

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
```

See `config.example.yaml` for the full annotated example, including
`api_key` auth (for CI/headless use) and the `auto_context` thresholds.

## Deployment

```yaml
# docker-compose.yml
services:
  contexts-mcp:
    build: .
    container_name: contexts-mcp
    restart: unless-stopped
    ports:
      - "8100:8000"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - CONFIG_PATH=/app/config.yaml
    networks:
      - mcp-network

  qdrant:
    image: qdrant/qdrant
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - mcp-network

volumes:
  qdrant_data:

networks:
  mcp-network:
```

## Automatic Context Management

Never hit context overflow again. contexts-mcp can monitor token usage and auto-save before you run out of space.

### Setup

1. **Disable the assistant's own auto-compaction** so contexts-mcp can drive saves on the threshold instead.

   On Claude Code (the setting recovers ~45k tokens):
   ```bash
   claude config set --global autoCompact false
   ```

   On opencode / other assistants: consult their docs for the equivalent
   "do not auto-compact" toggle, if any. The auto-save hooks below
   work without it; you just won't get the token reclaim.

2. **Configure thresholds** in config.yaml:
   ```yaml
   auto_context:
     enabled: true
     thresholds:
       warning: 70    # Suggest saving
       critical: 85   # Auto-save + restart
   ```

3. **Install git hook** (optional - saves on every commit):
   ```bash
   ln -sf /path/to/contexts-mcp/hooks/post-commit .git/hooks/post-commit
   export CONTEXTS_ENV=dev  # Set your environment
   ```

### Auto-Save API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/context/usage` | GET | Check token usage, get recommendations |
| `/context/auto-save` | POST | Emergency save at threshold |
| `/hooks/git-commit` | POST | Save on git commit |
| `/hooks/test-result` | POST | Save on test completion |

**Check context usage:**
```bash
curl "http://localhost:8100/context/usage?used_tokens=150000&max_tokens=200000"
```

Response:
```json
{
  "used": 150000,
  "max": 200000,
  "percent": 75.0,
  "status": "warning",
  "action": {
    "type": "save",
    "message": "Context at 75%. Consider saving session."
  }
}
```

**Auto-save (called by hooks):**
```bash
curl -X POST http://localhost:8100/context/auto-save \
  -H "Content-Type: application/json" \
  -d '{"environment": "dev", "used_tokens": 170000}'
```

### Hook Scripts

Located in `hooks/`:

- **`context-monitor.sh`** - Check context and auto-save if critical
- **`post-commit`** - Git hook to save on commits
- **`test-result.sh`** - Save after test runs

Example git hook output:
```
$ git commit -m "Add auth endpoint"
Session saved at a1b2c3d4
[main a1b2c3d4] Add auth endpoint
 2 files changed, 45 insertions(+)
```

## How it works

1. **Sessions are vectors** - When you save a session, the task/context/next_steps are embedded and stored in Qdrant
2. **Restore by time or meaning** - No query = most recent by timestamp. With query = semantic similarity search
3. **Environments isolate sessions, share globals** - Each environment has its own *session* history. *Knowledge* entries in the context collection are env-scoped by default, but entries stored without an environment are treated as cross-cutting "globals" and surface in every env's bootstrap results.
4. **MCP routing** - Token-based routing lets multiple assistant instances work against different environments simultaneously
5. **One-shot bootstrap** - `/full-context/{env}` issues a token and fans out MCP servers, priorities (read from the context collection), and last session in parallel, returning a single response with a pre-rendered `summary` for display
6. **Auto-save on thresholds** - Token monitoring triggers saves before context overflow
