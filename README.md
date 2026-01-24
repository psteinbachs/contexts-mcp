# mcp-contexts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Multi-environment session state management for Claude.

## What it does

Save your working context, close Claude, come back days later, and pick up exactly where you left off - with full semantic search across your session history.

- **Session memory** - `ss` saves task, context, blockers, next_steps to qdrant
- **Session restore** - `rs <env>` gets most recent, or `rs <env> "query"` searches semantically
- **Multi-environment** - Route to different MCP relays (dev, prod, staging, etc.)
- **Concurrent sessions** - Token-based routing so multiple Claude instances can run independently
- **Bootstrap context** - `/bootstrap/{env}` provides environment-specific context for new sessions

## Quick Start

### One-liner install

```bash
curl -fsSL https://raw.githubusercontent.com/psteinbachs/mcp-contexts/main/setup/install.sh | bash
```

### Or manual deploy

```bash
git clone https://github.com/psteinbachs/mcp-contexts.git
cd mcp-contexts
cp config.example.yaml config.yaml
# Edit config.yaml with your environments
docker compose up -d
```

### 2. Configure your environments

Each environment points to an MCP relay (or any MCP server) and can have custom context:

```yaml
environments:
  dev:
    url: http://mcp-relay:8000        # Your MCP server
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
      omega: true                      # Triggers extra caution in Claude
```

### 3. Add session commands to your CLAUDE.md

```markdown
## Session Restore (rs)

**`rs`** - Prompts for environment, then restores most recent session
**`rs <env>`** - Loads environment and restores most recent session  
**`rs <env> "<query>"`** - Loads environment and searches for specific session

### When user types `rs <env>`:
1. Read `~/.claude/env/<env>.md` for bootstrap context (optional)
2. Restore most recent session:
   ```bash
   curl -s -X POST http://localhost:8100/rs \
     -H "Content-Type: application/json" \
     -d '{"environment": "<env>", "limit": 1}'
   ```
3. Display session context and confirm ready to continue

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
# Start a new Claude session
rs dev                      # Restore most recent dev session
                            # Claude now knows what you were doing

# ... do work ...

# Before closing Claude
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

Store persistent knowledge that outlives individual sessions:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/context` | POST | Store knowledge |
| `/context` | GET | Search knowledge |
| `/context/{id}` | DELETE | Remove knowledge |
| `/bootstrap/{env}` | GET | Get environment context for new sessions |

**Store context:**
```bash
curl -X POST http://localhost:8100/context \
  -H "Content-Type: application/json" \
  -d '{
    "environment": "dev",
    "category": "architecture",
    "title": "API authentication flow",
    "content": "We use JWT tokens stored in httpOnly cookies..."
  }'
```

### MCP Proxy

Routes MCP calls to the active environment's relay:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/session/{env}` | POST | Create environment-bound token |
| `/mcp/sse` | GET | SSE proxy to environment's relay |
| `/mcp/messages` | POST | MCP message proxy |
| `/health` | GET | Health check |

## Configuration Reference

```yaml
# config.yaml
default_environment: dev

environments:
  dev:
    url: http://mcp-relay:8000
    description: Development environment
    context:
      networks:
        allowed: [192.168.0.0/16]    # Networks Claude can access
        forbidden: []                 # Networks to warn about
      omega: false                    # true = extra caution warnings

  prod:
    url: https://mcp.prod.example.com
    description: Production infrastructure
    context:
      networks:
        allowed: [10.0.0.0/8]
        forbidden: [192.168.0.0/16]
      omega: true

qdrant:
  url: http://qdrant:6333
  collections:
    sessions: session-memory    # Session storage
    context: global-context     # Long-term knowledge

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
```

## Deployment

```yaml
# docker-compose.yml
services:
  mcp-contexts:
    build: .
    container_name: mcp-contexts
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

## How it works

1. **Sessions are vectors** - When you save a session, the task/context/next_steps are embedded and stored in Qdrant
2. **Restore by time or meaning** - No query = most recent by timestamp. With query = semantic similarity search
3. **Environments isolate context** - Each environment has its own session history
4. **MCP routing** - Token-based routing lets multiple Claude instances work against different environments simultaneously
