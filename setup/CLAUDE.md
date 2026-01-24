# Session Management for Claude

Add this to your CLAUDE.md to enable session save/restore.

---

## Session Restore (rs)

**`rs`** - Prompts for environment, then restores most recent session
**`rs <env>`** - Loads environment and restores most recent session  
**`rs <env> "<query>"`** - Loads environment and searches for specific session

### When user types `rs` (no args):
1. List available environments from config
2. Ask: "Which environment?"
3. Once specified, proceed as `rs <env>`

### When user types `rs <env>`:
1. Restore most recent session:
   ```bash
   curl -s -X POST http://localhost:8100/rs \
     -H "Content-Type: application/json" \
     -d '{"environment": "<env>", "limit": 1}'
   ```
2. Display session context and confirm ready to continue

### When user types `rs <env> "<query>"`:
Same as above, but search with the query:
```bash
curl -s -X POST http://localhost:8100/rs \
  -H "Content-Type: application/json" \
  -d '{"environment": "<env>", "query": "<query>", "limit": 5}'
```

---

## Session Save (ss)

**`ss`** - Save current session

### When user types `ss`:
1. If no environment loaded in this conversation, ask which one
2. Save session:
   ```bash
   curl -s -X POST http://localhost:8100/ss \
     -H "Content-Type: application/json" \
     -d '{
       "environment": "<env>",
       "task": "<what you were doing>",
       "context": "<relevant details>",
       "next_steps": "<what comes next>"
     }'
   ```

---

## Example Session

```
User: rs dev
Claude: [calls /rs with environment=dev, limit=1]
        
        Session restored from 2 hours ago:
        
        Task: Implementing user authentication
        Context: Created JWT middleware, added /login endpoint
        Next steps: Add refresh token support, write tests
        
        Ready to continue. What would you like to work on?

User: [... works on the task ...]

User: ss
Claude: [calls /ss with current task/context/next_steps]
        
        Session saved. You can close this conversation and
        resume later with `rs dev`.
```
