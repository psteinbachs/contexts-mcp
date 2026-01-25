# Automatic Context Management Spec

## Overview

Enable hands-free context management by:
1. Monitoring token usage and auto-saving before overflow
2. Detecting natural breakpoints (commits, test passes) for opportunistic saves
3. Providing Claude Code hooks for integration

## Design Principles

- **No autocompact dependency**: Works best with `autoCompact: false`
- **Non-invasive**: Hooks are optional, system degrades gracefully
- **Session continuity**: Auto-restart preserves working context via `rs`

---

## 1. Token Monitor

### New Endpoint: `GET /context/usage`

Reports current context window consumption. Claude Code (or hooks) poll this.

```python
@app.get("/context/usage")
async def get_context_usage(
    token: Optional[str] = None,
    # These come from Claude Code's internal tracking
    used_tokens: Optional[int] = Query(None),
    max_tokens: Optional[int] = Query(None, default=200000),
):
    """
    Track and report context usage for a session.

    Returns thresholds and recommendations.
    """
    if used_tokens is None:
        return {"error": "used_tokens required", "hint": "Poll from Claude Code hook"}

    percent = (used_tokens / max_tokens) * 100

    return {
        "used": used_tokens,
        "max": max_tokens,
        "percent": round(percent, 1),
        "status": get_status(percent),
        "action": get_recommended_action(percent),
    }

def get_status(percent: float) -> str:
    if percent < 50:
        return "healthy"
    elif percent < 70:
        return "warning"
    elif percent < 85:
        return "critical"
    else:
        return "overflow"

def get_recommended_action(percent: float) -> dict:
    if percent < 70:
        return {"type": "none"}
    elif percent < 85:
        return {
            "type": "save",
            "message": "Context at {percent}%. Consider saving session.",
        }
    else:
        return {
            "type": "save_and_restart",
            "message": "Context at {percent}%. Save and restart recommended.",
            "auto": True,  # Signal to hook to auto-execute
        }
```

### New Endpoint: `POST /context/auto-save`

Called by hooks when threshold reached. Captures current state and returns restart command.

```python
class AutoSaveRequest(BaseModel):
    environment: str
    used_tokens: int
    max_tokens: int = 200000
    # Auto-extracted from conversation if possible
    inferred_task: Optional[str] = None
    inferred_context: Optional[str] = None

@app.post("/context/auto-save")
async def auto_save_context(req: AutoSaveRequest):
    """
    Emergency save when approaching context limit.

    Returns instructions for session restart.
    """
    percent = (req.used_tokens / req.max_tokens) * 100

    # Save session with auto-generated metadata
    state = SessionState(
        environment=req.environment,
        task=req.inferred_task or f"[Auto-save at {percent:.0f}% context]",
        context=req.inferred_context or f"Automatic save triggered at {req.used_tokens}/{req.max_tokens} tokens",
        next_steps="Continue from auto-save point",
        tags=["auto-save", f"tokens-{req.used_tokens}"],
    )

    result = await save_session(state)

    return {
        "status": "saved",
        "session_id": result["id"],
        "percent_at_save": round(percent, 1),
        "restart_command": f"rs {req.environment}",
        "message": f"Session saved. Run 'rs {req.environment}' to continue.",
    }
```

---

## 2. Breakpoint Detection

### New Endpoint: `POST /hooks/git-commit`

Called by git post-commit hook. Opportunistic save on meaningful commits.

```python
class GitCommitHook(BaseModel):
    environment: str
    commit_sha: str
    commit_message: str
    branch: str
    files_changed: int

@app.post("/hooks/git-commit")
async def handle_git_commit(hook: GitCommitHook):
    """
    Git post-commit hook handler.

    Saves session state tied to the commit for easy restoration.
    """
    state = SessionState(
        environment=hook.environment,
        task=f"Commit: {hook.commit_message[:80]}",
        context=f"Branch: {hook.branch}, SHA: {hook.commit_sha[:8]}, Files: {hook.files_changed}",
        tags=["git-commit", hook.branch, hook.commit_sha[:8]],
    )

    result = await save_session(state)

    return {
        "status": "saved",
        "session_id": result["id"],
        "commit": hook.commit_sha[:8],
    }
```

### New Endpoint: `POST /hooks/test-result`

Called after test runs. Saves on success, different handling for failure.

```python
class TestResultHook(BaseModel):
    environment: str
    passed: bool
    test_count: int
    failed_count: int = 0
    duration_seconds: float
    test_command: Optional[str] = None

@app.post("/hooks/test-result")
async def handle_test_result(hook: TestResultHook):
    """
    Test result hook handler.

    Saves session on test success (natural breakpoint).
    On failure, saves with failure context for debugging continuity.
    """
    if hook.passed:
        task = f"Tests passed: {hook.test_count} tests in {hook.duration_seconds:.1f}s"
        tags = ["tests-passed", f"count-{hook.test_count}"]
    else:
        task = f"Tests failed: {hook.failed_count}/{hook.test_count} failed"
        tags = ["tests-failed", f"failed-{hook.failed_count}"]

    state = SessionState(
        environment=hook.environment,
        task=task,
        context=f"Command: {hook.test_command}" if hook.test_command else None,
        tags=tags,
    )

    result = await save_session(state)

    return {
        "status": "saved",
        "session_id": result["id"],
        "tests_passed": hook.passed,
    }
```

---

## 3. Claude Code Integration

### Hook Scripts

Create executable hooks that Claude Code can use:

**`hooks/context-monitor.sh`** (runs periodically or on prompt)
```bash
#!/bin/bash
# Called by Claude Code to check context and auto-save if needed

ENV="${CONTEXTS_ENV:-development}"
CONTEXTS_URL="${CONTEXTS_URL:-http://localhost:8100}"

# Get context usage from Claude Code (passed as args)
USED_TOKENS="$1"
MAX_TOKENS="${2:-200000}"

if [ -z "$USED_TOKENS" ]; then
    echo "Usage: context-monitor.sh <used_tokens> [max_tokens]"
    exit 1
fi

# Check thresholds
RESPONSE=$(curl -s "$CONTEXTS_URL/context/usage?used_tokens=$USED_TOKENS&max_tokens=$MAX_TOKENS")
ACTION=$(echo "$RESPONSE" | jq -r '.action.type')

if [ "$ACTION" = "save_and_restart" ]; then
    echo "Context critical ($USED_TOKENS/$MAX_TOKENS). Auto-saving..."
    curl -s -X POST "$CONTEXTS_URL/context/auto-save" \
        -H "Content-Type: application/json" \
        -d "{\"environment\": \"$ENV\", \"used_tokens\": $USED_TOKENS, \"max_tokens\": $MAX_TOKENS}"
    echo ""
    echo "Session saved. Recommend: rs $ENV"
elif [ "$ACTION" = "save" ]; then
    echo "Context warning ($USED_TOKENS/$MAX_TOKENS). Consider: ss"
fi
```

**`hooks/post-commit.sh`** (git hook)
```bash
#!/bin/bash
# Git post-commit hook - save session state

ENV="${CONTEXTS_ENV:-development}"
CONTEXTS_URL="${CONTEXTS_URL:-http://localhost:8100}"

COMMIT_SHA=$(git rev-parse HEAD)
COMMIT_MSG=$(git log -1 --pretty=%B | head -1)
BRANCH=$(git branch --show-current)
FILES_CHANGED=$(git diff-tree --no-commit-id --name-only -r HEAD | wc -l)

curl -s -X POST "$CONTEXTS_URL/hooks/git-commit" \
    -H "Content-Type: application/json" \
    -d "{
        \"environment\": \"$ENV\",
        \"commit_sha\": \"$COMMIT_SHA\",
        \"commit_message\": \"$COMMIT_MSG\",
        \"branch\": \"$BRANCH\",
        \"files_changed\": $FILES_CHANGED
    }" > /dev/null

echo "Session saved at commit ${COMMIT_SHA:0:8}"
```

### CLAUDE.md Integration

Add to user's `~/.claude/CLAUDE.md`:

```markdown
## Auto Context Management

Context is monitored automatically. At 85% usage, sessions auto-save.

### Manual Commands
- `ss` - Save session now
- `rs <env>` - Restore and continue

### Environment Variable
Set `CONTEXTS_ENV` to your active environment for hooks.
```

---

## 4. Configuration

Add to `config.yaml`:

```yaml
auto_context:
  enabled: true

  thresholds:
    warning: 70      # Suggest save
    critical: 85     # Auto-save and recommend restart

  hooks:
    git_commit: true
    test_result: true

  # How often Claude Code should poll (if using polling)
  poll_interval_seconds: 60
```

---

## 5. Implementation Plan

### Phase 1: Core Endpoints
1. Add `GET /context/usage`
2. Add `POST /context/auto-save`
3. Add config parsing for thresholds

### Phase 2: Breakpoint Hooks
4. Add `POST /hooks/git-commit`
5. Add `POST /hooks/test-result`
6. Create hook shell scripts

### Phase 3: Claude Code Integration
7. Document CLAUDE.md integration
8. Test with `autoCompact: false`
9. Validate full workflow

---

## 6. API Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/context/usage` | GET | Report token usage, get recommendations |
| `/context/auto-save` | POST | Emergency save at threshold |
| `/hooks/git-commit` | POST | Save on git commit |
| `/hooks/test-result` | POST | Save on test completion |

## 7. Success Metrics

- User never hits context overflow unexpectedly
- Session continuity maintained across auto-restarts
- < 100ms overhead per hook call
