"""
contexts-mcp: Multi-environment session state management for Claude.

Provides:
- Session memory with semantic search (ss/rs workflow)
- Environment-scoped routing to MCP relays
- Context bootstrapping per environment
- Token-based concurrent session support

Session workflow:
- POST /ss - Save session state (task, context, blockers, next_steps)
- POST /rs - Restore session (most recent or semantic search)
- GET /bootstrap/{env} - Get environment context for new sessions
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsNullCondition,
    MatchValue,
    PayloadField,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="contexts-mcp",
    description="Multi-environment session state management for Claude",
)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or environment variable."""
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "/app/config.yaml")

    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Fallback to environment-based config for backwards compatibility
    return {
        "default_environment": os.getenv("DEFAULT_ENV", "development"),
        "environments": {},
        "qdrant": {
            "url": os.getenv("QDRANT_URL", "http://qdrant:6333"),
            "collection": os.getenv("QDRANT_COLLECTION", "session-memory"),
        },
        "embedding": {
            "model": os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
        },
    }


# Load configuration
config = load_config()

# Environment configuration from config file
ENVIRONMENTS = {}
for env_name, env_config in config.get("environments", {}).items():
    ENVIRONMENTS[env_name] = {
        "url": env_config.get("url"),
        "description": env_config.get("description", ""),
    }

# Qdrant configuration
QDRANT_URL = config.get("qdrant", {}).get("url", "http://qdrant:6333")
qdrant_collections = config.get("qdrant", {}).get("collections", {})
QDRANT_SESSIONS = qdrant_collections.get(
    "sessions", config.get("qdrant", {}).get("collection", "session-memory")
)
QDRANT_CONTEXT = qdrant_collections.get("context", "global-context")
EMBEDDING_MODEL = config.get("embedding", {}).get(
    "model", "sentence-transformers/all-MiniLM-L6-v2"
)

# Default environment
current_env: str = config.get("default_environment", "development")

# Auto-context configuration
auto_context_config = config.get("auto_context", {})
AUTO_CONTEXT_ENABLED = auto_context_config.get("enabled", True)
THRESHOLD_WARNING = auto_context_config.get("thresholds", {}).get("warning", 70)
THRESHOLD_CRITICAL = auto_context_config.get("thresholds", {}).get("critical", 85)
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[SentenceTransformer] = None

# Session mapping: meta-relay session_id -> backend session info
# Used for SSE connection tracking
session_map: dict[str, dict] = {}

# Token-based routing: token -> {env, created_at}
# Tokens are long-lived and persist across SSE connections
sessions: dict[str, dict] = {}


class SessionState(BaseModel):
    environment: str
    task: str
    context: Optional[str] = None
    blockers: Optional[str] = None
    next_steps: Optional[str] = None
    tags: Optional[list[str]] = None


class AutoSaveRequest(BaseModel):
    """Request for automatic context-triggered save."""
    environment: str
    used_tokens: int
    max_tokens: int = 200000
    inferred_task: Optional[str] = None
    inferred_context: Optional[str] = None


class GitCommitHook(BaseModel):
    """Git post-commit hook payload."""
    environment: str
    commit_sha: str
    commit_message: str
    branch: str
    files_changed: int = 0


class TestResultHook(BaseModel):
    """Test result hook payload."""
    environment: str
    passed: bool
    test_count: int
    failed_count: int = 0
    duration_seconds: float = 0.0
    test_command: Optional[str] = None


class RestoreQuery(BaseModel):
    query: Optional[str] = None
    environment: Optional[str] = None
    limit: int = 5


class ContextEntry(BaseModel):
    """A piece of context/knowledge to store."""

    environment: Optional[str] = None  # None = global across all envs
    category: str  # e.g., "infrastructure", "patterns", "hosts", "procedures"
    title: str
    content: str
    tags: Optional[list[str]] = None


@app.on_event("startup")
async def startup():
    global qdrant_client, embedding_model

    logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        # Create sessions collection if needed
        if QDRANT_SESSIONS not in collection_names:
            logger.info(f"Creating collection {QDRANT_SESSIONS}")
            qdrant_client.create_collection(
                collection_name=QDRANT_SESSIONS,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        # Create context collection if needed
        if QDRANT_CONTEXT not in collection_names:
            logger.info(f"Creating collection {QDRANT_CONTEXT}")
            qdrant_client.create_collection(
                collection_name=QDRANT_CONTEXT,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    except Exception as e:
        logger.warning(f"Qdrant not available: {e}")
        qdrant_client = None

    try:
        logger.info(f"Loading embedding model {EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        logger.warning(f"Embedding model not available: {e}")
        embedding_model = None

    logger.info(f"meta-relay started, default environment: {current_env}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "current_env": current_env,
        "environments": list(ENVIRONMENTS.keys()),
        "active_tokens": len(sessions),
    }


@app.get("/env")
async def get_env():
    return {
        "current": current_env,
        "url": ENVIRONMENTS[current_env]["url"],
        "description": ENVIRONMENTS[current_env]["description"],
        "available": list(ENVIRONMENTS.keys()),
    }


@app.post("/env/{name}")
async def set_env(name: str):
    """
    Set the global default environment.

    DEPRECATED: Use POST /session/{env} to create a token-based session instead.
    This endpoint is kept for backwards compatibility but will be removed in a future version.
    """
    global current_env
    if name not in ENVIRONMENTS:
        raise HTTPException(status_code=400, detail=f"Unknown environment: {name}")
    old_env = current_env
    current_env = name
    logger.info(f"Environment switched: {old_env} -> {current_env}")
    return {
        "previous": old_env,
        "current": current_env,
        "url": ENVIRONMENTS[current_env]["url"],
        "deprecated": True,
        "warning": "This endpoint is deprecated. Use POST /session/{env} to create a token-based session instead.",
    }


# --- Session Token Management ---


@app.post("/session/{env}")
async def create_session_token(env: str):
    """
    Create a new session token bound to an environment.

    Tokens are long-lived and can be reused across multiple SSE connections.
    Use the returned token in /mcp/sse?token=... and /mcp/messages?token=...
    """
    if env not in ENVIRONMENTS:
        raise HTTPException(status_code=400, detail=f"Unknown environment: {env}")

    token = str(uuid.uuid4())
    sessions[token] = {
        "env": env,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(f"Session token created: {token[:8]}... -> {env}")

    return {
        "token": token,
        "environment": env,
        "url": ENVIRONMENTS[env]["url"],
        "description": ENVIRONMENTS[env]["description"],
    }


@app.get("/session/{token}")
async def get_session_token(token: str):
    """
    Get information about a session token.
    """
    if token not in sessions:
        raise HTTPException(status_code=404, detail="Token not found")

    session = sessions[token]
    env = session["env"]

    return {
        "token": token,
        "environment": env,
        "url": ENVIRONMENTS[env]["url"],
        "description": ENVIRONMENTS[env]["description"],
        "created_at": session["created_at"],
    }


@app.delete("/session/{token}")
async def delete_session_token(token: str):
    """
    Invalidate a session token.
    """
    if token not in sessions:
        raise HTTPException(status_code=404, detail="Token not found")

    session = sessions.pop(token)
    logger.info(f"Session token invalidated: {token[:8]}... (was {session['env']})")

    return {
        "status": "deleted",
        "token": token,
        "environment": session["env"],
    }


@app.get("/sessions/tokens")
async def list_session_tokens():
    """
    List all active session tokens.
    """
    return {
        "count": len(sessions),
        "tokens": [
            {
                "token": token,
                "environment": info["env"],
                "created_at": info["created_at"],
            }
            for token, info in sessions.items()
        ],
    }


# --- Helper function for environment resolution ---


def resolve_environment(token: Optional[str]) -> tuple[str, str]:
    """
    Resolve which environment to use based on token.

    Returns (env_name, backend_url).
    Falls back to current_env if no token or token invalid.
    """
    if token and token in sessions:
        env = sessions[token]["env"]
        return env, ENVIRONMENTS[env]["url"]

    # Fallback to global current_env for backwards compatibility
    return current_env, ENVIRONMENTS[current_env]["url"]


@app.get("/mcp/sse")
async def mcp_sse(request: Request, token: Optional[str] = Query(None)):
    """
    MCP SSE endpoint that establishes a session and provides the messages endpoint.

    The backend relay-mcp uses a custom (non-MCP) SSE format, so we don't proxy it.
    Instead, we implement proper MCP SSE transport ourselves.

    Token-based routing:
    - If token is provided and valid, route to that token's environment
    - If no token, fall back to current_env (backwards compatibility)
    """
    import asyncio

    # Resolve environment from token or fallback
    env, backend_url = resolve_environment(token)

    # Generate a meta-relay session ID for this SSE connection
    meta_session_id = str(uuid.uuid4())

    # Store session info for later message routing
    session_map[meta_session_id] = {
        "env": env,
        "backend_url": backend_url,
        "token": token,  # Track which token created this session
    }

    logger.info(
        f"SSE session created: {meta_session_id} -> {env} (token: {token[:8] + '...' if token else 'none'})"
    )

    # Build the messages endpoint URL, including token if provided
    messages_endpoint = f"/mcp/messages?session_id={meta_session_id}"
    if token:
        messages_endpoint += f"&token={token}"

    async def stream_response():
        # Send the endpoint event immediately - this tells the client where to POST messages
        endpoint_event = f"event: endpoint\ndata: {messages_endpoint}\n\n"
        yield endpoint_event.encode("utf-8")

        # Keep connection alive with periodic pings
        # The client will use POST /mcp/messages for actual communication
        try:
            while True:
                await asyncio.sleep(30)
                # Send a comment as keepalive (SSE spec allows comments starting with :)
                yield b": keepalive\n\n"
        except asyncio.CancelledError:
            logger.info(f"SSE session closed: {meta_session_id}")
            # Clean up session on disconnect
            session_map.pop(meta_session_id, None)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Meta-Relay-Env": env,
            "X-Meta-Relay-Session": meta_session_id,
            "X-Meta-Relay-Token": token or "",
        },
    )


@app.post("/mcp/messages/")
@app.post("/mcp/messages")
async def mcp_messages(
    request: Request,
    session_id: Optional[str] = Query(None),
    token: Optional[str] = Query(None),
):
    """
    Translate MCP JSON-RPC protocol to relay-mcp's custom API.

    mcp-remote sends: {"jsonrpc": "2.0", "method": "tools/call", "params": {...}}
    relay-mcp expects: POST /mcp/tools/call with {"name": "...", "arguments": {...}}

    Routing priority:
    1. If token provided and valid, use token's environment
    2. Else if session_id provided and in session_map, use that session's environment
    3. Else fall back to current_env
    """
    body = await request.body()

    try:
        rpc_request = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    method = rpc_request.get("method", "")
    params = rpc_request.get("params", {})
    rpc_id = rpc_request.get("id")

    # Determine backend URL with routing priority
    if token and token in sessions:
        env = sessions[token]["env"]
        backend_url = ENVIRONMENTS[env]["url"]
    elif session_id and session_id in session_map:
        session_info = session_map[session_id]
        env = session_info["env"]
        backend_url = session_info["backend_url"]
    else:
        env = current_env
        backend_url = ENVIRONMENTS[current_env]["url"]

    logger.info(
        f"MCP request: method={method}, session={session_id}, token={token[:8] + '...' if token else 'none'}, env={env}"
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if method == "initialize":
                # Return MCP capabilities directly
                result = {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {
                            "name": f"meta-relay ({env})",
                            "version": "1.0.0",
                        },
                    },
                }

            elif method == "notifications/initialized":
                # Acknowledge initialization
                result = {"jsonrpc": "2.0", "id": rpc_id, "result": {}}

            elif method == "tools/list":
                # Fetch tools from relay-mcp's /api/tools
                response = await client.get(f"{backend_url}/api/tools")
                if response.status_code == 200:
                    tools_data = response.json()
                    # Convert to MCP format
                    mcp_tools = []
                    for tool in tools_data:
                        mcp_tools.append(
                            {
                                "name": tool.get("name"),
                                "description": tool.get("description", ""),
                                "inputSchema": tool.get(
                                    "inputSchema",
                                    tool.get(
                                        "input_schema",
                                        {"type": "object", "properties": {}},
                                    ),
                                ),
                            }
                        )
                    result = {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {"tools": mcp_tools},
                    }
                else:
                    result = {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "error": {
                            "code": -32000,
                            "message": f"Backend error: {response.status_code}",
                        },
                    }

            elif method == "tools/call":
                # Translate to relay-mcp's /mcp/tools/call
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})

                logger.info(f"Calling tool: {tool_name} with args: {arguments}")

                response = await client.post(
                    f"{backend_url}/mcp/tools/call",
                    json={"name": tool_name, "arguments": arguments},
                    timeout=120.0,
                )

                if response.status_code == 200:
                    relay_result = response.json()
                    # relay-mcp returns {"content": [...]} format
                    result = {"jsonrpc": "2.0", "id": rpc_id, "result": relay_result}
                else:
                    error_detail = response.text
                    logger.error(f"Tool call failed: {error_detail}")
                    result = {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "error": {"code": -32000, "message": error_detail},
                    }

            else:
                result = {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except httpx.RequestError as e:
            logger.error(f"Backend request error: {e}")
            result = {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32000, "message": str(e)},
            }

    return JSONResponse(content=result, headers={"X-Meta-Relay-Env": env})


# --- Auto Context Management ---


def get_context_status(percent: float) -> str:
    """Get status label for context usage percentage."""
    if percent < 50:
        return "healthy"
    elif percent < THRESHOLD_WARNING:
        return "ok"
    elif percent < THRESHOLD_CRITICAL:
        return "warning"
    else:
        return "critical"


def get_recommended_action(percent: float) -> dict:
    """Get recommended action based on context usage."""
    if percent < THRESHOLD_WARNING:
        return {"type": "none"}
    elif percent < THRESHOLD_CRITICAL:
        return {
            "type": "save",
            "message": f"Context at {percent:.0f}%. Consider saving session.",
            "auto": False,
        }
    else:
        return {
            "type": "save_and_restart",
            "message": f"Context at {percent:.0f}%. Save and restart recommended.",
            "auto": True,
        }


@app.get("/context/usage")
async def get_context_usage(
    used_tokens: Optional[int] = Query(None),
    max_tokens: int = Query(200000),
    token: Optional[str] = Query(None),
):
    """
    Report context window usage and get recommendations.

    Called by Claude Code hooks to monitor context and trigger auto-saves.

    Args:
        used_tokens: Current token count (from Claude Code)
        max_tokens: Maximum context window size (default 200k)
        token: Session token for environment resolution
    """
    if used_tokens is None:
        return {
            "error": "used_tokens parameter required",
            "hint": "Pass current token count from Claude Code",
            "thresholds": {
                "warning": THRESHOLD_WARNING,
                "critical": THRESHOLD_CRITICAL,
            },
        }

    percent = (used_tokens / max_tokens) * 100
    status = get_context_status(percent)
    action = get_recommended_action(percent)

    # Resolve environment from token if provided
    env = None
    if token and token in sessions:
        env = sessions[token]["env"]

    return {
        "used": used_tokens,
        "max": max_tokens,
        "percent": round(percent, 1),
        "status": status,
        "action": action,
        "environment": env,
        "thresholds": {
            "warning": THRESHOLD_WARNING,
            "critical": THRESHOLD_CRITICAL,
        },
    }


@app.post("/context/auto-save")
async def auto_save_context(req: AutoSaveRequest):
    """
    Automatic save triggered when approaching context limit.

    Called by hooks when context usage exceeds critical threshold.
    Saves session state and returns restart instructions.
    """
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    percent = (req.used_tokens / req.max_tokens) * 100

    # Build session state with auto-generated metadata
    task = req.inferred_task or f"[Auto-save at {percent:.0f}% context]"
    context = req.inferred_context or f"Automatic save at {req.used_tokens:,}/{req.max_tokens:,} tokens ({percent:.0f}%)"

    state = SessionState(
        environment=req.environment,
        task=task,
        context=context,
        next_steps="Continue from auto-save point",
        tags=["auto-save", f"tokens-{req.used_tokens}"],
    )

    # Reuse existing save logic
    text_parts = [f"Environment: {state.environment}", f"Task: {state.task}"]
    if state.context:
        text_parts.append(f"Context: {state.context}")
    if state.next_steps:
        text_parts.append(f"Next steps: {state.next_steps}")

    text = "\n".join(text_parts)
    embedding = embedding_model.encode(text).tolist()
    point_id = int(datetime.now(timezone.utc).timestamp() * 1000)

    qdrant_client.upsert(
        collection_name=QDRANT_SESSIONS,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "environment": state.environment,
                    "task": state.task,
                    "context": state.context,
                    "blockers": None,
                    "next_steps": state.next_steps,
                    "tags": state.tags,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                    "auto_save": True,
                    "tokens_at_save": req.used_tokens,
                },
            )
        ],
    )

    logger.info(f"Auto-saved session at {percent:.0f}% context: {task[:50]}...")

    return {
        "status": "saved",
        "session_id": point_id,
        "percent_at_save": round(percent, 1),
        "tokens_at_save": req.used_tokens,
        "restart_command": f"rs {req.environment}",
        "message": f"Session auto-saved. Run 'rs {req.environment}' to continue.",
    }


# --- Breakpoint Hooks ---


@app.post("/hooks/git-commit")
async def handle_git_commit(hook: GitCommitHook):
    """
    Git post-commit hook handler.

    Saves session state tied to the commit for easy restoration.
    Enables "return to commit X" workflow.
    """
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    # Build descriptive task from commit info
    commit_short = hook.commit_sha[:8] if len(hook.commit_sha) >= 8 else hook.commit_sha
    task = f"Commit: {hook.commit_message[:80]}"
    context = f"Branch: {hook.branch}, SHA: {commit_short}, Files changed: {hook.files_changed}"
    tags = ["git-commit", hook.branch, commit_short]

    state = SessionState(
        environment=hook.environment,
        task=task,
        context=context,
        tags=tags,
    )

    text_parts = [f"Environment: {state.environment}", f"Task: {state.task}"]
    if state.context:
        text_parts.append(f"Context: {state.context}")

    text = "\n".join(text_parts)
    embedding = embedding_model.encode(text).tolist()
    point_id = int(datetime.now(timezone.utc).timestamp() * 1000)

    qdrant_client.upsert(
        collection_name=QDRANT_SESSIONS,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "environment": state.environment,
                    "task": state.task,
                    "context": state.context,
                    "blockers": None,
                    "next_steps": None,
                    "tags": state.tags,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                    "git_commit": True,
                    "commit_sha": hook.commit_sha,
                    "branch": hook.branch,
                },
            )
        ],
    )

    logger.info(f"Saved session at commit {commit_short}: {hook.commit_message[:50]}...")

    return {
        "status": "saved",
        "session_id": point_id,
        "commit": commit_short,
        "branch": hook.branch,
    }


@app.post("/hooks/test-result")
async def handle_test_result(hook: TestResultHook):
    """
    Test result hook handler.

    Saves session on test completion - both success and failure.
    Success = natural breakpoint, failure = debugging context preserved.
    """
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    if hook.passed:
        task = f"Tests passed: {hook.test_count} tests in {hook.duration_seconds:.1f}s"
        tags = ["tests-passed", f"count-{hook.test_count}"]
    else:
        task = f"Tests failed: {hook.failed_count}/{hook.test_count} failed"
        tags = ["tests-failed", f"failed-{hook.failed_count}"]

    context = f"Command: {hook.test_command}" if hook.test_command else None

    state = SessionState(
        environment=hook.environment,
        task=task,
        context=context,
        tags=tags,
    )

    text_parts = [f"Environment: {state.environment}", f"Task: {state.task}"]
    if state.context:
        text_parts.append(f"Context: {state.context}")

    text = "\n".join(text_parts)
    embedding = embedding_model.encode(text).tolist()
    point_id = int(datetime.now(timezone.utc).timestamp() * 1000)

    qdrant_client.upsert(
        collection_name=QDRANT_SESSIONS,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "environment": state.environment,
                    "task": state.task,
                    "context": state.context,
                    "blockers": None,
                    "next_steps": None,
                    "tags": state.tags,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                    "test_result": True,
                    "tests_passed": hook.passed,
                    "test_count": hook.test_count,
                    "failed_count": hook.failed_count,
                },
            )
        ],
    )

    status_word = "passed" if hook.passed else "failed"
    logger.info(f"Saved session at test {status_word}: {task}")

    return {
        "status": "saved",
        "session_id": point_id,
        "tests_passed": hook.passed,
        "test_count": hook.test_count,
        "failed_count": hook.failed_count,
    }


# --- Session State Management ---


@app.post("/ss")
async def save_session(state: SessionState):
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    text_parts = [f"Environment: {state.environment}", f"Task: {state.task}"]
    if state.context:
        text_parts.append(f"Context: {state.context}")
    if state.blockers:
        text_parts.append(f"Blockers: {state.blockers}")
    if state.next_steps:
        text_parts.append(f"Next steps: {state.next_steps}")

    text = "\n".join(text_parts)
    embedding = embedding_model.encode(text).tolist()
    point_id = int(datetime.now(timezone.utc).timestamp() * 1000)

    qdrant_client.upsert(
        collection_name=QDRANT_SESSIONS,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "environment": state.environment,
                    "task": state.task,
                    "context": state.context,
                    "blockers": state.blockers,
                    "next_steps": state.next_steps,
                    "tags": state.tags or [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                },
            )
        ],
    )

    logger.info(f"Saved session state: {state.task[:50]}...")
    return {
        "status": "saved",
        "id": point_id,
        "environment": state.environment,
        "task": state.task,
    }


@app.post("/rs")
async def restore_session(query: RestoreQuery):
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    search_filter = None
    if query.environment:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="environment", match=MatchValue(value=query.environment)
                )
            ]
        )

    # If no query provided, get most recent by timestamp (not vector similarity)
    if not query.query:
        results, _ = qdrant_client.scroll(
            collection_name=QDRANT_SESSIONS,
            scroll_filter=search_filter,
            limit=query.limit * 3,  # Fetch more to sort properly
            with_payload=True,
            with_vectors=False,
        )

        sessions_result = sorted(
            [
                {
                    "id": r.id,
                    "score": None,
                    "environment": r.payload.get("environment"),
                    "task": r.payload.get("task"),
                    "context": r.payload.get("context"),
                    "blockers": r.payload.get("blockers"),
                    "next_steps": r.payload.get("next_steps"),
                    "tags": r.payload.get("tags"),
                    "timestamp": r.payload.get("timestamp"),
                }
                for r in results
            ],
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )[: query.limit]

        return {
            "query": None,
            "environment_filter": query.environment,
            "count": len(sessions_result),
            "sessions": sessions_result,
        }

    # With a query, use vector similarity search
    search_text = query.query
    embedding = embedding_model.encode(search_text).tolist()

    results = qdrant_client.query_points(
        collection_name=QDRANT_SESSIONS,
        query=embedding,
        query_filter=search_filter,
        limit=query.limit,
    ).points

    sessions_result = [
        {
            "id": r.id,
            "score": r.score,
            "environment": r.payload.get("environment"),
            "task": r.payload.get("task"),
            "context": r.payload.get("context"),
            "blockers": r.payload.get("blockers"),
            "next_steps": r.payload.get("next_steps"),
            "tags": r.payload.get("tags"),
            "timestamp": r.payload.get("timestamp"),
        }
        for r in results
    ]

    return {
        "query": search_text,
        "environment_filter": query.environment,
        "count": len(sessions_result),
        "sessions": sessions_result,
    }


@app.get("/sessions")
async def list_sessions(environment: Optional[str] = None, limit: int = 20):
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    search_filter = None
    if environment:
        search_filter = Filter(
            must=[
                FieldCondition(key="environment", match=MatchValue(value=environment))
            ]
        )

    results, _ = qdrant_client.scroll(
        collection_name=QDRANT_SESSIONS,
        scroll_filter=search_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    sessions_result = sorted(
        [
            {
                "id": r.id,
                "environment": r.payload.get("environment"),
                "task": r.payload.get("task"),
                "timestamp": r.payload.get("timestamp"),
                "tags": r.payload.get("tags"),
            }
            for r in results
        ],
        key=lambda x: x.get("timestamp", ""),
        reverse=True,
    )

    return {
        "environment_filter": environment,
        "count": len(sessions_result),
        "sessions": sessions_result,
    }


# --- Context Memory Endpoints ---


@app.post("/context")
async def save_context(entry: ContextEntry):
    """
    Save a piece of context/knowledge to long-term memory.

    Context is searchable and can be scoped to an environment or global.
    Categories help organize knowledge (infrastructure, patterns, hosts, procedures).
    """
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    # Build searchable text
    text_parts = [f"Title: {entry.title}", f"Category: {entry.category}"]
    if entry.environment:
        text_parts.append(f"Environment: {entry.environment}")
    text_parts.append(f"Content: {entry.content}")

    text = "\n".join(text_parts)
    embedding = embedding_model.encode(text).tolist()
    point_id = int(datetime.now(timezone.utc).timestamp() * 1000)

    qdrant_client.upsert(
        collection_name=QDRANT_CONTEXT,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "environment": entry.environment,
                    "category": entry.category,
                    "title": entry.title,
                    "content": entry.content,
                    "tags": entry.tags or [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                },
            )
        ],
    )

    logger.info(f"Saved context: [{entry.category}] {entry.title}")
    return {
        "status": "saved",
        "id": point_id,
        "category": entry.category,
        "title": entry.title,
        "environment": entry.environment,
    }


@app.get("/context")
async def search_context(
    query: str,
    environment: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
):
    """
    Search context/knowledge by semantic similarity.

    Optionally filter by environment and/or category.
    """
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    embedding = embedding_model.encode(query).tolist()

    # Build filter conditions
    must_conditions = []
    if environment:
        # Match specific env OR global (null environment)
        must_conditions.append(
            FieldCondition(key="environment", match=MatchValue(value=environment))
        )
    if category:
        must_conditions.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )

    search_filter = Filter(must=must_conditions) if must_conditions else None

    results = qdrant_client.query_points(
        collection_name=QDRANT_CONTEXT,
        query=embedding,
        query_filter=search_filter,
        limit=limit,
    ).points

    return {
        "query": query,
        "environment": environment,
        "category": category,
        "count": len(results),
        "results": [
            {
                "id": r.id,
                "score": r.score,
                "category": r.payload.get("category"),
                "title": r.payload.get("title"),
                "content": r.payload.get("content"),
                "environment": r.payload.get("environment"),
                "tags": r.payload.get("tags"),
                "timestamp": r.payload.get("timestamp"),
            }
            for r in results
        ],
    }


@app.delete("/context/{point_id}")
async def delete_context(point_id: int):
    """Delete a specific context entry by ID."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    qdrant_client.delete(
        collection_name=QDRANT_CONTEXT,
        points_selector=[point_id],
    )

    return {"status": "deleted", "id": point_id}


@app.get("/bootstrap/{env}")
async def get_bootstrap(env: str):
    """
    Get minimal bootstrap context for a Claude session.

    Returns:
    - Environment config from YAML (url, description, context settings)
    - Top relevant context entries from qdrant for this environment
    """
    if env not in ENVIRONMENTS:
        raise HTTPException(status_code=404, detail=f"Unknown environment: {env}")

    env_config = ENVIRONMENTS[env]

    # Get environment-specific context from config
    context_config = config.get("environments", {}).get(env, {}).get("context", {})

    # Fetch relevant context from qdrant (env-specific + global)
    context_entries = []
    if qdrant_client and embedding_model:
        # Search for context relevant to this environment
        search_text = f"environment {env} infrastructure setup"
        embedding = embedding_model.encode(search_text).tolist()

        # Get env-specific context
        try:
            env_results = qdrant_client.query_points(
                collection_name=QDRANT_CONTEXT,
                query=embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="environment", match=MatchValue(value=env))
                    ]
                ),
                limit=5,
            ).points

            # Get global context (environment=null)
            global_results = qdrant_client.query_points(
                collection_name=QDRANT_CONTEXT,
                query=embedding,
                query_filter=Filter(
                    must=[IsNullCondition(is_null=PayloadField(key="environment"))]
                ),
                limit=5,
            ).points

            for r in env_results + global_results:
                context_entries.append(
                    {
                        "category": r.payload.get("category"),
                        "title": r.payload.get("title"),
                        "content": r.payload.get("content"),
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to fetch context: {e}")

    return {
        "environment": env,
        "url": env_config["url"],
        "description": env_config["description"],
        "context": {
            "config": context_config,
            "knowledge": context_entries,
        },
    }


@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to show active SSE session mappings and tokens."""
    return {
        "active_sse_sessions": len(session_map),
        "active_tokens": len(sessions),
        "sse_sessions": {
            sid: {
                "env": info["env"],
                "token": info.get("token", "none"),
            }
            for sid, info in session_map.items()
        },
        "tokens": {
            token: {
                "env": info["env"],
                "created_at": info["created_at"],
            }
            for token, info in sessions.items()
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
