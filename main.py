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

import asyncio
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
    Direction,
    Distance,
    FieldCondition,
    Filter,
    IsNullCondition,
    MatchValue,
    OrderBy,
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
# Each session includes an asyncio.Queue for sending responses via SSE
session_map: dict[str, dict] = {}

# Lock for thread-safe session_map access
session_map_lock = asyncio.Lock()

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
    key_artifacts: Optional[list[str]] = None  # Important files/docs from this session


class AutoSaveRequest(BaseModel):
    """Request for automatic context-triggered save."""

    environment: str
    used_tokens: int
    max_tokens: int = 200000
    inferred_task: Optional[str] = None
    inferred_context: Optional[str] = None
    transcript_content: Optional[str] = None  # JSONL transcript from Claude Code


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
    include_auto_save: bool = False  # Exclude auto-saves by default


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

        # Ensure payload index exists for timestamp ordering
        try:
            qdrant_client.create_payload_index(
                collection_name=QDRANT_SESSIONS,
                field_name="timestamp",
                field_schema="datetime",  # Range index for ISO timestamps
            )
            logger.info(f"Created timestamp datetime index on {QDRANT_SESSIONS}")
        except Exception as idx_err:
            # Index might already exist
            logger.debug(f"Timestamp index creation: {idx_err}")

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
async def get_env(token: Optional[str] = Query(None)):
    """
    Get current environment info.

    If token is provided and valid, returns that token's environment.
    Otherwise falls back to global default (for backwards compatibility).
    """
    env, url = resolve_environment(token)
    return {
        "current": env,
        "url": url,
        "description": ENVIRONMENTS[env]["description"],
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

    Per MCP SSE spec: responses to POST /mcp/messages are sent back through
    this SSE stream as 'message' events, NOT as HTTP POST responses.

    Token-based routing:
    - If token is provided and valid, route to that token's environment
    - If no token, fall back to current_env (backwards compatibility)
    """
    # Resolve environment from token or fallback
    env, backend_url = resolve_environment(token)

    # Generate a meta-relay session ID for this SSE connection
    meta_session_id = str(uuid.uuid4())

    # Create a queue for this session to receive response messages
    response_queue: asyncio.Queue = asyncio.Queue()

    # Store session info for later message routing
    session_map[meta_session_id] = {
        "env": env,
        "backend_url": backend_url,
        "token": token,  # Track which token created this session
        "queue": response_queue,  # Queue for SSE responses
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

        # Now loop: check for queued messages and send keepalives
        # Per MCP SSE spec, responses come through as 'message' events
        try:
            while True:
                try:
                    # Wait for a message with timeout (for keepalive)
                    message = await asyncio.wait_for(response_queue.get(), timeout=30.0)
                    # Send the response as an SSE message event
                    message_json = json.dumps(message)
                    message_event = f"event: message\ndata: {message_json}\n\n"
                    logger.debug(f"SSE sending message to {meta_session_id}: {message_json[:100]}...")
                    yield message_event.encode("utf-8")
                except asyncio.TimeoutError:
                    # No message received, send keepalive
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

    Per MCP SSE spec: responses are sent back through the SSE stream as 'message'
    events, NOT returned directly from this endpoint. This endpoint returns 202
    Accepted and queues the response for the SSE stream.

    Routing priority:
    1. If token provided and valid, use token's environment
    2. Else if session_id provided and in session_map, use that session's environment
    3. Else fall back to current_env
    """
    global current_env
    body = await request.body()

    try:
        rpc_request = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    method = rpc_request.get("method", "")
    params = rpc_request.get("params", {})
    rpc_id = rpc_request.get("id")

    # Get response queue for this session (if exists)
    response_queue = None
    session_info = None
    if session_id and session_id in session_map:
        session_info = session_map[session_id]
        response_queue = session_info.get("queue")

    # Determine backend URL with routing priority
    if token and token in sessions:
        env = sessions[token]["env"]
        backend_url = ENVIRONMENTS[env]["url"]
    elif session_info:
        env = session_info["env"]
        backend_url = session_info["backend_url"]
    else:
        env = current_env
        backend_url = ENVIRONMENTS[current_env]["url"]

    logger.info(
        f"MCP request: method={method}, session={session_id}, token={token[:8] + '...' if token else 'none'}, env={env}, has_queue={response_queue is not None}"
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
                # Acknowledge initialization - this is a notification, no response needed
                # But we still send an empty result for compatibility
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
                    # Add contexts-mcp's own meta-tools
                    mcp_tools.append(
                        {
                            "name": "mcp__contexts__set_environment",
                            "description": "Switch to a different environment. This changes which MCP tools are available. Use this when you need to work with a different infrastructure environment.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "environment": {
                                        "type": "string",
                                        "description": f"Environment name. Available: {', '.join(ENVIRONMENTS.keys())}",
                                    }
                                },
                                "required": ["environment"],
                            },
                        }
                    )
                    mcp_tools.append(
                        {
                            "name": "mcp__contexts__get_environment",
                            "description": "Get the current environment name and available environments.",
                            "inputSchema": {"type": "object", "properties": {}},
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

                # Handle contexts-mcp's own meta-tools
                if tool_name == "mcp__contexts__set_environment":
                    new_env = arguments.get("environment", "")
                    if new_env not in ENVIRONMENTS:
                        result = {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Unknown environment: {new_env}. Available: {', '.join(ENVIRONMENTS.keys())}",
                                    }
                                ],
                                "isError": True,
                            },
                        }
                    else:
                        old_env = env
                        # Update session mapping for this SSE connection
                        if session_id and session_id in session_map:
                            session_map[session_id]["env"] = new_env
                            session_map[session_id]["backend_url"] = ENVIRONMENTS[
                                new_env
                            ]["url"]
                        current_env = new_env
                        logger.info(
                            f"Environment switched: {old_env} -> {new_env} (session: {session_id})"
                        )
                        result = {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Switched from '{old_env}' to '{new_env}'. Backend: {ENVIRONMENTS[new_env]['url']}. Note: Run tools/list again to see the new environment's tools.",
                                    }
                                ],
                            },
                        }

                elif tool_name == "mcp__contexts__get_environment":
                    result = {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Current environment: {env}\nBackend: {backend_url}\nAvailable: {', '.join(ENVIRONMENTS.keys())}",
                                }
                            ],
                        },
                    }

                else:
                    # Proxy to backend relay-mcp
                    response = await client.post(
                        f"{backend_url}/mcp/tools/call",
                        json={"name": tool_name, "arguments": arguments},
                        timeout=120.0,
                    )

                    if response.status_code == 200:
                        relay_result = response.json()
                        # relay-mcp returns {"content": [...]} format
                        result = {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": relay_result,
                        }
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

    # Per MCP SSE spec: send response through SSE stream, not HTTP response
    if response_queue is not None:
        # Queue the response for the SSE stream
        await response_queue.put(result)
        logger.debug(f"Queued response for session {session_id}: {str(result)[:100]}...")
        # Return 202 Accepted - the actual response goes via SSE
        return JSONResponse(
            content={"status": "accepted", "message": "Response will be sent via SSE"},
            status_code=202,
            headers={"X-Meta-Relay-Env": env},
        )
    else:
        # No SSE session - fall back to direct response (for testing/debugging)
        logger.warning(f"No SSE session for {session_id}, returning direct response")
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


def extract_session_context(transcript_content: str) -> dict:
    """
    Extract meaningful task/context/next_steps from Claude Code transcript.

    Parses JSONL transcript and extracts:
    - Recent user requests (task)
    - Key actions taken (context)
    - Ongoing work indicators (next_steps)

    Returns dict with task, context, next_steps keys.
    """
    lines = transcript_content.strip().split("\n")

    user_messages = []
    assistant_summaries = []
    tool_calls = []
    files_modified = set()

    # Parse recent transcript entries (last ~50 lines to stay focused)
    recent_lines = lines[-50:] if len(lines) > 50 else lines

    for line in recent_lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = entry.get("type")

        if msg_type == "human" or entry.get("role") == "user":
            # User message
            content = entry.get("content") or entry.get("message", "")
            if isinstance(content, str) and len(content) > 10:
                # Skip very short messages like "yes", "ok", etc.
                user_messages.append(content[:500])

        elif msg_type == "assistant" or entry.get("role") == "assistant":
            # Assistant message - look for tool use
            content = entry.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            tool_calls.append(tool_name)
                            # Track file modifications
                            tool_input = block.get("input", {})
                            if isinstance(tool_input, dict):
                                file_path = tool_input.get("file_path") or tool_input.get("path")
                                if file_path and tool_name in ("Edit", "Write"):
                                    files_modified.add(file_path)
                        elif block.get("type") == "text":
                            text = block.get("text", "")
                            if len(text) > 50:
                                assistant_summaries.append(text[:300])
            elif isinstance(content, str) and len(content) > 50:
                assistant_summaries.append(content[:300])

    # Build task from recent user messages
    if user_messages:
        # Use the most recent substantive user message as the task
        task = user_messages[-1]
        if len(task) > 200:
            task = task[:200] + "..."
    else:
        task = None

    # Build context from tool calls and files modified
    context_parts = []
    if tool_calls:
        # Count tool usage
        tool_counts = {}
        for t in tool_calls:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:5]
        context_parts.append(f"Tools used: {', '.join(f'{t}({c})' for t, c in top_tools)}")

    if files_modified:
        context_parts.append(f"Files modified: {', '.join(sorted(files_modified)[:10])}")

    if assistant_summaries:
        # Use the last assistant summary as additional context
        context_parts.append(f"Recent work: {assistant_summaries[-1][:200]}")

    context = " | ".join(context_parts) if context_parts else None

    # Infer next steps from context
    next_steps = None
    if files_modified:
        next_steps = f"Continue working on: {', '.join(sorted(files_modified)[:5])}"
    elif user_messages:
        next_steps = "Continue from last user request"

    return {
        "task": task,
        "context": context,
        "next_steps": next_steps,
    }


@app.post("/context/auto-save")
async def auto_save_context(req: AutoSaveRequest):
    """
    Automatic save triggered when approaching context limit.

    Called by hooks when context usage exceeds critical threshold.
    Saves session state and returns restart instructions.

    If transcript_content is provided, extracts meaningful context from it.
    """
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    percent = (req.used_tokens / req.max_tokens) * 100

    # Extract context from transcript if provided
    extracted = {}
    if req.transcript_content:
        try:
            extracted = extract_session_context(req.transcript_content)
            logger.info(f"Extracted context from transcript: task={extracted.get('task', '')[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to extract context from transcript: {e}")

    # Build session state - prefer extracted over inferred over defaults
    task = (
        extracted.get("task")
        or req.inferred_task
        or f"[Auto-save at {percent:.0f}% context]"
    )
    context = (
        extracted.get("context")
        or req.inferred_context
        or f"Automatic save at {req.used_tokens:,}/{req.max_tokens:,} tokens ({percent:.0f}%)"
    )

    next_steps = extracted.get("next_steps") or "Continue from auto-save point"

    state = SessionState(
        environment=req.environment,
        task=task,
        context=context,
        next_steps=next_steps,
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
        "task": task,  # Include extracted task for display
        "context": context,
        "next_steps": next_steps,
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

    logger.info(
        f"Saved session at commit {commit_short}: {hook.commit_message[:50]}..."
    )

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
                    "key_artifacts": state.key_artifacts or [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                    "auto_save": False,  # Explicit: manual save
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
        "key_artifacts": state.key_artifacts or [],
    }


@app.post("/rs")
async def restore_session(query: RestoreQuery):
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    # Build filter conditions
    must_conditions = []
    must_not_conditions = []

    if query.environment:
        must_conditions.append(
            FieldCondition(
                key="environment", match=MatchValue(value=query.environment)
            )
        )
    # Exclude auto-saves by default (auto_save=True records)
    if not query.include_auto_save:
        must_not_conditions.append(
            FieldCondition(
                key="auto_save", match=MatchValue(value=True)
            )
        )

    search_filter = None
    if must_conditions or must_not_conditions:
        search_filter = Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None,
        )

    # If no query provided, get most recent by timestamp (not vector similarity)
    if not query.query:
        results, _ = qdrant_client.scroll(
            collection_name=QDRANT_SESSIONS,
            scroll_filter=search_filter,
            limit=query.limit,
            order_by=OrderBy(key="timestamp", direction=Direction.DESC),
            with_payload=True,
            with_vectors=False,
        )

        sessions_result = [
            {
                "id": r.id,
                "score": None,
                "environment": r.payload.get("environment"),
                "task": r.payload.get("task"),
                "context": r.payload.get("context"),
                "blockers": r.payload.get("blockers"),
                "next_steps": r.payload.get("next_steps"),
                "tags": r.payload.get("tags"),
                "key_artifacts": r.payload.get("key_artifacts"),
                "timestamp": r.payload.get("timestamp"),
            }
            for r in results
        ]

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
            "key_artifacts": r.payload.get("key_artifacts"),
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
        order_by=OrderBy(key="timestamp", direction=Direction.DESC),
        with_payload=True,
        with_vectors=False,
    )

    sessions_result = [
        {
            "id": r.id,
            "environment": r.payload.get("environment"),
            "task": r.payload.get("task"),
            "timestamp": r.payload.get("timestamp"),
            "tags": r.payload.get("tags"),
        }
        for r in results
    ]

    return {
        "environment_filter": environment,
        "count": len(sessions_result),
        "sessions": sessions_result,
    }


@app.delete("/sessions/auto-saves")
async def purge_auto_saves(environment: Optional[str] = None, dry_run: bool = True):
    """
    Delete all auto-save sessions.

    Use dry_run=true (default) to preview what would be deleted.
    Set dry_run=false to actually delete.
    """
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not initialized")

    # Build filter for auto-saves
    must_conditions = [
        FieldCondition(key="auto_save", match=MatchValue(value=True))
    ]
    if environment:
        must_conditions.append(
            FieldCondition(key="environment", match=MatchValue(value=environment))
        )

    search_filter = Filter(must=must_conditions)

    # First, find all matching auto-saves
    results, _ = qdrant_client.scroll(
        collection_name=QDRANT_SESSIONS,
        scroll_filter=search_filter,
        limit=1000,  # Get up to 1000 auto-saves
        with_payload=True,
        with_vectors=False,
    )

    point_ids = [r.id for r in results]

    if dry_run:
        return {
            "dry_run": True,
            "would_delete": len(point_ids),
            "environment_filter": environment,
            "sample_tasks": [r.payload.get("task", "")[:50] for r in results[:5]],
            "hint": "Set dry_run=false to actually delete",
        }

    # Actually delete
    if point_ids:
        qdrant_client.delete(
            collection_name=QDRANT_SESSIONS,
            points_selector=point_ids,
        )

    logger.info(f"Purged {len(point_ids)} auto-save sessions (env={environment})")

    return {
        "dry_run": False,
        "deleted": len(point_ids),
        "environment_filter": environment,
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
    - MCP servers registered with the backend relay (tools summary)
    """
    if env not in ENVIRONMENTS:
        raise HTTPException(status_code=404, detail=f"Unknown environment: {env}")

    env_config = ENVIRONMENTS[env]
    backend_url = env_config["url"]

    # Get environment-specific context from config
    context_config = config.get("environments", {}).get(env, {}).get("context", {})

    # Fetch MCP servers from backend relay-mcp
    mcp_servers = []
    total_tools = 0
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{backend_url}/api/servers")
            if response.status_code == 200:
                servers_data = response.json()
                for server in servers_data:
                    if server.get("enabled", True):
                        tools_count = server.get("tools_count", 0)
                        total_tools += tools_count
                        mcp_servers.append({
                            "name": server.get("name"),
                            "tools": tools_count,
                            "status": server.get("status", "unknown"),
                        })
        except Exception as e:
            logger.warning(f"Failed to fetch MCP servers from {backend_url}: {e}")

    # Fetch global context from qdrant (facts tagged "global" for this environment)
    context_entries = []
    if qdrant_client:
        try:
            # Get facts tagged "global" for this environment
            global_results = qdrant_client.scroll(
                collection_name=QDRANT_CONTEXT,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="environment", match=MatchValue(value=env)),
                        FieldCondition(key="tags", match=MatchValue(value="global")),
                    ]
                ),
                limit=10,
            )[0]

            for r in global_results:
                context_entries.append(
                    {
                        "category": r.payload.get("category"),
                        "title": r.payload.get("title"),
                        "content": r.payload.get("content"),
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to fetch global context: {e}")

    # Extract critical directive for top-level prominence
    critical_directive = context_config.get("critical_directive")

    return {
        "environment": env,
        "url": env_config["url"],
        "description": env_config["description"],
        "critical_directive": critical_directive,
        "mcp_servers": {
            "total_tools": total_tools,
            "servers": mcp_servers,
        },
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
