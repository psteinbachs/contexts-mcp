#!/bin/bash
# Context monitor hook for Claude Code
# Checks token usage and triggers auto-save when critical
#
# Usage: context-monitor.sh <used_tokens> [max_tokens]
#
# Environment variables:
#   CONTEXTS_ENV - Active environment (default: development)
#   CONTEXTS_URL - contexts-mcp URL (default: http://localhost:8100)

set -euo pipefail

ENV="${CONTEXTS_ENV:-development}"
CONTEXTS_URL="${CONTEXTS_URL:-http://localhost:8100}"

USED_TOKENS="${1:-}"
MAX_TOKENS="${2:-200000}"

if [ -z "$USED_TOKENS" ]; then
    echo "Usage: context-monitor.sh <used_tokens> [max_tokens]"
    echo ""
    echo "Environment variables:"
    echo "  CONTEXTS_ENV - Active environment (default: development)"
    echo "  CONTEXTS_URL - contexts-mcp URL (default: http://localhost:8100)"
    exit 1
fi

# Query context usage endpoint
RESPONSE=$(curl -s "$CONTEXTS_URL/context/usage?used_tokens=$USED_TOKENS&max_tokens=$MAX_TOKENS" 2>/dev/null || echo '{"error": "connection failed"}')

# Check for errors
if echo "$RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
    echo "Error: $(echo "$RESPONSE" | jq -r '.error')"
    exit 1
fi

# Get action type
ACTION=$(echo "$RESPONSE" | jq -r '.action.type')
PERCENT=$(echo "$RESPONSE" | jq -r '.percent')
STATUS=$(echo "$RESPONSE" | jq -r '.status')

case "$ACTION" in
    "save_and_restart")
        echo "CRITICAL: Context at ${PERCENT}% (${USED_TOKENS}/${MAX_TOKENS} tokens)"
        echo "Auto-saving session..."

        SAVE_RESPONSE=$(curl -s -X POST "$CONTEXTS_URL/context/auto-save" \
            -H "Content-Type: application/json" \
            -d "{
                \"environment\": \"$ENV\",
                \"used_tokens\": $USED_TOKENS,
                \"max_tokens\": $MAX_TOKENS
            }" 2>/dev/null)

        if echo "$SAVE_RESPONSE" | jq -e '.status == "saved"' > /dev/null 2>&1; then
            SESSION_ID=$(echo "$SAVE_RESPONSE" | jq -r '.session_id')
            echo "Session saved (ID: $SESSION_ID)"
            echo ""
            echo "Recommended: Start new session with 'rs $ENV'"
        else
            echo "Auto-save failed: $SAVE_RESPONSE"
            exit 1
        fi
        ;;

    "save")
        echo "WARNING: Context at ${PERCENT}% (${USED_TOKENS}/${MAX_TOKENS} tokens)"
        echo "Consider saving session with 'ss'"
        ;;

    "none")
        echo "OK: Context at ${PERCENT}% (status: $STATUS)"
        ;;

    *)
        echo "Unknown action: $ACTION"
        exit 1
        ;;
esac
