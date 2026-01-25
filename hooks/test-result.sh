#!/bin/bash
# Test result hook - saves session state after test runs
#
# Usage: test-result.sh <passed|failed> <test_count> [failed_count] [duration_seconds] [command]
#
# Examples:
#   test-result.sh passed 42 0 12.5 "pytest tests/"
#   test-result.sh failed 42 3 15.2 "npm test"
#
# Environment variables:
#   CONTEXTS_ENV - Active environment (required)
#   CONTEXTS_URL - contexts-mcp URL (default: http://localhost:8100)

set -euo pipefail

ENV="${CONTEXTS_ENV:-}"
if [ -z "$ENV" ]; then
    echo "Error: CONTEXTS_ENV not set"
    exit 1
fi

CONTEXTS_URL="${CONTEXTS_URL:-http://localhost:8100}"

RESULT="${1:-}"
TEST_COUNT="${2:-0}"
FAILED_COUNT="${3:-0}"
DURATION="${4:-0}"
COMMAND="${5:-}"

if [ -z "$RESULT" ] || [ -z "$TEST_COUNT" ]; then
    echo "Usage: test-result.sh <passed|failed> <test_count> [failed_count] [duration] [command]"
    exit 1
fi

# Convert result to boolean
if [ "$RESULT" = "passed" ] || [ "$RESULT" = "true" ] || [ "$RESULT" = "1" ]; then
    PASSED="true"
else
    PASSED="false"
fi

# Build JSON payload
PAYLOAD="{
    \"environment\": \"$ENV\",
    \"passed\": $PASSED,
    \"test_count\": $TEST_COUNT,
    \"failed_count\": $FAILED_COUNT,
    \"duration_seconds\": $DURATION"

if [ -n "$COMMAND" ]; then
    PAYLOAD="$PAYLOAD, \"test_command\": \"$COMMAND\""
fi

PAYLOAD="$PAYLOAD}"

# Send to contexts-mcp
RESPONSE=$(curl -s -X POST "$CONTEXTS_URL/hooks/test-result" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" 2>/dev/null)

if echo "$RESPONSE" | jq -e '.status == "saved"' > /dev/null 2>&1; then
    SESSION_ID=$(echo "$RESPONSE" | jq -r '.session_id')
    if [ "$PASSED" = "true" ]; then
        echo "Session saved: $TEST_COUNT tests passed"
    else
        echo "Session saved: $FAILED_COUNT/$TEST_COUNT tests failed"
    fi
else
    echo "Warning: Failed to save session"
fi
