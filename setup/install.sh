#!/bin/bash
set -e

# mcp-contexts installer
# Usage: curl -fsSL https://raw.githubusercontent.com/psteinbachs/mcp-contexts/main/setup/install.sh | bash

REPO_URL="https://github.com/psteinbachs/mcp-contexts.git"
INSTALL_DIR="${MCP_CONTEXTS_DIR:-$HOME/.mcp-contexts}"

echo "┌─────────────────────────────────────────┐"
echo "│  mcp-contexts installer                 │"
echo "│  Multi-environment session management   │"
echo "└─────────────────────────────────────────┘"
echo

# Detect container runtime
if command -v podman &> /dev/null; then
    RUNTIME="podman"
    COMPOSE="podman-compose"
    if ! command -v podman-compose &> /dev/null; then
        COMPOSE="podman compose"
    fi
elif command -v docker &> /dev/null; then
    RUNTIME="docker"
    COMPOSE="docker compose"
else
    echo "Error: Neither docker nor podman found. Please install one first."
    exit 1
fi

echo "Using container runtime: $RUNTIME"
echo "Install directory: $INSTALL_DIR"
echo

# Clone or update repo
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Create config if it doesn't exist
if [ ! -f "config.yaml" ]; then
    echo "Creating default config.yaml..."
    cp config.example.yaml config.yaml
    echo
    echo ">>> Edit $INSTALL_DIR/config.yaml to configure your environments <<<"
    echo
fi

# Start services
echo "Starting services..."
$COMPOSE up -d

# Wait for startup
echo "Waiting for services to start..."
sleep 10

# Check health
if curl -s http://localhost:8100/health | grep -q "healthy"; then
    echo
    echo "┌─────────────────────────────────────────┐"
    echo "│  Installation complete!                 │"
    echo "└─────────────────────────────────────────┘"
    echo
    echo "mcp-contexts is running at http://localhost:8100"
    echo
    echo "Next steps:"
    echo "  1. Edit $INSTALL_DIR/config.yaml with your environments"
    echo "  2. Restart: cd $INSTALL_DIR && $COMPOSE restart"
    echo "  3. Add session commands to your CLAUDE.md (see setup/CLAUDE.md)"
    echo
else
    echo
    echo "Warning: Service may still be starting (embedding model download)."
    echo "Check status with: curl http://localhost:8100/health"
    echo "View logs with: cd $INSTALL_DIR && $COMPOSE logs -f"
fi
