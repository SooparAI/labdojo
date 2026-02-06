#!/bin/bash
# ============================================================================
# Lab Dojo v10 - One-Click Installer for Mac
# ============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    🧪 Lab Dojo v10                            ║"
echo "║                  Serverless Edition                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}[1/5]${NC} Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "  ${GREEN}✓${NC} Python $PYTHON_VERSION found"
else
    echo "  ✗ Python not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "  Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install python@3.11
fi

echo
echo -e "${BLUE}[2/5]${NC} Installing Python dependencies..."
pip3 install --quiet --upgrade pip
pip3 install --quiet fastapi uvicorn aiohttp aiosqlite pydantic

echo -e "  ${GREEN}✓${NC} Dependencies installed"

echo
echo -e "${BLUE}[3/5]${NC} Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Ollama found"
else
    echo "  Installing Ollama..."
    if command -v brew &> /dev/null; then
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "  Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Pull model if needed
echo "  Checking for Qwen model..."
if ! ollama list 2>/dev/null | grep -q "qwen2.5:7b"; then
    echo "  Pulling Qwen 2.5 7B model (this may take a few minutes)..."
    ollama pull qwen2.5:7b
fi
echo -e "  ${GREEN}✓${NC} Ollama ready with Qwen 2.5 7B"

echo
echo -e "${BLUE}[4/5]${NC} Creating config directory..."
mkdir -p ~/.labdojo/logs
echo -e "  ${GREEN}✓${NC} Config directory: ~/.labdojo"

echo
echo -e "${BLUE}[5/5]${NC} Starting Lab Dojo..."
echo

# Kill any existing instance
pkill -f "labdojo.py" 2>/dev/null || true
sleep 1

# Start Lab Dojo
python3 "$SCRIPT_DIR/labdojo.py"
