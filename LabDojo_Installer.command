#!/bin/bash
# ============================================================================
# Lab Dojo v10 - One-Click Installer & Launcher for macOS
# Works on Intel and Apple Silicon (M1/M2/M3/M4)
# ============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Lab Dojo v10                             ║"
echo "║              Research-Grade AI Assistant                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory (works even when double-clicked from Finder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo -e "${BLUE}[INFO]${NC} Detected Apple Silicon ($ARCH)"
    # Ensure Homebrew ARM path is in PATH
    export PATH="/opt/homebrew/bin:$PATH"
else
    echo -e "${BLUE}[INFO]${NC} Detected Intel Mac ($ARCH)"
    export PATH="/usr/local/bin:$PATH"
fi

# ========== STEP 1: Python ==========
echo
echo -e "${BLUE}[1/5]${NC} Checking Python 3..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        echo -e "  ${GREEN}✓${NC} Python $PYTHON_VERSION"
    else
        echo -e "  ${YELLOW}!${NC} Python $PYTHON_VERSION is too old (need 3.9+). Installing..."
        if command -v brew &> /dev/null; then
            brew install python@3.11
        else
            echo -e "  ${RED}✗${NC} Please install Python 3.9+ from https://python.org"
            read -p "Press Enter to exit..."
            exit 1
        fi
    fi
else
    echo "  Python 3 not found. Installing..."
    if ! command -v brew &> /dev/null; then
        echo "  Installing Homebrew first (required for macOS package management)..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Add brew to path for this session
        if [ "$ARCH" = "arm64" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        else
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
    brew install python@3.11
    echo -e "  ${GREEN}✓${NC} Python installed"
fi

# ========== STEP 2: Dependencies ==========
echo
echo -e "${BLUE}[2/5]${NC} Installing Python dependencies..."
python3 -m pip install --quiet --upgrade pip 2>/dev/null || pip3 install --quiet --upgrade pip
python3 -m pip install --quiet fastapi uvicorn aiohttp aiosqlite pydantic 2>/dev/null || pip3 install --quiet fastapi uvicorn aiohttp aiosqlite pydantic
echo -e "  ${GREEN}✓${NC} Dependencies installed"

# ========== STEP 3: Ollama ==========
echo
echo -e "${BLUE}[3/5]${NC} Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Ollama found"
else
    echo -e "  ${YELLOW}!${NC} Ollama not found. Installing..."
    if command -v brew &> /dev/null; then
        brew install ollama
    else
        echo -e "  ${YELLOW}!${NC} Downloading Ollama installer..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    echo -e "  ${GREEN}✓${NC} Ollama installed"
fi

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo "  Starting Ollama service..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Check what models are available
echo "  Checking available models..."
MODELS=$(ollama list 2>/dev/null || echo "")
if echo "$MODELS" | grep -q "llama3"; then
    MODEL_NAME=$(echo "$MODELS" | grep "llama3" | head -1 | awk '{print $1}')
    echo -e "  ${GREEN}✓${NC} Found model: $MODEL_NAME"
elif echo "$MODELS" | grep -q "qwen"; then
    MODEL_NAME=$(echo "$MODELS" | grep "qwen" | head -1 | awk '{print $1}')
    echo -e "  ${GREEN}✓${NC} Found model: $MODEL_NAME"
elif echo "$MODELS" | grep -q "mistral"; then
    MODEL_NAME=$(echo "$MODELS" | grep "mistral" | head -1 | awk '{print $1}')
    echo -e "  ${GREEN}✓${NC} Found model: $MODEL_NAME"
else
    echo -e "  ${YELLOW}!${NC} No model found. Pulling llama3:8b (this may take a few minutes)..."
    ollama pull llama3:8b
    echo -e "  ${GREEN}✓${NC} Model downloaded: llama3:8b"
fi

# ========== STEP 4: Config ==========
echo
echo -e "${BLUE}[4/5]${NC} Setting up config directory..."
mkdir -p ~/.labdojo/logs ~/.labdojo/knowledge
echo -e "  ${GREEN}✓${NC} Config: ~/.labdojo"

# ========== STEP 5: Launch ==========
echo
echo -e "${BLUE}[5/5]${NC} Starting Lab Dojo..."
echo

# Kill any existing instance
pkill -f "labdojo.py" 2>/dev/null || true
sleep 1

# Open browser after a short delay
(sleep 4 && open "http://localhost:8080") &

# Start Lab Dojo
echo -e "${GREEN}Lab Dojo is starting at http://localhost:8080${NC}"
echo -e "Press Ctrl+C to stop."
echo
python3 "$SCRIPT_DIR/labdojo.py"
