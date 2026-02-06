#!/bin/bash
# ============================================================================
# Lab Dojo v8 - Serverless Setup Script
# Sets up Vast.ai serverless worker group
# ============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Lab Dojo v8 - Serverless Setup                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Configuration
VASTAI_API_KEY="3b891248bfa1eb4c0811be10a08afa3fa87765d5672a5150c4ec68f81f81cebf"
ENDPOINT_NAME="labdojo-qwen32b"
DOCKER_IMAGE="labdojo/inference-brain:v1.0"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Step 1: Check Docker${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found!${NC}"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker found"
echo

echo -e "${BLUE}Step 2: Build Docker image${NC}"
cd "$(dirname "$0")/../docker"
docker build -t ${DOCKER_IMAGE} .
echo -e "${GREEN}✓${NC} Docker image built"
echo

echo -e "${BLUE}Step 3: Push to Docker Hub${NC}"
echo "Please login to Docker Hub if not already logged in..."
docker login
docker push ${DOCKER_IMAGE}
echo -e "${GREEN}✓${NC} Docker image pushed"
echo

echo -e "${BLUE}Step 4: Create Vast.ai Template${NC}"
echo -e "${YELLOW}Manual step required:${NC}"
echo "1. Go to https://console.vast.ai/templates"
echo "2. Click 'Create Template'"
echo "3. Enter:"
echo "   - Name: labdojo-inference-v1"
echo "   - Image: ${DOCKER_IMAGE}"
echo "   - Port: 8000"
echo "   - Environment Variables:"
echo "     MODEL_NAME=Qwen/Qwen2.5-32B-Instruct-AWQ"
echo "     PORT=8000"
echo "4. Click 'Create'"
echo "5. Copy the Template ID"
echo
read -p "Enter Template ID: " TEMPLATE_ID

if [ -z "$TEMPLATE_ID" ]; then
    echo -e "${RED}Template ID required!${NC}"
    exit 1
fi
echo

echo -e "${BLUE}Step 5: Create Serverless Worker Group${NC}"
RESPONSE=$(curl -s --request POST \
  --url https://console.vast.ai/api/v0/workergroups/ \
  --header "Authorization: Bearer ${VASTAI_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data "{
    \"endpoint_name\": \"${ENDPOINT_NAME}\",
    \"template_id\": ${TEMPLATE_ID},
    \"search_params\": \"gpu_name=RTX_4080S rentable=true verified=true\",
    \"min_load\": 0,
    \"target_util\": 0.8,
    \"cold_mult\": 2,
    \"cold_workers\": 0,
    \"max_workers\": 5,
    \"test_workers\": 1,
    \"gpu_ram\": 16
  }")

echo "Response: $RESPONSE"

if echo "$RESPONSE" | grep -q '"success": true'; then
    WORKER_GROUP_ID=$(echo "$RESPONSE" | grep -o '"id": [0-9]*' | head -1 | grep -o '[0-9]*')
    echo -e "${GREEN}✓${NC} Worker group created! ID: ${WORKER_GROUP_ID}"
else
    echo -e "${RED}Failed to create worker group${NC}"
    echo "Response: $RESPONSE"
    exit 1
fi
echo

echo -e "${BLUE}Step 6: Test Endpoint${NC}"
echo "Sending test request (may take 60-90 seconds for cold start)..."

TEST_RESPONSE=$(curl -s --request POST \
  --url https://console.vast.ai/api/v0/route \
  --header "Authorization: Bearer ${VASTAI_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data "{
    \"endpoint_name\": \"${ENDPOINT_NAME}\",
    \"input\": {
      \"prompt\": \"Hello! What is 2+2?\",
      \"max_tokens\": 50
    }
  }" \
  --max-time 180)

echo "Response: $TEST_RESPONSE"

if echo "$TEST_RESPONSE" | grep -q '"text"'; then
    echo -e "${GREEN}✓${NC} Serverless endpoint working!"
else
    echo -e "${YELLOW}Note: First request may take longer due to cold start${NC}"
fi
echo

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Endpoint: ${ENDPOINT_NAME}"
echo "║  Worker Group ID: ${WORKER_GROUP_ID}"
echo "║  Scaling: 0 to 5 workers (auto)"
echo "║  Cost when idle: \$0/hour"
echo "║  Cost when active: ~\$0.30/hour"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "Now run Lab Dojo:"
echo "  Mac: ./LabDojo_Installer.command"
echo "  Windows: LabDojo_Installer.bat"
