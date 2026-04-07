#!/bin/bash
# =============================================================================
# AI Video Generator - Setup Script
# =============================================================================
# Run this script on your GPU server to:
#   1. Install system dependencies
#   2. Set up Python virtual environment
#   3. Install Python packages
#   4. Download default model
#   5. Start the server with ngrok
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  🎬 AI Video Generator - Setup${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""

# ---- Check GPU ----
echo -e "${YELLOW}[1/7] Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected - will use CPU (slow)${NC}"
fi
echo ""

# ---- Install system deps ----
echo -e "${YELLOW}[2/7] Installing system dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
    python3.10 python3-pip python3.10-venv ffmpeg curl wget git

if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✅ ffmpeg installed${NC}"
else
    echo -e "${RED}❌ ffmpeg installation failed${NC}"
    exit 1
fi
echo ""

# ---- Setup Python venv ----
echo -e "${YELLOW}[3/7] Setting up Python environment...${NC}"
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo -e "${GREEN}✅ Python packages installed${NC}"
echo ""

# ---- Create directories ----
echo -e "${YELLOW}[4/7] Creating data directories...${NC}"
mkdir -p models outputs uploads
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# ---- Install ngrok ----
echo -e "${YELLOW}[5/7] Setting up ngrok...${NC}"
if command -v ngrok &> /dev/null; then
    echo -e "${GREEN}✅ ngrok already installed${NC}"
else
    echo "Installing ngrok..."
    curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz
    sudo mv ngrok /usr/local/bin/ 2>/dev/null || mv ngrok /usr/local/bin/
    echo -e "${GREEN}✅ ngrok installed${NC}"
fi

# Configure ngrok if token provided
if [ -n "$NGROK_AUTH_TOKEN" ]; then
    ngrok config add-authtoken "$NGROK_AUTH_TOKEN"
    echo -e "${GREEN}✅ ngrok configured${NC}"
else
    echo -e "${YELLOW}⚠️  NGROK_AUTH_TOKEN not set. Run: ngrok config add-authtoken YOUR_TOKEN${NC}"
fi
echo ""

# ---- Download model ----
echo -e "${YELLOW}[6/7] Downloading AI model...${NC}"
MODEL="${DEFAULT_MODEL:-stabilityai/stable-video-diffusion-img2vid-xt}"

# Check if already downloaded
SAFE_NAME=$(echo "$MODEL" | sed 's|/|_|g')
if [ -d "models/$SAFE_NAME" ] && [ "$(ls -A models/$SAFE_NAME 2>/dev/null)" ]; then
    echo -e "${GREEN}✅ Model already exists: $MODEL${NC}"
else
    echo "Downloading model: $MODEL (this may take a while...)"
    source venv/bin/activate
    python3 -c "
from app.model_manager import model_manager
info = model_manager.download_model(model_id='$MODEL', source='huggingface')
print(f'Downloaded: {info.name} ({info.size_mb} MB)')
"
    echo -e "${GREEN}✅ Model downloaded${NC}"
fi
echo ""

# ---- Start server ----
echo -e "${YELLOW}[7/7] Starting server...${NC}"
source venv/bin/activate

# Kill existing processes
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f ngrok 2>/dev/null || true
sleep 2

# Start server
echo "Starting AI Video Generator on port 8000..."
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info > server.log 2>&1 &
echo $! > server.pid

# Wait for server
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/api/system/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is running on port 8000${NC}"
        break
    fi
    sleep 2
done

# Start ngrok
echo "Starting ngrok tunnel..."
ngrok http 8000 --log=stdout > ngrok.log 2>&1 &
echo $! > ngrok.pid

sleep 5
TUNNEL_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['tunnels'][0]['public_url'])" 2>/dev/null)

echo ""
echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}  🎉 AI Video Generator is LIVE!${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""
if [ -n "$TUNNEL_URL" ]; then
    echo -e "  🌍 Public URL: ${GREEN}${TUNNEL_URL}${NC}"
    echo -e "  📍 Local URL:  http://localhost:8000"
else
    echo -e "  📍 Local URL:  http://localhost:8000"
    echo -e "  ${YELLOW}⚠️  ngrok tunnel may not be established yet${NC}"
fi
echo ""
echo -e "  Server PID: $(cat server.pid)"
echo -e "  Ngrok PID:  $(cat ngrok.pid)"
echo ""
echo -e "  To view logs: ${BLUE}tail -f server.log${NC}"
echo -e "  To stop:      ${BLUE}kill \$(cat server.pid); kill \$(cat ngrok.pid)${NC}"
echo ""
