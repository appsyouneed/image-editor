#!/bin/bash
set -e

echo "=== Picgen Image Editor VPS Setup ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip python3-venv ffmpeg wget git git-lfs bc curl software-properties-common

echo "Installing CUDA 12.1 toolkit..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-1
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> /etc/environment
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> /etc/environment
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

echo "Creating Python virtual environment..."
python3 -m venv "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Installing Hugging Face CLI..."
pip install "huggingface_hub[cli]>=1.5.0"

echo "Creating local model directories..."
mkdir -p "$SCRIPT_DIR/models/Qwen-Image-Edit-2511"
mkdir -p "$SCRIPT_DIR/models/rapid-aio/v23"
chmod -R 777 "$SCRIPT_DIR/models"

echo "=== Model Download ==="
echo "Models will be downloaded automatically on first run."
echo "To pre-download models now, run:"
echo "  source $SCRIPT_DIR/venv/bin/activate"
echo "  huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir $SCRIPT_DIR/models/Qwen-Image-Edit-2511"
echo "  huggingface-cli download Phr00t/Qwen-Image-Edit-Rapid-AIO --include 'v23/Qwen-Rapid-AIO-NSFW-v23.safetensors' --local-dir $SCRIPT_DIR/models/rapid-aio"

echo "Verifying GPU accessibility..."
"$SCRIPT_DIR/venv/bin/python3" -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Total VRAM: {total_vram:.1f}GB')
else:
    print('GPU Device: None (CPU mode)')
"

echo "=== Setup Complete ==="
echo ""
echo "Setting up systemd service..."

cat > /etc/systemd/system/picgen.service <<EOF
[Unit]
Description=Picgen Image Editor Application
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
Environment="PYTHONUNBUFFERED=1"
Environment="HF_HOME=/root/.cache/huggingface"
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="CUDA_LAUNCH_BLOCKING=0"
ExecStart=$SCRIPT_DIR/venv/bin/python3 $SCRIPT_DIR/app.py
Restart=always
RestartSec=10
StandardOutput=append:$SCRIPT_DIR/picgen.log
StandardError=append:$SCRIPT_DIR/picgen.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable picgen
systemctl start picgen

echo ""
echo "Service commands:"
echo "  systemctl start picgen   - Start picgen"
echo "  systemctl stop picgen    - Stop picgen"
echo "  systemctl status picgen  - Check status"
echo "  systemctl restart picgen - Restart picgen"
echo ""
echo "View live output:"
echo "  tail -f $SCRIPT_DIR/picgen.log"
echo ""
echo "To run manually: $SCRIPT_DIR/venv/bin/python3 $SCRIPT_DIR/app.py"
echo "The app will be accessible at: http://0.0.0.0:7860"
