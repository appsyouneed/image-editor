#!/bin/bash

sudo cp /root/image-editor/image-editor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable image-editor.service
sudo systemctl start image-editor.service

echo "✓ Image editor service started!"
echo ""
echo "Check status: sudo systemctl status image-editor"
echo "View logs: sudo journalctl -u image-editor -f"
