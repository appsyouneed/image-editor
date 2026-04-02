#!/bin/bash

sudo cp /root/picgen/picgen.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable picgen.service
sudo systemctl start picgen.service

echo "✓ picgen service started!"
echo ""
echo "Check status: sudo systemctl status picgen"
echo "View logs: sudo journalctl -u picgen -f"
