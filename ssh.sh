#!/bin/bash
ssh-keygen -f '/root/.ssh/known_hosts' -R '194.93.48.12'
ssh -o StrictHostKeyChecking=no root@194.93.48.12
echo ""
echo "To connect manually, run:"
echo "ssh root@194.93.48.12"
