#!/bin/bash

sudo systemctl stop picgen.service
sudo systemctl disable picgen.service

echo "✓ picgen service stopped!"
