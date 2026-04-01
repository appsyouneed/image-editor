#!/bin/bash

sudo systemctl stop image-editor.service
sudo systemctl disable image-editor.service

echo "✓ Image editor service stopped!"
