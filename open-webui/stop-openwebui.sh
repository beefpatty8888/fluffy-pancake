#!/bin/bash

# Navigate to the OpenWebUI directory
cd /home/repos/fluffy-pancake/open-webui

# Stop the OpenWebUI docker container
docker compose down

# Stop the Ollama server
sudo systemctl stop ollama
