#!/bin/bash

# Start the Ollama server
sudo systemctl start ollama

# Navigate to the OpenWebUI directory
cd /home/repos/fluffy-pancake/open-webui

# Start the OpenWebUI docker container
docker compose up -d
