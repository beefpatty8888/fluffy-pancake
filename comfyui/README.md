# Create docker image
```
docker build --progress=plain --no-cache -t comfy-ui .
```

# Run docker image
```
docker run -d --network=host --gpus all --name comfy-ui comfy-ui:latest
```

# NOTES
You can drag the image to Comfy UI to view the workflow.