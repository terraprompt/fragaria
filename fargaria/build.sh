#!/bin/bash

# Build the Svelte app
cd frontend
npm install
npm run build

# Copy the built files to the FastAPI static directory
mkdir -p ../static/build
cp -r public/build/* ../static/build/

echo "Build complete. Svelte app is ready to be served by FastAPI."