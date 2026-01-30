#!/bin/bash
# Start full scratch-3d stack (API + UI + FLUX + reconstruction + Redis)
cd "$(dirname "$0")/.."
docker-compose up -d
