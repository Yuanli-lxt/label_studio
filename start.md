cd ~/projects/label-platform
docker compose --env-file .env -f infra/docker-compose.yml up -d
docker compose --env-file .env -f infra/docker-compose.yml ps


/label-studio/files/images

Local URLs:
- Label Studio OSS: `http://localhost:18080`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`
- ML backend health: `http://localhost:9090/health`
- Trainer health: `http://localhost:9091/health`
