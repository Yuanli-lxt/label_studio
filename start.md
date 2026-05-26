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


目的:它比较适合这些场景：做训练数据集；给大模型/RAG/Agent 做人工评测

使用 Superpowers 的流程，自动选择合适的 skills。不要跳过设计、测试、review 和验证。