# 公开数据库与上传原件快照

`rag_db-20260928T073335Z.dump` 是 **未加密** 的 PostgreSQL custom-format 归档。它由先前公开快照恢复到隔离 PostgreSQL 16 + pgvector 后，使用项目锁定的 GPU 推理依赖对 1,867 条文档重新生成向量，并按 CPU/CUDA 通用指纹更新快照，再于 2026-09-28 07:33 UTC 导出。此仓库公开，任何人都可以下载归档与上传原件。

- SHA-256：`05bab7eceebfe12ac1ed04c527061d71c3a82c3c02e670ee7ef96f4e09554e62`
- 文件大小：9,837,089 字节
- `documents` 为 1,867 行，`rag_tenants` 为 3 行
- 嵌入指纹：`a10b143bd127ba129eda09ba3cde67e20015a3335b9d0bc21440bc2a43b4803f`
- 已在空白数据库中完整恢复；CUDA API 与 CPU worker 均报告语料 `ready`。API 的 PostgreSQL、Embedding、Reranker 和 Redis 检查均通过；LLM 因模型 API Key 占位符而尚未就绪。

归档包含文档正文、向量、元数据、租户和迁移记录；仅供迁移回退的旧向量备份表已从归档排除。仓库另跟踪根目录 `.env`、`industrial-rag/.env` 和 `industrial-rag/data/uploads` 中的 7 个原始上传文件（合计 21,702,381 字节）。只有模型服务 API Key 保持占位符；其他当前服务凭据按要求公开。模型权重、Redis 缓存/任务历史和 Keycloak 数据卷不在仓库中。

## 从新克隆恢复

仓库已包含两份 `.env` 和上传原件。在**全新、空白的 PostgreSQL 卷**上，从仓库根目录执行：

```powershell
docker compose --env-file industrial-rag/.env up -d postgres
docker cp database/rag_db-20260928T073335Z.dump rag-postgres:/tmp/rag_db.dump
docker exec rag-postgres pg_restore -U postgres -d rag_db `
  --exit-on-error --single-transaction --no-owner --no-privileges /tmp/rag_db.dump
docker exec rag-postgres psql -U postgres -d rag_db -tAc "SELECT count(*) FROM documents;"
```

最后一条命令应返回 `1867`。数据库角色不包含在 `pg_dump` 中：本机 `laptop` 配置启动 API 时会自动迁移并创建运行角色；使用 Compose 的 `api`/`worker` profile 时，`migrate` 服务会先执行。不要跳过迁移而直接启动只读 API。

随后按根目录 `README.md` 安装锁定依赖、下载模型、填写自己的模型服务 API Key 并启动服务。旧 Conda 环境中的推理依赖会造成语料指纹不匹配，必须按安装脚本更新到锁定版本。此快照是单次备份；后续数据库写入或新上传的原件需要再次备份。
