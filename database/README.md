# 公开数据库快照

`rag_db-20260928T015634Z.dump` 是 2026-09-28 01:56 UTC 从本地 `rag_db`
导出的、**未加密**的 PostgreSQL custom-format 归档。此仓库公开，任何人都可以下载该文件。

- SHA-256：`8fab1f4619d85433fa515fb399bb35c28caebc118328ae6b7daf462dfc529c61`
- 文件大小：37,635,274 字节
- 导出时：`documents` 为 1,867 行，`rag_tenants` 为 3 行
- 已用 PostgreSQL 16 + pgvector 在一次性数据库容器中完整恢复，并核对上述行数

归档包含数据库中的文档正文、向量、元数据、租户和迁移记录。它不包含 `.env`、
`industrial-rag/data/uploads` 中的原始上传文件、Redis/Keycloak 卷或模型权重。
仓库同时跟踪根目录 `.env` 和 `industrial-rag/.env`：模型服务 API Key 保持占位符，
其余当前本地凭据公开保留。需要独立凭据时可参考两份 `.env.example`。

## 从新克隆恢复

仓库已包含 `industrial-rag/.env` 和数据库凭据；先按根目录 `README.md` 核对配置。
在**全新、空白的 PostgreSQL 卷**上，从仓库根目录执行：

```powershell
docker compose --env-file industrial-rag/.env up -d postgres
docker cp database/rag_db-20260928T015634Z.dump rag-postgres:/tmp/rag_db.dump
docker exec rag-postgres pg_restore -U postgres -d rag_db `
  --exit-on-error --single-transaction --no-owner --no-privileges /tmp/rag_db.dump
docker exec rag-postgres psql -U postgres -d rag_db -tAc "SELECT count(*) FROM documents;"
```

最后一条命令应返回 `1867`。之后再按根目录 `README.md` 准备模型并启动 API、worker 和前端。
此归档是单次快照；后续写入数据库的数据需要另行备份。
