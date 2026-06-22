-- 初始化 PostgreSQL 数据库
-- 创建扩展
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 创建文档表
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}',
    partition TEXT DEFAULT 'text',
    created_at TIMESTAMP DEFAULT NOW()
);

-- 创建索引（数据量足够后手动创建 IVFFlat 索引）
CREATE INDEX IF NOT EXISTS idx_documents_content_trgm ON documents USING gin (content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_documents_partition ON documents (partition);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin (metadata);

-- 提示信息
DO $$
BEGIN
    RAISE NOTICE 'RAG database initialized successfully!';
    RAISE NOTICE 'Note: IVFFlat index will be created automatically when sufficient data is available.';
END $$;
