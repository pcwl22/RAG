DO $$
DECLARE
    primary_key_columns name[];
    rls_enabled boolean;
    rls_forced boolean;
    rag_app_is_privileged boolean;
BEGIN
    SELECT array_agg(attribute.attname ORDER BY key_column.ordinality)
      INTO primary_key_columns
    FROM pg_constraint AS constraint_row
    CROSS JOIN LATERAL unnest(constraint_row.conkey)
        WITH ORDINALITY AS key_column(attnum, ordinality)
    JOIN pg_attribute AS attribute
      ON attribute.attrelid = constraint_row.conrelid
     AND attribute.attnum = key_column.attnum
    WHERE constraint_row.conrelid = 'public.documents'::regclass
      AND constraint_row.contype = 'p';

    IF primary_key_columns IS DISTINCT FROM ARRAY['tenant_id', 'id']::name[] THEN
        RAISE EXCEPTION 'documents primary key must be (tenant_id, id)';
    END IF;

    SELECT relrowsecurity, relforcerowsecurity
      INTO rls_enabled, rls_forced
    FROM pg_class
    WHERE oid = 'public.documents'::regclass;

    IF NOT COALESCE(rls_enabled, false) OR NOT COALESCE(rls_forced, false) THEN
        RAISE EXCEPTION 'documents must have enabled and forced row-level security';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_policy
        WHERE polrelid = 'public.documents'::regclass
          AND polname = 'documents_tenant_isolation'
    ) THEN
        RAISE EXCEPTION 'documents tenant isolation policy is missing';
    END IF;

    SELECT rolsuper OR rolbypassrls
      INTO rag_app_is_privileged
    FROM pg_roles
    WHERE rolname = 'rag_app';

    IF rag_app_is_privileged IS DISTINCT FROM false THEN
        RAISE EXCEPTION 'rag_app must not be superuser or bypass RLS';
    END IF;

    IF to_regclass('public.idx_documents_tenant_partition') IS NULL
       OR to_regclass('public.idx_documents_tenant_citation') IS NULL
       OR to_regclass('public.idx_documents_embedding_ivfflat') IS NULL THEN
        RAISE EXCEPTION 'required tenant retrieval indexes are missing';
    END IF;
END
$$;
