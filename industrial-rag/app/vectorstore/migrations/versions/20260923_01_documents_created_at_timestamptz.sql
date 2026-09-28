DO $$
DECLARE
    created_at_type text;
BEGIN
    SELECT format_type(attribute.atttypid, attribute.atttypmod)
      INTO created_at_type
    FROM pg_attribute AS attribute
    WHERE attribute.attrelid = 'public.documents'::regclass
      AND attribute.attname = 'created_at'
      AND NOT attribute.attisdropped;

    IF created_at_type = 'timestamp without time zone' THEN
        -- Historical values were written by UTC-configured application
        -- databases. Preserve those instants while making the timezone
        -- contract explicit for API serialization and browser clients.
        ALTER TABLE documents
            ALTER COLUMN created_at TYPE TIMESTAMPTZ
            USING created_at AT TIME ZONE 'UTC';
    ELSIF created_at_type IS DISTINCT FROM 'timestamp with time zone' THEN
        RAISE EXCEPTION 'documents.created_at has unexpected type: %', created_at_type;
    END IF;

    ALTER TABLE documents ALTER COLUMN created_at SET DEFAULT NOW();
END
$$;
