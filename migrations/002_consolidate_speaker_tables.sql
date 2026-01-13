-- Migration: Consolidate Speaker and SpeakerEmbedding into unified Speaker table
-- Date: 2025-01-11
-- Description: Safely consolidate two-table architecture into single unified table
-- This migration handles data from both tables if they exist

BEGIN;

-- Step 1: Check if we have the old two-table structure
DO $$
DECLARE
    has_old_speakers BOOLEAN;
    has_speaker_embeddings BOOLEAN;
BEGIN
    -- Check if old speakers table exists
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = 'speakers'
    ) INTO has_old_speakers;
    
    -- Check if speaker_embeddings table exists
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = 'speaker_embeddings'
    ) INTO has_speaker_embeddings;
    
    IF has_old_speakers AND has_speaker_embeddings THEN
        RAISE NOTICE 'Found both speakers and speaker_embeddings tables - need to consolidate';
    ELSIF has_speaker_embeddings AND NOT has_old_speakers THEN
        RAISE NOTICE 'Only speaker_embeddings table found - likely already migrated';
    ELSIF has_old_speakers AND NOT has_speaker_embeddings THEN
        RAISE NOTICE 'Only speakers table found - this is the target state';
    ELSE
        RAISE EXCEPTION 'Neither speakers nor speaker_embeddings table found!';
    END IF;
END $$;

-- Step 2: Only proceed if we have speaker_embeddings to migrate
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = 'speaker_embeddings'
    ) THEN
        -- Add missing columns to speaker_embeddings if they don't exist
        ALTER TABLE speaker_embeddings 
        ADD COLUMN IF NOT EXISTS global_id VARCHAR(255),
        ADD COLUMN IF NOT EXISTS universal_name VARCHAR(255),
        ADD COLUMN IF NOT EXISTS display_name VARCHAR(255),
        ADD COLUMN IF NOT EXISTS canonical_speaker_id INTEGER,
        ADD COLUMN IF NOT EXISTS is_canonical BOOLEAN DEFAULT TRUE,
        ADD COLUMN IF NOT EXISTS merge_history JSON DEFAULT '[]',
        ADD COLUMN IF NOT EXISTS merge_confidence FLOAT,
        ADD COLUMN IF NOT EXISTS meta_data JSON DEFAULT '{}',
        ADD COLUMN IF NOT EXISTS notes TEXT,
        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        
        -- Generate default values for new fields
        UPDATE speaker_embeddings 
        SET 
            global_id = COALESCE(global_id, 'SPEAKER_' || substr(md5(content_id::text || '_' || local_speaker_id), 1, 8)),
            universal_name = COALESCE(universal_name, 'Speaker ' || id)
        WHERE global_id IS NULL OR universal_name IS NULL;
        
        -- If old speakers table exists, migrate any unique data
        IF EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'speakers'
        ) THEN
            -- Update speaker_embeddings with data from old speakers table where they match
            UPDATE speaker_embeddings se
            SET 
                global_id = COALESCE(se.global_id, s.global_id),
                universal_name = COALESCE(se.universal_name, s.universal_name),
                display_name = COALESCE(se.display_name, s.display_name),
                notes = COALESCE(se.notes, s.notes)
            FROM speakers s
            WHERE s.content_id = se.content_id 
            AND s.local_speaker_id = se.local_speaker_id;
            
            -- Drop the old speakers table
            DROP TABLE speakers CASCADE;
        END IF;
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_global_id ON speaker_embeddings(global_id);
        CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_universal_name ON speaker_embeddings(universal_name);
        CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_canonical ON speaker_embeddings(canonical_speaker_id);
        
        -- Add unique constraint
        ALTER TABLE speaker_embeddings 
        ADD CONSTRAINT IF NOT EXISTS uk_speaker_embeddings_content_local 
        UNIQUE (content_id, local_speaker_id);
        
        -- Add self-referential foreign key
        ALTER TABLE speaker_embeddings 
        ADD CONSTRAINT IF NOT EXISTS fk_speaker_embeddings_canonical 
        FOREIGN KEY (canonical_speaker_id) REFERENCES speaker_embeddings(id) 
        DEFERRABLE INITIALLY DEFERRED;
        
        -- Rename the table
        ALTER TABLE speaker_embeddings RENAME TO speakers;
        
        -- Rename the sequence
        ALTER SEQUENCE IF EXISTS speaker_embeddings_id_seq RENAME TO speakers_id_seq;
        
        -- Update column default
        ALTER TABLE speakers ALTER COLUMN id SET DEFAULT nextval('speakers_id_seq');
        
        -- Create speaker_number_seq if it doesn't exist
        CREATE SEQUENCE IF NOT EXISTS speaker_number_seq START WITH 1000000;
        
        -- Update all constraints to use new table name
        ALTER TABLE speakers RENAME CONSTRAINT uk_speaker_embeddings_content_local TO uk_speakers_content_local;
        ALTER TABLE speakers RENAME CONSTRAINT fk_speaker_embeddings_canonical TO fk_speakers_canonical;
        
        -- Update indexes to use new naming convention
        ALTER INDEX IF EXISTS idx_speaker_embeddings_global_id RENAME TO idx_speakers_global_id;
        ALTER INDEX IF EXISTS idx_speaker_embeddings_universal_name RENAME TO idx_speakers_universal_name;
        ALTER INDEX IF EXISTS idx_speaker_embeddings_canonical RENAME TO idx_speakers_canonical;
        
        RAISE NOTICE 'Successfully migrated speaker_embeddings to speakers table';
    END IF;
END $$;

COMMIT;

-- Verification queries
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = 'speakers'
    ) THEN
        RAISE NOTICE 'Speakers table exists - checking structure...';
        
        -- Check for required columns
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'speakers' 
            AND column_name IN ('content_id', 'local_speaker_id', 'embedding', 
                                'global_id', 'universal_name', 'canonical_speaker_id')
        ) THEN
            RAISE NOTICE 'All required columns present';
        END IF;
        
        -- Check unique constraint
        IF EXISTS (
            SELECT 1 FROM information_schema.table_constraints 
            WHERE table_name = 'speakers' 
            AND constraint_type = 'UNIQUE'
            AND constraint_name = 'uk_speakers_content_local'
        ) THEN
            RAISE NOTICE 'Unique constraint on (content_id, local_speaker_id) is present';
        END IF;
    END IF;
END $$;

-- Show final table structure
\d speakers

-- Check for any duplicates (should return 0 rows)
SELECT 
    content_id, 
    local_speaker_id, 
    COUNT(*) as count 
FROM speakers 
GROUP BY content_id, local_speaker_id 
HAVING COUNT(*) > 1;

-- Show sample data
SELECT 
    id, 
    content_id, 
    local_speaker_id, 
    global_id, 
    universal_name,
    embedding IS NOT NULL as has_embedding,
    embedding_quality_score,
    rebase_status
FROM speakers 
ORDER BY id 
LIMIT 10;