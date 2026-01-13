-- Migration: Remove legacy temporary_speaker_id column
-- Date: 2025-01-11
-- Description: Remove the unused temporary_speaker_id column and its index

BEGIN;

-- Step 1: Drop the index if it exists
DROP INDEX IF EXISTS idx_speakers_temp_speaker;

-- Step 2: Drop the column
ALTER TABLE speakers DROP COLUMN IF EXISTS temporary_speaker_id;

COMMIT;

-- Verification
\d speakers