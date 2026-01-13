-- Speaker Statistics Triggers
-- These triggers automatically maintain statistics in the speakers table
-- based on changes to the speaker_transcriptions table.

-- Function to update speaker statistics
CREATE OR REPLACE FUNCTION update_speaker_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Update total_segments, total_duration, last_seen, and last_content_id
        UPDATE speakers s
        SET 
            total_segments = total_segments + 1,
            total_duration = total_duration + (NEW.end_time - NEW.start_time),
            last_seen = CURRENT_TIMESTAMP,
            last_content_id = (SELECT content_id FROM content WHERE id = NEW.content_id),
            appearance_count = (
                SELECT COUNT(DISTINCT content_id) 
                FROM speaker_transcriptions 
                WHERE speaker_id = NEW.speaker_id
            )
        WHERE s.id = NEW.speaker_id;
        
    ELSIF TG_OP = 'DELETE' THEN
        -- Update total_segments, total_duration, and appearance_count
        UPDATE speakers s
        SET 
            total_segments = total_segments - 1,
            total_duration = total_duration - (OLD.end_time - OLD.start_time),
            appearance_count = (
                SELECT COUNT(DISTINCT content_id) 
                FROM speaker_transcriptions 
                WHERE speaker_id = OLD.speaker_id
            )
        WHERE s.id = OLD.speaker_id;
        
        -- Update last_seen and last_content_id to the most recent transcription
        UPDATE speakers s
        SET 
            last_seen = (
                SELECT created_at 
                FROM speaker_transcriptions 
                WHERE speaker_id = OLD.speaker_id 
                ORDER BY created_at DESC 
                LIMIT 1
            ),
            last_content_id = (
                SELECT c.content_id 
                FROM speaker_transcriptions st 
                JOIN content c ON c.id = st.content_id 
                WHERE st.speaker_id = OLD.speaker_id 
                ORDER BY st.created_at DESC 
                LIMIT 1
            )
        WHERE s.id = OLD.speaker_id;
        
    ELSIF TG_OP = 'UPDATE' THEN
        -- Update total_duration to reflect the change
        UPDATE speakers s
        SET 
            total_duration = total_duration - (OLD.end_time - OLD.start_time) + (NEW.end_time - NEW.start_time),
            last_seen = CURRENT_TIMESTAMP,
            last_content_id = (SELECT content_id FROM content WHERE id = NEW.content_id),
            appearance_count = (
                SELECT COUNT(DISTINCT content_id) 
                FROM speaker_transcriptions 
                WHERE speaker_id = NEW.speaker_id
            )
        WHERE s.id = NEW.speaker_id;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create the triggers
DROP TRIGGER IF EXISTS speaker_stats_insert_trigger ON speaker_transcriptions;
DROP TRIGGER IF EXISTS speaker_stats_update_trigger ON speaker_transcriptions;
DROP TRIGGER IF EXISTS speaker_stats_delete_trigger ON speaker_transcriptions;

CREATE TRIGGER speaker_stats_insert_trigger
    AFTER INSERT ON speaker_transcriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_speaker_stats();

CREATE TRIGGER speaker_stats_update_trigger
    AFTER UPDATE ON speaker_transcriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_speaker_stats();

CREATE TRIGGER speaker_stats_delete_trigger
    AFTER DELETE ON speaker_transcriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_speaker_stats();

-- Function to recalculate all speaker stats
CREATE OR REPLACE FUNCTION recalculate_all_speaker_stats()
RETURNS void AS $$
BEGIN
    UPDATE speakers s SET
        total_segments = (
            SELECT COUNT(*) 
            FROM speaker_transcriptions 
            WHERE speaker_id = s.id
        ),
        total_duration = (
            SELECT COALESCE(SUM(end_time - start_time), 0) 
            FROM speaker_transcriptions 
            WHERE speaker_id = s.id
        ),
        appearance_count = (
            SELECT COUNT(DISTINCT content_id) 
            FROM speaker_transcriptions 
            WHERE speaker_id = s.id
        ),
        last_seen = (
            SELECT MAX(created_at) 
            FROM speaker_transcriptions 
            WHERE speaker_id = s.id
        ),
        last_content_id = (
            SELECT c.content_id 
            FROM speaker_transcriptions st 
            JOIN content c ON c.id = st.content_id 
            WHERE st.speaker_id = s.id 
            ORDER BY st.created_at DESC 
            LIMIT 1
        );
END;
$$ LANGUAGE plpgsql;

-- To verify triggers are installed:
/*
SELECT 
    event_object_table AS table_name,
    trigger_name,
    event_manipulation AS trigger_event,
    action_statement AS trigger_action
FROM information_schema.triggers
WHERE event_object_table = 'speaker_transcriptions'
ORDER BY trigger_name;
*/

-- To manually recalculate all stats:
-- SELECT recalculate_all_speaker_stats();

-- To check trigger status:
/*
SELECT tgname AS trigger_name,
       tgenabled AS trigger_enabled,
       tgtype AS trigger_type
FROM pg_trigger
WHERE tgrelid = 'speaker_transcriptions'::regclass;
*/ 