# Database SQL Files

This directory contains SQL files that define database-specific functionality such as triggers, functions, and indexes that are not easily represented in SQLAlchemy models.

## Files

### triggers.sql
Contains database trigger definitions that maintain automatic statistics and relationships between tables. Currently includes:

- Speaker statistics triggers that automatically update the following fields in the `speakers` table when `speaker_transcriptions` are modified:
  - `total_segments`
  - `total_duration`
  - `appearance_count`
  - `last_seen`
  - `last_content_id`

### Managing Triggers

To install/update triggers:
```bash
psql -d content_processing -f src/database/sql/triggers.sql
```

To verify triggers are installed:
```sql
SELECT 
    event_object_table AS table_name,
    trigger_name,
    event_manipulation AS trigger_event,
    action_statement AS trigger_action
FROM information_schema.triggers
WHERE event_object_table = 'speaker_transcriptions'
ORDER BY trigger_name;
```

To check trigger status:
```sql
SELECT tgname AS trigger_name,
       tgenabled AS trigger_enabled,
       tgtype AS trigger_type
FROM pg_trigger
WHERE tgrelid = 'speaker_transcriptions'::regclass;
```

To manually recalculate all speaker stats:
```sql
SELECT recalculate_all_speaker_stats();
```

## Adding New SQL Files

When adding new SQL files to this directory:

1. Document the purpose and functionality in this README
2. Add appropriate comments in the SQL file
3. Update relevant model documentation in `models.py`
4. Include verification queries to check if the functionality is properly installed
5. Add installation/update instructions if needed 