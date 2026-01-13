# PostgreSQL Worker Manager

This script automatically manages PostgreSQL `pg_hba.conf` entries for worker nodes based on your `config.yaml` file.

## Features

- Extracts worker IP addresses from `config/config.yaml`
- Updates `pg_hba.conf` with proper authentication entries
- Creates backups before making changes
- Validates PostgreSQL configuration
- Reloads PostgreSQL configuration automatically

## Usage

### Check Current Configuration
```bash
python src/database/pg_worker_manager.py --check
```

### Preview Changes (Dry Run)
```bash
python src/database/pg_worker_manager.py --dry-run
```

### Update PostgreSQL Configuration
```bash
python src/database/pg_worker_manager.py --update
```

### Update Without Reloading PostgreSQL
```bash
python src/database/pg_worker_manager.py --update --no-reload
```

## Adding New Workers

1. **Add worker to config.yaml**:
   ```yaml
   processing:
     workers:
       worker6:
         eth: 10.0.0.100
         wifi: 10.0.0.101
         enabled: true
         task_types: ["transcribe:3"]
         max_concurrent_tasks: 1
         type: "worker"
   ```

2. **Update PostgreSQL configuration**:
   ```bash
   python src/database/pg_worker_manager.py --update
   ```

That's it! The script will automatically:
- Extract the new worker's IP addresses
- Add them to `pg_hba.conf` with proper authentication
- Reload PostgreSQL to apply the changes

## Configuration Details

The script uses these settings from `config.yaml`:
- `database.database`: Database name (default: `av_content`)
- `database.user`: Database user (default: `signal4`)
- `processing.workers`: Worker configuration with IP addresses

Authentication method used: `scram-sha-256`

## File Locations

- PostgreSQL config: `/opt/homebrew/var/postgresql@15/postgresql.conf`
- Access control: `/opt/homebrew/var/postgresql@15/pg_hba.conf`
- Backups: `/opt/homebrew/var/postgresql@15/pg_hba.conf.backup`

## Troubleshooting

### Permission Errors
If you get permission errors, make sure you can write to the PostgreSQL configuration directory:
```bash
sudo chown -R $(whoami) /opt/homebrew/var/postgresql@15/
```

### Manual PostgreSQL Restart
If automatic reload fails, restart PostgreSQL manually:
```bash
brew services restart postgresql@15
```

### Verify Changes
After updating, verify that workers can connect:
```bash
psql -h 10.0.0.22 -U signal4 -d av_content -c "SELECT version();"
```

## Integration with Existing Scripts

You can integrate this into your existing worker management scripts:

```python
from src.database.pg_worker_manager import PostgreSQLWorkerManager

# Update PostgreSQL configuration when workers change
manager = PostgreSQLWorkerManager()
changes_made = manager.update_pg_hba()

if changes_made:
    manager.reload_postgresql()
```