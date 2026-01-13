#!/usr/bin/env python3
"""
PostgreSQL Worker Manager

This script manages PostgreSQL pg_hba.conf entries for worker nodes.
It reads worker IP addresses from config.yaml and updates the PostgreSQL
configuration to allow connections from all workers.

Usage:
    python src/database/pg_worker_manager.py --update
    python src/database/pg_worker_manager.py --check
    python src/database/pg_worker_manager.py --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


class PostgreSQLWorkerManager:
    """Manages PostgreSQL configuration for worker nodes."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.pg_hba_path = Path("/opt/homebrew/var/postgresql@15/pg_hba.conf")
        self.pg_config_path = Path("/opt/homebrew/var/postgresql@15/postgresql.conf")
        
        # Load configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from config.yaml."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_worker_ips(self) -> List[Tuple[str, str, str]]:
        """
        Extract worker IP addresses from config.yaml.
        
        Returns:
            List of tuples (worker_name, ip_address, interface_type)
        """
        worker_ips = []
        
        if 'processing' not in self.config or 'workers' not in self.config['processing']:
            raise ValueError("No workers configuration found in config.yaml")
            
        workers = self.config['processing']['workers']
        
        for worker_name, worker_config in workers.items():
            if not worker_config.get('enabled', False):
                continue
                
            # Add ethernet IP if present
            if 'eth' in worker_config:
                worker_ips.append((worker_name, worker_config['eth'], 'eth'))
                
            # Add wifi IP if present
            if 'wifi' in worker_config:
                worker_ips.append((worker_name, worker_config['wifi'], 'wifi'))
                
        return worker_ips
    
    def get_database_config(self) -> Dict:
        """Get database configuration from config.yaml."""
        if 'database' not in self.config:
            raise ValueError("No database configuration found in config.yaml")
            
        return self.config['database']
    
    def backup_pg_hba(self) -> Path:
        """Create a backup of pg_hba.conf."""
        backup_path = self.pg_hba_path.with_suffix('.conf.backup')
        shutil.copy2(self.pg_hba_path, backup_path)
        print(f"Created backup: {backup_path}")
        return backup_path
    
    def generate_pg_hba_entries(self) -> List[str]:
        """Generate pg_hba.conf entries for all workers."""
        worker_ips = self.get_worker_ips()
        db_config = self.get_database_config()
        
        database_name = db_config.get('database', 'av_content')
        username = db_config.get('user', 'signal4')
        auth_method = 'scram-sha-256'
        
        entries = []
        entries.append("# Allow connections from all workers")
        
        # Group by worker for cleaner output
        current_worker = None
        for worker_name, ip_address, interface_type in worker_ips:
            if worker_name != current_worker:
                entries.append(f"# {worker_name.capitalize()}")
                current_worker = worker_name
                
            entry = f"host    {database_name:<15} {username:<15} {ip_address}/32{' ':<10} {auth_method}"
            entries.append(entry)
            
        return entries
    
    def update_pg_hba(self, dry_run: bool = False) -> bool:
        """
        Update pg_hba.conf with current worker IP addresses.
        
        Args:
            dry_run: If True, only show what would be changed without making changes
            
        Returns:
            True if changes were made (or would be made), False otherwise
        """
        if not self.pg_hba_path.exists():
            raise FileNotFoundError(f"pg_hba.conf not found at {self.pg_hba_path}")
            
        # Read current pg_hba.conf
        with open(self.pg_hba_path, 'r') as f:
            lines = f.readlines()
        
        # Find the worker section
        worker_section_start = None
        worker_section_end = None
        
        for i, line in enumerate(lines):
            if line.strip() == "# Allow connections from all workers":
                worker_section_start = i
            elif worker_section_start is not None and line.strip() == "":
                # Empty line after worker section
                worker_section_end = i
                break
        
        if worker_section_start is None:
            # No existing worker section, add at the end
            new_entries = self.generate_pg_hba_entries()
            lines.extend(["\n"] + [entry + "\n" for entry in new_entries] + ["\n"])
            changes_made = True
        else:
            # Replace existing worker section
            if worker_section_end is None:
                worker_section_end = len(lines)
            
            new_entries = self.generate_pg_hba_entries()
            new_lines = (
                lines[:worker_section_start] +
                [entry + "\n" for entry in new_entries] +
                ["\n"] +
                lines[worker_section_end:]
            )
            
            changes_made = lines != new_lines
            lines = new_lines
        
        if dry_run:
            if changes_made:
                print("Changes that would be made to pg_hba.conf:")
                print("=" * 50)
                for entry in self.generate_pg_hba_entries():
                    print(entry)
                print("=" * 50)
            else:
                print("No changes needed to pg_hba.conf")
            return changes_made
        
        if changes_made:
            # Create backup before making changes
            self.backup_pg_hba()
            
            # Write updated configuration
            with open(self.pg_hba_path, 'w') as f:
                f.writelines(lines)
                
            print(f"Updated pg_hba.conf with {len(self.get_worker_ips())} worker IP addresses")
            return True
        else:
            print("No changes needed to pg_hba.conf")
            return False
    
    def check_listen_addresses(self) -> bool:
        """Check if PostgreSQL is configured to listen on all addresses."""
        if not self.pg_config_path.exists():
            raise FileNotFoundError(f"postgresql.conf not found at {self.pg_config_path}")
            
        with open(self.pg_config_path, 'r') as f:
            content = f.read()
            
        # Check for listen_addresses = '*'
        if "listen_addresses = '*'" in content:
            print("✓ PostgreSQL is configured to listen on all addresses")
            return True
        else:
            print("✗ PostgreSQL is not configured to listen on all addresses")
            print("  Add 'listen_addresses = \"*\"' to postgresql.conf")
            return False
    
    def reload_postgresql(self) -> bool:
        """Reload PostgreSQL configuration."""
        try:
            # Try using brew services first
            result = subprocess.run(
                ["brew", "services", "reload", "postgresql@15"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✓ PostgreSQL configuration reloaded successfully")
                return True
            else:
                # Try pg_ctl reload as fallback
                result = subprocess.run(
                    ["pg_ctl", "reload", "-D", "/opt/homebrew/var/postgresql@15"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("✓ PostgreSQL configuration reloaded successfully")
                    return True
                else:
                    print(f"✗ Failed to reload PostgreSQL configuration: {result.stderr}")
                    return False
                    
        except subprocess.TimeoutExpired:
            print("✗ Timeout while reloading PostgreSQL configuration")
            return False
        except Exception as e:
            print(f"✗ Error reloading PostgreSQL configuration: {e}")
            return False
    
    def check_configuration(self) -> bool:
        """Check current PostgreSQL configuration for workers."""
        print("Checking PostgreSQL configuration for workers...")
        print("=" * 50)
        
        # Check worker IPs
        worker_ips = self.get_worker_ips()
        print(f"Found {len(worker_ips)} worker IP addresses in config.yaml:")
        for worker_name, ip, interface in worker_ips:
            print(f"  {worker_name} ({interface}): {ip}")
        
        print()
        
        # Check listen_addresses
        listen_ok = self.check_listen_addresses()
        
        print()
        
        # Check pg_hba.conf entries
        print("Current pg_hba.conf worker entries:")
        if self.pg_hba_path.exists():
            with open(self.pg_hba_path, 'r') as f:
                lines = f.readlines()
                
            in_worker_section = False
            found_entries = 0
            
            for line in lines:
                if line.strip() == "# Allow connections from all workers":
                    in_worker_section = True
                    continue
                elif in_worker_section and line.strip() == "":
                    break
                elif in_worker_section and line.strip().startswith("host"):
                    print(f"  {line.strip()}")
                    found_entries += 1
                elif in_worker_section and line.strip().startswith("#"):
                    print(f"  {line.strip()}")
            
            if found_entries == 0:
                print("  No worker entries found in pg_hba.conf")
        else:
            print("  pg_hba.conf not found")
        
        return listen_ok and found_entries > 0


def main():
    parser = argparse.ArgumentParser(description="Manage PostgreSQL configuration for worker nodes")
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--update", action="store_true", help="Update pg_hba.conf with current workers")
    group.add_argument("--check", action="store_true", help="Check current PostgreSQL configuration")
    group.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    
    parser.add_argument("--no-reload", action="store_true", help="Don't reload PostgreSQL after updating")
    
    args = parser.parse_args()
    
    try:
        manager = PostgreSQLWorkerManager(args.config)
        
        if args.check:
            success = manager.check_configuration()
            sys.exit(0 if success else 1)
            
        elif args.dry_run:
            changes_needed = manager.update_pg_hba(dry_run=True)
            if changes_needed:
                print("\nRun with --update to apply these changes")
            sys.exit(0)
            
        elif args.update:
            changes_made = manager.update_pg_hba(dry_run=False)
            
            if changes_made and not args.no_reload:
                print("\nReloading PostgreSQL configuration...")
                reload_success = manager.reload_postgresql()
                if not reload_success:
                    print("Warning: Failed to reload PostgreSQL configuration automatically")
                    print("You may need to restart PostgreSQL manually:")
                    print("  brew services restart postgresql@15")
            
            sys.exit(0)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()