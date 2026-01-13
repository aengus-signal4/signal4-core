#!/usr/bin/env python3
import os
import json
import logging
import tempfile
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, Optional, List
import browser_cookie3
import yaml
from datetime import datetime, timezone
import argparse
import sys

# Add project root to Python path if run as script
# This ensures config can be found relative to project root
project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Basic logging setup when run as script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cookie_manager_script")

class CookieManager:
    """Manages multiple YouTube account cookies with consistent browser profiles"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.cookie_dir = Path(self.config.get('cookie_dir', 'data/cookies'))
        self.cookie_dir.mkdir(parents=True, exist_ok=True)
        self.current_account_index = 0
        self.accounts = self._load_accounts()
        logger.info(f"Initialized CookieManager with {len(self.accounts)} accounts")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                return config.get('processing', {}).get('youtube_downloader', {})
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return {}
            
    def _load_accounts(self) -> List[Dict]:
        """Load account configurations"""
        accounts = []
        # Load from cookie_behavior.profiles structure
        cookie_behavior = self.config.get('cookie_behavior', {})
        accounts_config = cookie_behavior.get('profiles', [])
        
        for account in accounts_config:
            account_info = {
                'name': account.get('name', 'default'),
                'profile_path': account.get('profile_path'),
                'cookie_file': self.cookie_dir / f"{account['name']}_cookies.txt",
                'last_used': None,
                'use_count': 0
            }
            accounts.append(account_info)
            
        return accounts
        
    def _extract_cookies(self, profile_path: str, cookie_file: Path) -> bool:
        """Extract cookies from Firefox profile and save to file"""
        try:
            # Get cookies from Firefox profile
            # browser_cookie3 now uses a different method to specify profile
            os.environ['FIREFOX_PROFILE'] = profile_path
            cookies = browser_cookie3.firefox(domain_name='.youtube.com')
            
            # Write cookies to file in Netscape format
            with open(cookie_file, 'w') as f:
                f.write("# Netscape HTTP Cookie File\n")
                for cookie in cookies:
                    if cookie.domain.endswith('.youtube.com'):
                        secure = "TRUE" if cookie.secure else "FALSE"
                        # HttpOnly isn't standard Netscape format, so we omit it.
                        # The standard format is: domain flag path secure expires name value
                        # Set expires to 0 if it's None (session cookie)
                        expires_timestamp = int(cookie.expires) if cookie.expires is not None else 0
                        f.write(f"{cookie.domain}\tTRUE\t{cookie.path}\t{secure}\t{expires_timestamp}\t{cookie.name}\t{cookie.value}\n")
            
            logger.info(f"Successfully extracted cookies to {cookie_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract cookies: {str(e)}")
            return False
            
    def _update_account_usage(self, account: Dict):
        """Update account usage statistics"""
        account['last_used'] = datetime.now(timezone.utc)
        account['use_count'] += 1
        
        # Save usage stats
        stats_file = self.cookie_dir / f"{account['name']}_stats.json"
        stats = {
            'last_used': account['last_used'].isoformat(),
            'use_count': account['use_count']
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
    def get_next_cookie_file(self) -> Optional[str]:
        """Get the next cookie file to use, rotating through accounts"""
        if not self.accounts:
            logger.error("No accounts configured")
            return None
            
        # Get next account
        account = self.accounts[self.current_account_index]
        self.current_account_index = (self.current_account_index + 1) % len(self.accounts)
        
        # Check if cookie file exists and is recent
        if not account['cookie_file'].exists():
            logger.info(f"Cookie file not found for {account['name']}, extracting from profile")
            if not self._extract_cookies(account['profile_path'], account['cookie_file']):
                return None
                
        # Update usage stats
        self._update_account_usage(account)
        
        return str(account['cookie_file'])
        
    def refresh_cookies(self, account_name: Optional[str] = None) -> bool:
        """Refresh cookies for specified account or all accounts"""
        if account_name:
            accounts = [a for a in self.accounts if a['name'] == account_name]
        else:
            accounts = self.accounts
            
        success = True
        for account in accounts:
            logger.info(f"Refreshing cookies for account {account['name']}")
            if not self._extract_cookies(account['profile_path'], account['cookie_file']):
                success = False
                
        return success
        
    def get_account_stats(self) -> Dict:
        """Get usage statistics for all accounts"""
        stats = {}
        for account in self.accounts:
            stats[account['name']] = {
                'last_used': account['last_used'].isoformat() if account['last_used'] else None,
                'use_count': account['use_count']
            }
        return stats 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and refresh YouTube cookies from Firefox profiles.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the configuration file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--account", help="Name of the specific account to refresh cookies for.")
    group.add_argument("--all", action="store_true", help="Refresh cookies for all accounts in the config.")

    args = parser.parse_args()

    logger.info(f"Running Cookie Manager script with config: {args.config}")
    # Ensure Firefox is closed before running
    logger.warning("Please ensure Firefox is completely closed before proceeding.")
    input("Press Enter to continue after closing Firefox...")

    try:
        manager = CookieManager(config_path=args.config)
        
        account_to_refresh = None
        if args.account:
            account_to_refresh = args.account
            logger.info(f"Attempting to refresh cookies for account: {account_to_refresh}")
        elif args.all:
            logger.info("Attempting to refresh cookies for ALL accounts.")

        success = manager.refresh_cookies(account_name=account_to_refresh)

        if success:
            logger.info("Cookie refresh process completed successfully.")
        else:
            logger.error("Cookie refresh process failed for one or more accounts. Check logs above.")
            sys.exit(1) # Exit with error code
            
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        sys.exit(1) 