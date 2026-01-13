#!/usr/bin/env python3
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import logging
import yaml
from cookie_manager import CookieManager

# Add project root to Python path
sys.path.append(str(get_project_root()))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize cookie manager
    config_path = get_config_path()
    cookie_manager = CookieManager(str(config_path))
    
    # Get cookie file
    cookie_file = cookie_manager.get_next_cookie_file()
    if cookie_file:
        logger.info(f"Successfully extracted cookies to: {cookie_file}")
        
        # Read and display cookie file contents (excluding sensitive data)
        with open(cookie_file) as f:
            lines = f.readlines()
            logger.info(f"Found {len(lines)-1} cookies (excluding header line)")
            for line in lines[1:]:  # Skip header line
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    domain, path, name = parts[0], parts[2], parts[5]
                    logger.info(f"Cookie: {domain} {path} {name}")
    else:
        logger.error("Failed to extract cookies")

if __name__ == "__main__":
    main() 