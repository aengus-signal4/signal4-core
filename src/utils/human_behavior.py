#!/usr/bin/env python3
import asyncio
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set, List, Callable, Any, Tuple
import logging
import json
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import subprocess
import os
import sys
from functools import wraps
import aiohttp
from dataclasses import dataclass, field
import yaml

# Add project root to Python path if necessary (adjust path as needed)
project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import setup_worker_logger
logger = setup_worker_logger('human_behavior')

@dataclass
class WorkerBehaviorState:
    """Tracks behavior state for a single worker"""
    worker_id: str
    daily_task_count: int = 0  # Total daily count (backward compatibility)
    daily_task_counts_by_type: Dict[str, int] = field(default_factory=dict)  # Per-type counters
    last_task_time: Optional[datetime] = None
    last_task_time_by_type: Dict[str, Optional[datetime]] = field(default_factory=dict)  # Per-type last task times
    current_session_start: Optional[datetime] = None
    current_session_start_by_type: Dict[str, Optional[datetime]] = field(default_factory=dict)  # Per-type session starts
    break_end_time: Optional[datetime] = None
    break_end_time_by_type: Dict[str, Optional[datetime]] = field(default_factory=dict)  # Per-type break times
    consecutive_blocked: int = 0
    last_dns_flush: Optional[datetime] = None
    last_user_agent_change: Optional[datetime] = None
    current_user_agent: Optional[str] = None
    current_cookie_profile: Optional[str] = None  # Track current cookie profile
    last_cookie_change: Optional[datetime] = None  # Track when cookies were last changed
    task_history: List[Dict] = field(default_factory=list)
    current_session_target_duration: Optional[float] = None # Target duration for the ongoing session
    current_session_target_duration_by_type: Dict[str, Optional[float]] = field(default_factory=dict)  # Per-type session durations
    
    # Default configurations matching config.yaml
    daily_limits: Dict = field(default_factory=lambda: {
        'min_downloads': 25,  # Minimum downloads per day
        'max_downloads': 35,  # Maximum downloads per day
        'target_downloads': 300  # Target downloads per day
    })
    
    operating_hours: Dict = field(default_factory=lambda: {
        'start_hour_min': 8,  # Start between 8-10 AM
        'start_hour_max': 10,
        'start_minute_min': random.randint(0, 15), # Randomize minutes
        'start_minute_max': random.randint(45, 59),
        'end_hour_min': 18,  # End between 6-8 PM
        'end_hour_max': 20,
        'end_minute_min': random.randint(0, 15), # Randomize minutes
        'end_minute_max': random.randint(45, 59)
    })
    
    session_management: Dict = field(default_factory=lambda: {
        'session_duration_min': 3600,  # Default min session 1 hour
        'session_duration_max': 7200,   # Default max session 2 hours
        'break_duration_min': 1800,  # 30 min breaks
        'break_duration_max': 1800,  # 30 min breaks
    })
    
    watch_time: Dict = field(default_factory=lambda: {
        'min_duration': 50,  # 50 second default wait time
        'max_duration': 50,  # 50 second default wait time
        'chunk_size': 30,  # 30 second chunks
        'min_pause': 2,  # 2 second minimum pause
        'max_pause': 5,  # 5 second maximum pause
        'variation_factor': 0.2  # 20% variation
    })
    
    network_settings: Dict = field(default_factory=lambda: {
        'dns_flush_interval': 1800,  # Flush DNS every 30 minutes
        'user_agent_change_interval': 900,  # Change user agent every 15 minutes
        'max_consecutive_failures': 1,  # More conservative failure handling
        'failure_cooldown': 900  # 15 minute cooldown after failures
    })
    
    # Per-type behavior configurations (can be different for youtube vs rumble)
    behavior_config_by_type: Dict[str, Dict] = field(default_factory=dict)

class HumanBehaviorManager:
    """Manages human-like behavior patterns across workers"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.worker_states: Dict[str, WorkerBehaviorState] = {}
        self.config = self._load_config(config_path)
        
        # Load simplified behavior configurations
        downloader_config = self.config.get('processing', {}).get('downloader', {})
        behavior_config = downloader_config.get('behavior', {})
        
        # Build behavior configs from simplified structure
        self.cookie_behavior_config = {
            'daily_limits': {'target_downloads': behavior_config.get('youtube', {}).get('target_downloads_per_day', 300)},
            'estimated_task_execution_time': behavior_config.get('youtube', {}).get('estimated_task_time', 50),
            'network_settings': {
                'max_consecutive_failures': behavior_config.get('youtube', {}).get('max_consecutive_failures', 3),
                'failure_cooldown': behavior_config.get('youtube', {}).get('failure_cooldown', 600)
            }
        }
        self.proxy_behavior_config = {
            'daily_target_downloads': behavior_config.get('proxy', {}).get('total_target_downloads_per_day', 1000),
            'num_proxies': behavior_config.get('proxy', {}).get('num_proxies', 20),
            'estimated_task_execution_time': behavior_config.get('proxy', {}).get('estimated_task_time', 60),
            'network_settings': {
                'max_consecutive_failures': behavior_config.get('proxy', {}).get('max_consecutive_failures', 3),
                'failure_cooldown': behavior_config.get('proxy', {}).get('failure_cooldown', 600)
            }
        }
        self.rumble_behavior_config = {
            'daily_limits': {'target_downloads': behavior_config.get('rumble', {}).get('target_downloads_per_day', 100)},
            'estimated_task_execution_time': behavior_config.get('rumble', {}).get('estimated_task_time', 40),
            'network_settings': {
                'max_consecutive_failures': behavior_config.get('rumble', {}).get('max_consecutive_failures', 2),
                'failure_cooldown': behavior_config.get('rumble', {}).get('failure_cooldown', 1800)
            }
        }
        
        # Load user agents (prioritize proxy-specific, then default)
        self.user_agents = self._load_user_agents() 
        
        if not self.cookie_behavior_config:
            logger.warning("Cookie downloader behavior config ('processing.youtube_downloader.cookie_behavior') not found or empty. Calculations will use defaults.")
        if not self.proxy_behavior_config:
            logger.warning("Proxy downloader behavior config ('processing.youtube_downloader.proxy_behavior') not found or empty. Proxy mode calculations might fail or use defaults.")
        if not self.rumble_behavior_config:
            logger.warning("Rumble behavior config ('processing.youtube_downloader.rumble_behavior') not found or empty. Rumble downloads will use defaults.")
            
        # Get overall target for proxy mode from the new path
        self.proxy_daily_target = self.proxy_behavior_config.get('daily_target_downloads', 0)
        self.num_proxies = self.proxy_behavior_config.get('num_proxies', 1) # Default to 1
        
        logger.debug(f"Initialized HumanBehaviorManager with {len(self.user_agents)} user agents.")
        logger.debug(f"Cookie behavior config loaded: {'Yes' if self.cookie_behavior_config else 'No'}")
        logger.debug(f"Proxy behavior config loaded: {'Yes' if self.proxy_behavior_config else 'No'} (Target: {self.proxy_daily_target}, Proxies: {self.num_proxies})")
        logger.debug(f"Rumble behavior config loaded: {'Yes' if self.rumble_behavior_config else 'No'}")

        # Load available cookie profiles (still relevant for cookie mode)
        self.cookie_profiles = self._load_cookie_profiles()
        logger.debug(f"Loaded {len(self.cookie_profiles)} cookie profiles")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {}
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded behavior config from {config_path}")
            return config
            
    def _load_user_agents(self) -> List[str]:
        """Load user agents from config or use defaults"""
        default_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) Gecko/20100101 Firefox/138.0'
        ]
        
        # Load from simplified config structure
        downloader_config = self.config.get('processing', {}).get('downloader', {})
        agents = downloader_config.get('user_agents', default_agents)
        
        logger.debug(f"Loaded {len(agents)} user agents")
        return agents
        
    def _load_cookie_profiles(self) -> List[str]:
        """Load available cookie profiles from config"""
        try:
            # Load from youtube_downloader cookie_behavior config structure
            youtube_config = self.config.get('processing', {}).get('youtube_downloader', {})
            cookie_behavior = youtube_config.get('cookie_behavior', {})
            profiles_config = cookie_behavior.get('profiles', [])
            
            profiles = [profile['name'] for profile in profiles_config if 'name' in profile]
            logger.info(f"Loaded {len(profiles)} cookie profiles from config: {profiles}")
            return profiles
        except Exception as e:
            logger.error(f"Failed to load cookie profiles: {str(e)}")
            return []
        
    def _calculate_behavior_parameters(self, worker_id: str, mode: str, total_proxy_workers: int) -> Dict:
        """Calculate behavior parameters based on worker mode ('cookie' or 'proxy')."""
        logger.debug(f"Calculating behavior for worker {worker_id} in '{mode}' mode.")
        
        if mode == 'cookie':
            # Use cookie config directly - assume it contains all necessary sections
            config = self.cookie_behavior_config
            if not config:
                logger.error(f"Cannot calculate behavior for {worker_id} in cookie mode: Cookie config is missing or empty.")
                return {} # Return empty dict, worker will use defaults
            
            # --- DERIVE Cookie Mode Parameters from Target --- #
            target = config.get('daily_limits', {}).get('target_downloads', 300) # Default to 300 if not specified
            if target <= 0:
                logger.warning(f"Cookie mode target for {worker_id} is {target}. Using default 300.")
                target = 300
            logger.debug(f"Cookie worker {worker_id} target: {target}")
            
            # Daily Limits (derived from target)
            daily_limits = {
                'min_downloads': max(1, int(target * 0.85)), # Slightly tighter range than proxy
                'max_downloads': int(target * 1.15),
                'target_downloads': target
            }
            
            # Operating Hours (Derive defaults - Aim for ~13 hours max span)
            start_h_min = random.randint(6, 7) # Start between 6-8 AM
            start_h_max = start_h_min + 1 
            end_h_max = random.randint(start_h_min + 11, start_h_min + 13) # End 11-13 hours after earliest start
            end_h_min = end_h_max - 1
            # Ensure hours wrap around correctly if needed (though unlikely with these ranges)
            end_h_min %= 24
            end_h_max %= 24
            
            operating_hours = {
                'start_hour_min': start_h_min, 
                'start_hour_max': start_h_max,
                'start_minute_min': random.randint(0, 15), # Randomize minutes
                'start_minute_max': random.randint(45, 59),
                'end_hour_min': end_h_min, 
                'end_hour_max': end_h_max,
                'end_minute_min': random.randint(0, 15), # Randomize minutes
                'end_minute_max': random.randint(45, 59)
            }

            # Session Management (Derive defaults)
            start_h = operating_hours['start_hour_min']
            end_h = operating_hours['end_hour_max']
            approx_operating_hours = (end_h - start_h + 24) % 24
            if approx_operating_hours == 0: approx_operating_hours = 13 # Default if calc fails
            total_operating_seconds = approx_operating_hours * 3600

            # Choose number of sessions (e.g., 3-5)
            num_sessions = random.randint(3, 5)
            videos_per_session = target // num_sessions
            extra_videos = target % num_sessions
            
            # Set break duration RANGE directly (1-2 hours)
            break_duration_min_seconds = 3600 # 1 hour
            break_duration_max_seconds = 7200 # 2 hours
            # Calculate total break time estimate using the average of the range for session planning
            avg_break_duration = (break_duration_min_seconds + break_duration_max_seconds) / 2
            total_break_time = avg_break_duration * max(0, num_sessions - 1) 
            buffer_time = random.randint(1200, 2400)
            available_session_time = max(0, total_operating_seconds - total_break_time - buffer_time)
            
            # Set session duration RANGE directly (2-3 hours)
            session_duration_min_seconds = 7200 # 2 hours
            session_duration_max_seconds = 10800 # 3 hours
            
            session_management = {
                'session_duration_min': session_duration_min_seconds, # Store range
                'session_duration_max': session_duration_max_seconds,
                'break_duration_min': break_duration_min_seconds, # Store range now
                'break_duration_max': break_duration_max_seconds,
                'videos_per_session': videos_per_session,
                'extra_videos': extra_videos
            }

            # Watch Time / Request Timing (Derive wait time after accounting for task execution)
            estimated_task_time = self.cookie_behavior_config.get('estimated_task_execution_time', 50)
            avg_time_per_request = available_session_time / target if target > 0 else 30 # Default 30s
            # Calculate the target average *wait* time
            target_avg_wait_time = max(5.0, avg_time_per_request - estimated_task_time)
            logger.debug(f"Cookie Worker {worker_id}: AvgTimeSlot={avg_time_per_request:.1f}s, EstTask={estimated_task_time}s => TargetAvgWait={target_avg_wait_time:.1f}s")
            
            # Derive min/max wait times based on the target average wait time
            min_wait_time = int(target_avg_wait_time * 0.8) # e.g., 70%
            max_wait_time = int(target_avg_wait_time * 1.1) # e.g., 130%
            
            actual_min_duration = max(20, min_wait_time)  # Ensure minimum 20 seconds wait
            actual_max_duration = max(actual_min_duration + 10, max_wait_time)  # Ensure max > min + buffer
            
            watch_time = {
                'min_duration': actual_min_duration,
                'max_duration': actual_max_duration,
                'chunk_size': 30,
                'min_pause': 1,
                'max_pause': 5,
                'variation_factor': random.uniform(0.2, 0.4) # Slightly more variation
            }

            # Network Settings (Use defaults - can be tuned)
            network_settings = {
                'dns_flush_interval': random.randint(1800, 3600), # 30-60 min
                'user_agent_change_interval': random.randint(900, 1800), # 15-30 min
                'max_consecutive_failures': self.cookie_behavior_config.get('network_settings', {}).get('max_consecutive_failures', 3), # Read from cookie_behavior config
                'failure_cooldown': self.cookie_behavior_config.get('network_settings', {}).get('failure_cooldown', 600) # Read from cookie_behavior config
            }
            # --- End Parameter Derivation --- 
            
            logger.debug(f"Derived behavior parameters for cookie mode worker {worker_id}.")
            return {
                'daily_limits': daily_limits,
                'operating_hours': operating_hours,
                'session_management': session_management,
                'watch_time': watch_time,
                'network_settings': network_settings
            }

        elif mode == 'proxy':
            # Calculate proxy behavior based on shared target
            if not self.proxy_behavior_config or self.proxy_daily_target <= 0 or total_proxy_workers <= 0:
                logger.error(f"Cannot calculate behavior for {worker_id} in proxy mode: Proxy config/target missing, invalid, or no proxy workers.")
                return {} # Return empty dict, worker will use defaults
                
            # Calculate base target per proxy worker
            base_target = self.proxy_daily_target // total_proxy_workers
            remainder = self.proxy_daily_target % total_proxy_workers
            
            # Determine worker index for distributing remainder (use hash for robustness)
            try:
                worker_index = int(''.join(filter(str.isdigit, worker_id)))
            except (ValueError, IndexError):
                worker_index = hash(worker_id) % total_proxy_workers
            
            target = base_target + 1 if worker_index < remainder else base_target
            logger.debug(f"Proxy worker {worker_id} target calculated: {target} (Base: {base_target}, Total: {self.proxy_daily_target}, Workers: {total_proxy_workers})")

            # --- Parameter Derivation for Proxy Mode --- 
            # Use target to derive session, timing, etc. (similar to old logic)
            # If specific overrides exist in proxy_behavior_config, use them.
            proxy_overrides = self.proxy_behavior_config # Already loaded with the correct path
            
            # Daily Limits (derived from target)
            daily_limits = {
                'min_downloads': max(1, int(target * 0.8)),
                'max_downloads': int(target * 1.2),
                'target_downloads': target
            }
            
            # Operating Hours (Use override or derive defaults)
            operating_hours = proxy_overrides.get('operating_hours', {
                'start_hour_min': random.randint(5, 7), # Proxies might start earlier/run longer
                'start_hour_max': random.randint(8, 9),
                'start_minute_min': random.randint(0, 15), # Randomize minutes
                'start_minute_max': random.randint(45, 59),
                'end_hour_min': random.randint(20, 21),
                'end_hour_max': random.randint(22, 23),
                'end_minute_min': random.randint(0, 15), # Randomize minutes
                'end_minute_max': random.randint(45, 59)
            })

            # Session Management (Use override or derive defaults)
            # Estimate total operating time based on derived/override hours
            start_h = operating_hours['start_hour_min']
            end_h = operating_hours['end_hour_max']
            # Calculate approximate total operating hours (handle day wrap around if needed, simplified here)
            approx_operating_hours = (end_h - start_h + 24) % 24 
            if approx_operating_hours == 0: approx_operating_hours = 12 # Assume default if calculation seems off
            total_operating_seconds = approx_operating_hours * 3600
            
            # Estimate number of sessions needed based on target (e.g., 50 videos per session?)
            target_per_session = 50 
            num_sessions = max(1, (target + target_per_session - 1) // target_per_session) # Ceiling division
            videos_per_session = target // num_sessions
            extra_videos = target % num_sessions
            
            # Determine break duration RANGE (e.g., 15-30 min for proxies)
            break_duration_min_seconds = proxy_overrides.get('session_management', {}).get('break_duration_min', 900) 
            break_duration_max_seconds = proxy_overrides.get('session_management', {}).get('break_duration_max', 1800)
            avg_break_duration = (break_duration_min_seconds + break_duration_max_seconds) / 2
            total_break_time = avg_break_duration * max(0, num_sessions - 1)
            buffer_time = random.randint(900, 1800) # Buffer for the day
            available_session_time = max(0, total_operating_seconds - total_break_time - buffer_time)
            avg_session_duration = available_session_time / num_sessions if num_sessions > 0 else 3600
            # Ensure avg session duration is reasonable for proxies (e.g., 0.75 - 2 hours)
            avg_session_duration = max(2700, min(avg_session_duration, 7200))
            
            # Define session duration range around the calculated average (e.g., +/- 20 mins)
            session_duration_min_seconds = max(1800, int(avg_session_duration - 1200))
            session_duration_max_seconds = int(avg_session_duration + 1200)
            
            session_management = proxy_overrides.get('session_management', {
                'session_duration_min': session_duration_min_seconds, # Store range
                'session_duration_max': session_duration_max_seconds,
                'break_duration_min': break_duration_min_seconds, # Store range
                'break_duration_max': break_duration_max_seconds,
                'videos_per_session': videos_per_session,
                'extra_videos': extra_videos
            })

            # Watch Time / Request Timing (Derive wait time after accounting for task execution)
            estimated_task_time = self.proxy_behavior_config.get('estimated_task_execution_time', 60)
            # Estimate avg time per request based on target and operating hours
            avg_time_per_request = available_session_time / target if target > 0 else 60 # Default 60s if target is 0
            # Calculate the target average *wait* time
            target_avg_wait_time = max(5.0, avg_time_per_request - estimated_task_time)
            
            logger.debug(f"Proxy Worker {worker_id}: AvgTimeSlot={avg_time_per_request:.1f}s, EstTask={estimated_task_time}s => TargetAvgWait={target_avg_wait_time:.1f}s")
            
            # Derive min/max wait times based on the target average wait time
            min_wait_time = int(target_avg_wait_time * 0.8) # e.g., 80%
            max_wait_time = int(target_avg_wait_time * 1.2) # e.g., 120%
            
            actual_min_duration = max(20, min_wait_time)  # Ensure minimum 20 seconds wait
            actual_max_duration = max(actual_min_duration + 10, max_wait_time)  # Ensure max > min + buffer
            
            watch_time = proxy_overrides.get('watch_time', {
                'min_duration': actual_min_duration,
                'max_duration': actual_max_duration,
                'chunk_size': 30, # Keep chunk/pause simple for proxies
                'min_pause': 1,
                'max_pause': 3,
                'variation_factor': random.uniform(0.1, 0.3)
            })

            # Network Settings (Use override or derive defaults)
            network_settings = proxy_overrides.get('network_settings', {
                'dns_flush_interval': random.randint(1800, 7200), # Flush less often? (30-120 min)
                'user_agent_change_interval': random.randint(1800, 3600), # Change less often? (30-60 min)
                'max_consecutive_failures': proxy_overrides.get('network_settings', {}).get('max_consecutive_failures', 3),
                'failure_cooldown': proxy_overrides.get('network_settings', {}).get('failure_cooldown', random.randint(300, 600))
            })
            # --- End Parameter Derivation --- 

            return {
                'daily_limits': daily_limits,
                'operating_hours': operating_hours,
                'session_management': session_management,
                'watch_time': watch_time,
                'network_settings': network_settings
            }
        
        elif mode == 'rumble':
            # Calculate rumble behavior dynamically (similar to cookie mode)
            if not self.rumble_behavior_config:
                logger.error(f"Cannot calculate behavior for {worker_id} in rumble mode: Rumble config missing.")
                return {} # Return empty dict
                
            config = self.rumble_behavior_config
            target = config.get('daily_limits', {}).get('target_downloads', 100)
            if target <= 0:
                logger.warning(f"Rumble mode target for {worker_id} is {target}. Using default 100.")
                target = 100
            logger.debug(f"Rumble worker {worker_id} target: {target}")
            
            # Daily Limits (with slight variation)
            daily_limits = {
                'min_downloads': max(1, int(target * 0.8)),
                'max_downloads': int(target * 1.2),
                'target_downloads': target
            }
            
            # Operating Hours (randomized start/end)
            operating_hours = {
                'start_hour_min': 7,
                'start_hour_max': 9,
                'start_minute_min': random.randint(0, 15),
                'start_minute_max': random.randint(45, 59),
                'end_hour_min': 21,
                'end_hour_max': 22,
                'end_minute_min': random.randint(0, 15),
                'end_minute_max': random.randint(45, 59)
            }
            
            # Determine operating hours
            actual_start_hour = random.randint(operating_hours['start_hour_min'], operating_hours['start_hour_max'])
            actual_start_minute = random.randint(operating_hours['start_minute_min'], operating_hours['start_minute_max'])
            actual_end_hour = random.randint(operating_hours['end_hour_min'], operating_hours['end_hour_max'])
            actual_end_minute = random.randint(operating_hours['end_minute_min'], operating_hours['end_minute_max'])
            
            # Calculate total operating seconds
            total_operating_minutes = (actual_end_hour - actual_start_hour) * 60 + (actual_end_minute - actual_start_minute)
            total_operating_seconds = total_operating_minutes * 60
            
            # Calculate number of sessions and breaks
            num_sessions = max(2, min(5, target // 25))  # 2-5 sessions based on target
            
            # Determine break durations
            min_break_mins = 25
            max_break_mins = 40
            break_duration_min_seconds = min_break_mins * 60
            break_duration_max_seconds = max_break_mins * 60
            avg_break_duration = (break_duration_min_seconds + break_duration_max_seconds) / 2
            total_break_time = avg_break_duration * max(0, num_sessions - 1)
            buffer_time = random.randint(600, 1200)  # 10-20 min buffer
            
            # Calculate session durations
            available_session_time = max(0, total_operating_seconds - total_break_time - buffer_time)
            avg_session_duration = available_session_time / num_sessions if num_sessions > 0 else 3600
            avg_session_duration = max(2700, min(avg_session_duration, 7200))  # 45min-2hr
            
            session_duration_min_seconds = max(1800, int(avg_session_duration - 900))
            session_duration_max_seconds = int(avg_session_duration + 900)
            
            session_management = {
                'session_duration_min': session_duration_min_seconds,
                'session_duration_max': session_duration_max_seconds,
                'break_duration_min': break_duration_min_seconds,
                'break_duration_max': break_duration_max_seconds
            }
            
            # Calculate dynamic watch times
            estimated_task_time = self.rumble_behavior_config.get('estimated_task_execution_time', 40)
            avg_time_per_request = available_session_time / target if target > 0 else 60
            target_avg_wait_time = max(20.0, avg_time_per_request - estimated_task_time)
            
            logger.debug(f"Rumble Worker {worker_id}: AvgTimeSlot={avg_time_per_request:.1f}s, EstTask={estimated_task_time}s => TargetAvgWait={target_avg_wait_time:.1f}s")
            
            min_wait_time = int(target_avg_wait_time * 0.7)
            max_wait_time = int(target_avg_wait_time * 1.3)
            
            actual_min_duration = max(20, min_wait_time)  # Ensure minimum 20 seconds wait
            actual_max_duration = max(actual_min_duration + 10, max_wait_time)  # Ensure max > min + buffer
            
            watch_time = {
                'min_duration': actual_min_duration,
                'max_duration': actual_max_duration,
                'chunk_size': 30,
                'min_pause': 2,
                'max_pause': 5,
                'variation_factor': random.uniform(0.2, 0.4)
            }
            
            # Network Settings
            network_settings = {
                'dns_flush_interval': random.randint(1800, 3600),
                'user_agent_change_interval': random.randint(900, 1800),
                'max_consecutive_failures': self.rumble_behavior_config.get('network_settings', {}).get('max_consecutive_failures', 2),
                'failure_cooldown': self.rumble_behavior_config.get('network_settings', {}).get('failure_cooldown', 1800)
            }
            
            logger.debug(f"Dynamically calculated behavior parameters for rumble mode worker {worker_id}.")
            return {
                'daily_limits': daily_limits,
                'operating_hours': operating_hours,
                'session_management': session_management,
                'watch_time': watch_time,
                'network_settings': network_settings
            }
        
        else:
            logger.error(f"Invalid download mode '{mode}' received for worker {worker_id}. Cannot calculate parameters.")
            return {} # Return empty dict
        
    def register_worker(self, worker_id: str, config: Dict):
        """Register a new worker, reading its download_mode from config and task types."""
        if worker_id in self.worker_states:
            logger.warning(f"Worker {worker_id} already registered")
            return

        # Determine worker's download mode and task types
        download_mode = config.get('download_mode', 'cookie') # Default to cookie mode if not specified
        enabled_tasks = config.get('enabled_tasks', [])
        
        # Check if worker handles rumble downloads
        handles_rumble = any('download_rumble' in str(task) for task in enabled_tasks)
        handles_youtube = any('download_youtube' in str(task) for task in enabled_tasks)
        
        # If worker handles rumble, treat it as rumble mode for behavior
        if handles_rumble and not handles_youtube:
            # Pure rumble worker
            effective_mode = 'rumble'
        elif handles_youtube and not handles_rumble:
            # Pure youtube worker
            effective_mode = download_mode  # Use configured mode (cookie/proxy)
        elif handles_rumble and handles_youtube:
            # Mixed worker - will need per-task behavior
            effective_mode = 'mixed'
            logger.info(f"Worker {worker_id} handles both YouTube and Rumble downloads - will use per-task behavior")
        else:
            # Worker doesn't handle downloads
            effective_mode = download_mode
            
        if effective_mode not in ['cookie', 'proxy', 'rumble', 'mixed']:
            logger.warning(f"Worker {worker_id} has invalid effective_mode '{effective_mode}'. Defaulting to 'cookie'.")
            effective_mode = 'cookie'
        
        state = WorkerBehaviorState(worker_id=worker_id)
        
        # Calculate total number of *proxy* workers for parameter calculation
        # This requires knowing the mode of other workers. We estimate based on current state.
        # Note: This might be slightly inaccurate if called during initial registration loop.
        # A better approach might be to pass total_proxy_workers externally if needed precisely.
        total_proxy_workers = sum(1 for w_cfg in self.config.get('processing', {}).get('workers', {}).values() 
                                if w_cfg.get('enabled') and w_cfg.get('download_mode') == 'proxy')
        # Ensure at least 1 if current worker is proxy, even if estimation fails
        if download_mode == 'proxy' and total_proxy_workers == 0:
            total_proxy_workers = 1 
        
        # Calculate behavior parameters based on mode
        if effective_mode == 'mixed':
            # For mixed workers, calculate and store behavior for each task type
            # YouTube behavior
            youtube_params = self._calculate_behavior_parameters(worker_id, download_mode, total_proxy_workers)
            if youtube_params:
                state.behavior_config_by_type['download_youtube'] = youtube_params
                
            # Rumble behavior
            rumble_params = self._calculate_behavior_parameters(worker_id, 'rumble', 0)
            if rumble_params:
                state.behavior_config_by_type['download_rumble'] = rumble_params
                
            # Set default behavior to youtube for backward compatibility
            if youtube_params:
                state.daily_limits = youtube_params.get('daily_limits', state.daily_limits)
                state.operating_hours = youtube_params.get('operating_hours', state.operating_hours)
                state.session_management = youtube_params.get('session_management', state.session_management)
                state.watch_time = youtube_params.get('watch_time', state.watch_time)
                state.network_settings = youtube_params.get('network_settings', state.network_settings)
                
            logger.debug(f"Worker {worker_id} registered in 'mixed' mode (handles both YouTube and Rumble).")
            logger.debug(f"  YouTube daily target: {youtube_params.get('daily_limits', {}).get('target_downloads', 'N/A')} videos")
            logger.debug(f"  Rumble daily target: {rumble_params.get('daily_limits', {}).get('target_downloads', 'N/A')} videos")
        else:
            # Single mode worker
            behavior_params = self._calculate_behavior_parameters(worker_id, effective_mode, total_proxy_workers)
            
            # Apply calculated parameters only if calculation was successful
            if behavior_params:
                state.daily_limits = behavior_params.get('daily_limits', state.daily_limits) # Use calculated or default
                state.operating_hours = behavior_params.get('operating_hours', state.operating_hours)
                state.session_management = behavior_params.get('session_management', state.session_management)
                state.watch_time = behavior_params.get('watch_time', state.watch_time)
                state.network_settings = behavior_params.get('network_settings', state.network_settings)
                
                # For single-mode workers, also store in per-type config
                if handles_youtube:
                    state.behavior_config_by_type['download_youtube'] = behavior_params
                elif handles_rumble:
                    state.behavior_config_by_type['download_rumble'] = behavior_params
                
                logger.debug(f"Worker {worker_id} registered in '{effective_mode}' mode.")
                logger.debug(f"  Daily target: {state.daily_limits.get('target_downloads', 'N/A')} videos")
                logger.debug(f"  Operating hours: {state.operating_hours.get('start_hour_min', '?')}:{state.operating_hours.get('start_minute_min', '?')} - {state.operating_hours.get('end_hour_max', '?')}:{state.operating_hours.get('end_minute_max', '?')}")
                logger.debug(f"  Session duration range: {state.session_management.get('session_duration_min', 0)/60:.1f}-{state.session_management.get('session_duration_max', 0)/60:.1f} minutes")
                logger.info(f"  Break duration range: {state.session_management.get('break_duration_min', 0)/60:.1f}-{state.session_management.get('break_duration_max', 0)/60:.1f} minutes")
                logger.info(f"  Time between requests: {state.watch_time.get('min_duration', '?')}-{state.watch_time.get('max_duration', '?')} seconds")
            else:
                logger.error(f"Failed to calculate behavior parameters for worker {worker_id} ('{effective_mode}' mode). Worker will use default state values.")
                # Worker state remains with defaults from WorkerBehaviorState definition
            
        # Don't initialize session on registration - sessions will start on first task of the day
        state.current_session_start = None
        state.current_session_target_duration = None
        logger.debug(f"Worker {worker_id} registered - session will start on first task")
            
        self.worker_states[worker_id] = state
        # logger.info(f"Registered worker {worker_id} with behavior tracking") # Redundant log message
        
    def _convert_to_local_time(self, utc_time: datetime) -> datetime:
        """Convert UTC time to local time"""
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        return utc_time.astimezone()

    def _is_within_operating_hours(self, worker_id: str, current_time_override: Optional[datetime] = None) -> bool:
        """Check if current time (or override) is within worker's operating hours"""
        state = self.worker_states.get(worker_id)
        if not state:
            logger.warning(f"No state found for worker {worker_id}")
            return True
            
        now = current_time_override or datetime.now(timezone.utc)
        local_now = self._convert_to_local_time(now)
        current_hour = local_now.hour
        current_minute = local_now.minute
        
        start_h = state.operating_hours['start_hour_min']
        start_m = state.operating_hours['start_minute_min']
        end_h = state.operating_hours['end_hour_max']
        end_m = state.operating_hours['end_minute_max']
        
        # Debug log for checking times
        logger.debug(f"Worker {worker_id} Operating Hours Check: Now={current_hour:02}:{current_minute:02} local time, Start={start_h:02}:{start_m:02}, End={end_h:02}:{end_m:02}")
        
        # Check if we're before start time
        if current_hour < start_h or (current_hour == start_h and current_minute < start_m):
            logger.debug(f"Worker {worker_id} outside operating hours: too early")
            return False
            
        # Check if we're after end time
        if current_hour > end_h or (current_hour == end_h and current_minute > end_m):
            logger.debug(f"Worker {worker_id} outside operating hours: too late")
            return False
            
        logger.debug(f"Worker {worker_id} within operating hours")
        return True
    
    def _is_early_in_operating_hours(self, worker_id: str, current_time_override: Optional[datetime] = None) -> bool:
        """Check if current time is within the first hour of operating hours"""
        state = self.worker_states.get(worker_id)
        if not state:
            return False
            
        if not self._is_within_operating_hours(worker_id, current_time_override):
            return False  # Not in operating hours at all
            
        now = current_time_override or datetime.now(timezone.utc)
        local_now = self._convert_to_local_time(now)
        current_hour = local_now.hour
        current_minute = local_now.minute
        
        start_h = state.operating_hours['start_hour_min']
        start_m = state.operating_hours['start_minute_min']
        
        # Calculate time since start of operating hours
        start_today = local_now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        if local_now < start_today:
            # Before today's start time
            return False
            
        time_since_start = (local_now - start_today).total_seconds()
        is_early = time_since_start < 3600  # Within first hour (3600 seconds)
        
        logger.debug(f"Worker {worker_id} early hours check: {time_since_start:.1f}s since start, IsEarly={is_early}")
        return is_early
        
    def _check_daily_youtube_safety_limit(self, current_time_override: Optional[datetime] = None) -> bool:
        """Check if total daily YouTube downloads across all workers exceeds 375 limit"""
        from src.database.session import get_session
        from src.database.models import Content
        from sqlalchemy import func, and_
        
        try:
            now = current_time_override or datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(days=1)
            
            with get_session() as session:
                # Count YouTube downloads for today
                daily_youtube_count = session.query(func.count(Content.id)).filter(
                    and_(
                        Content.platform == 'youtube',
                        Content.is_downloaded == True,
                        Content.download_date >= today_start,
                        Content.download_date < today_end
                    )
                ).scalar() or 0
                
                safety_limit = 375
                is_safe = daily_youtube_count < safety_limit
                
                if not is_safe:
                    logger.warning(f"Daily YouTube safety limit reached: {daily_youtube_count}/{safety_limit} downloads today")
                else:
                    logger.debug(f"Daily YouTube safety check passed: {daily_youtube_count}/{safety_limit} downloads today")
                    
                return is_safe
                
        except Exception as e:
            logger.error(f"Error checking daily YouTube safety limit: {e}")
            # If we can't check, be conservative and return False
            return False

    def _has_daily_capacity(self, worker_id: str, task_type: Optional[str] = None) -> bool:
        """Check if worker has remaining daily task capacity (optionally for specific task type)"""
        state = self.worker_states.get(worker_id)
        if not state:
            logger.warning(f"No state found for worker {worker_id}")
            return True
            
        logger.debug(f"Worker {worker_id} _has_daily_capacity called with task_type: {task_type}")
        
        # Special safety check for YouTube downloads
        if task_type == 'download_youtube':
            logger.debug(f"Worker {worker_id} checking YouTube safety limit for task_type: {task_type}")
            if not self._check_daily_youtube_safety_limit():
                logger.warning(f"Worker {worker_id} blocked: Daily YouTube safety limit (375) exceeded")
                return False
            
        # If task type specified and we have per-type config, use that
        if task_type and task_type in state.behavior_config_by_type:
            task_count = state.daily_task_counts_by_type.get(task_type, 0)
            max_downloads = state.behavior_config_by_type[task_type].get('daily_limits', {}).get('max_downloads', 35)
            has_capacity = task_count < max_downloads
            logger.debug(f"Worker {worker_id} daily capacity check for {task_type}: Count={task_count}, Max={max_downloads}, HasCapacity={has_capacity}")
        else:
            # For mixed workers without specific task type, check if ANY task type has capacity
            # rather than using overall count which combines all task types
            if state.behavior_config_by_type:
                # Check each configured task type
                has_any_capacity = False
                for task_t, config in state.behavior_config_by_type.items():
                    task_count = state.daily_task_counts_by_type.get(task_t, 0)
                    max_downloads = config.get('daily_limits', {}).get('max_downloads', 35)
                    if task_count < max_downloads:
                        has_any_capacity = True
                        logger.debug(f"Worker {worker_id} has capacity for {task_t}: Count={task_count}, Max={max_downloads}")
                    else:
                        logger.debug(f"Worker {worker_id} no capacity for {task_t}: Count={task_count}, Max={max_downloads}")
                has_capacity = has_any_capacity
                logger.debug(f"Worker {worker_id} daily capacity check (any type): HasCapacity={has_capacity}")
            else:
                # Fall back to overall limits for workers without per-type config
                has_capacity = state.daily_task_count < state.daily_limits['max_downloads']
                logger.debug(f"Worker {worker_id} daily capacity check (overall): Count={state.daily_task_count}, Max={state.daily_limits['max_downloads']}, HasCapacity={has_capacity}")
        
        # Additional safety check: if no task_type was specified but this might be for YouTube downloads,
        # check the global YouTube safety limit as a fallback
        if has_capacity and task_type is None:
            if not self._check_daily_youtube_safety_limit():
                logger.warning(f"Worker {worker_id} blocked: Daily YouTube safety limit (375) exceeded (fallback check)")
                has_capacity = False
                
        return has_capacity
        
    def _should_take_break(self, worker_id: str, task_type: Optional[str] = None, current_time_override: Optional[datetime] = None) -> bool:
        """Check if worker should take a break based on session duration"""
        state = self.worker_states.get(worker_id)
        if not state:
            logger.debug(f"Worker {worker_id} cannot check for break: No state found.")
            return False
            
        # Use per-type session tracking if task_type is provided and configured
        if task_type and task_type in state.behavior_config_by_type:
            session_start = state.current_session_start_by_type.get(task_type)
            if not session_start:
                logger.debug(f"Worker {worker_id} cannot check for break for {task_type}: No session start time.")
                return False
                
            target_duration = state.current_session_target_duration_by_type.get(task_type)
            if target_duration is None:
                logger.warning(f"Worker {worker_id} has no target duration set for {task_type}. Using default.")
                config = state.behavior_config_by_type[task_type]
                session_mgmt = config.get('session_management', state.session_management)
                target_duration = session_mgmt.get('session_duration_max', 7200)
        else:
            # Use overall session tracking
            if not state.current_session_start:
                logger.debug(f"Worker {worker_id} cannot check for break: No session start time.")
                return False
                
            session_start = state.current_session_start
            target_duration = state.current_session_target_duration
            if target_duration is None:
                logger.warning(f"Worker {worker_id} has no current_session_target_duration set. Using default max from config.")
                target_duration = state.session_management.get('session_duration_max', 7200)
        
        now = current_time_override or datetime.now(timezone.utc)
        elapsed_session_seconds = (now - session_start).total_seconds()
        should_break = elapsed_session_seconds > target_duration
        
        logger.debug(f"Worker {worker_id} Break Check" + 
                    (f" for {task_type}" if task_type else "") + 
                    f": Elapsed={elapsed_session_seconds:.1f}s, TargetDuration={target_duration:.1f}s, ShouldBreak={should_break}")
        return should_break
        
    def _is_in_break(self, worker_id: str, task_type: Optional[str] = None, current_time_override: Optional[datetime] = None) -> bool:
        """Check if worker is currently in a break period"""
        state = self.worker_states.get(worker_id)
        if not state:
            logger.debug(f"Worker {worker_id} is not in a break (no state found)")
            return False
            
        # Use per-type break tracking if task_type is provided and configured
        if task_type and task_type in state.behavior_config_by_type:
            break_end_time = state.break_end_time_by_type.get(task_type)
            if not break_end_time:
                logger.debug(f"Worker {worker_id} is not in a break for {task_type} (no break_end_time set)")
                return False
        else:
            # Use overall break tracking
            break_end_time = state.break_end_time
            if not break_end_time:
                logger.debug(f"Worker {worker_id} is not in a break (no break_end_time set)")
                return False
            
        now = current_time_override or datetime.now(timezone.utc)
        is_break = now < break_end_time
        
        if is_break:
            remaining = (break_end_time - now).total_seconds()
            logger.debug(f"Worker {worker_id} is in break period" + 
                       (f" for {task_type}" if task_type else "") + 
                       f": {remaining:.1f}s remaining")
        else:
            # Break has ended, clear the break time and start a new session
            logger.debug(f"Worker {worker_id} break period finished" + 
                       (f" for {task_type}" if task_type else "") + 
                       ". Starting new session")
            
            # Get appropriate session management config
            if task_type and task_type in state.behavior_config_by_type:
                config = state.behavior_config_by_type[task_type]
                session_mgmt = config.get('session_management', state.session_management)
            else:
                session_mgmt = state.session_management
                
            # Set target duration for the new session
            try:
                s_min = session_mgmt.get('session_duration_min', 3600)
                s_max = session_mgmt.get('session_duration_max', 7200)
                new_duration = random.uniform(s_min, s_max)
                
                if task_type and task_type in state.behavior_config_by_type:
                    state.current_session_target_duration_by_type[task_type] = new_duration
                else:
                    state.current_session_target_duration = new_duration
                    
                logger.debug(f"New session target duration for {worker_id}" + 
                           (f" ({task_type})" if task_type else "") + 
                           f": {new_duration/60:.1f} minutes")
            except Exception as e:
                logger.error(f"Error setting new session target duration for {worker_id}: {e}. Using default 3600s.")
                if task_type and task_type in state.behavior_config_by_type:
                    state.current_session_target_duration_by_type[task_type] = 3600
                else:
                    state.current_session_target_duration = 3600
            
        return is_break
        
    def _start_break(self, worker_id: str, duration: float, task_type: Optional[str] = None, current_time_override: Optional[datetime] = None):
        """Start a break period for a worker with a SPECIFIED duration."""
        state = self.worker_states.get(worker_id)
        if not state:
            return
            
        now = current_time_override or datetime.now(timezone.utc)
        break_end = now + timedelta(seconds=duration)
        
        # Set break end time for specific task type or overall
        if task_type and task_type in state.behavior_config_by_type:
            state.break_end_time_by_type[task_type] = break_end
            state.current_session_start_by_type[task_type] = None  # Clear session start during break
            logger.debug(f"Started break for worker {worker_id} ({task_type}) (Duration: {duration:.1f}s) until {break_end}")
        else:
            state.break_end_time = break_end
            state.current_session_start = None  # Clear session start during break
            logger.debug(f"Started break for worker {worker_id} (Duration: {duration:.1f}s) until {break_end}")
        
    def _end_break(self, worker_id: str):
        """End a break period for a worker"""
        state = self.worker_states.get(worker_id)
        if not state:
            return
            
        state.break_end_time = None
        state.current_session_start = datetime.now(timezone.utc)
        logger.debug(f"Ended break for worker {worker_id}, starting new session")
        
    async def _simulate_watch_time(self, worker_id: str, duration: float, task_type: Optional[str] = None):
        """Simulate human-like watch time"""
        state = self.worker_states.get(worker_id)
        if not state:
            return
            
        # Get appropriate watch_time config
        if task_type and task_type in state.behavior_config_by_type:
            config = state.behavior_config_by_type[task_type]
            watch_time = config.get('watch_time', state.watch_time)
        else:
            watch_time = state.watch_time
            
        # Add random variation to duration
        variation = random.uniform(0.8, 1.2)
        wait_time = duration * variation
        
        # Ensure wait time is within configured limits
        wait_time = max(watch_time['min_duration'], 
                       min(wait_time, watch_time['max_duration']))
        
        logger.debug(f"Worker {worker_id} simulating watch time" + 
                   (f" for {task_type}" if task_type else "") + 
                   f": {wait_time:.1f}s (original: {duration:.1f}s)")
        
        # Split wait into smaller chunks with random pauses
        chunk_size = min(watch_time['chunk_size'], wait_time)
        remaining = wait_time
        
        while remaining > 0:
            current_chunk = min(chunk_size, remaining)
            await asyncio.sleep(current_chunk)
            
            # Random pause between chunks
            if remaining > current_chunk:
                pause = random.uniform(watch_time['min_pause'], watch_time['max_pause'])
                logger.debug(f"Worker {worker_id} pausing for {pause:.1f}s between chunks")
                await asyncio.sleep(pause)
                
            remaining -= current_chunk
            
    async def _flush_dns(self):
        """Flush DNS cache based on the operating system."""
        # REMOVED: Conditional sys import check
        # if 'sys' not in sys.modules:
        #    import sys 
        #    logger.warning("Imported sys module inside _flush_dns - ideally should be at top level.")
            
        current_os = os.name
        current_platform = sys.platform # sys is now globally available
        logger.debug(f"Attempting DNS flush. Detected OS: {current_os}, Platform: {current_platform}")
        
        try:
            if current_os == 'nt':  # Windows
                logger.debug("Running ipconfig /flushdns for Windows DNS flush")
                subprocess.run(['ipconfig', '/flushdns'], check=True, capture_output=True)
                logger.debug("Successfully flushed DNS cache on Windows")
            elif current_platform == 'darwin': # macOS
                logger.debug("Running dscacheutil -flushcache for macOS DNS flush")
                subprocess.run(['dscacheutil', '-flushcache'], check=True, capture_output=True)
                # REMOVED: sudo killall -HUP mDNSResponder to avoid password
                logger.debug("Successfully flushed DNS cache on macOS using dscacheutil")
            elif current_platform.startswith('linux'): # Linux (more specific than just 'else')
                logger.debug("Running sudo systemd-resolve --flush-caches for Linux DNS flush")
                # It's common for this to require sudo on Linux
                subprocess.run(['sudo', 'systemd-resolve', '--flush-caches'], check=True, capture_output=True)
                logger.debug("Successfully flushed DNS cache on Linux")
            else:
                logger.warning(f"DNS flush not implemented for OS: {current_os} Platform: {current_platform}. Skipping DNS flush.")
                
        except FileNotFoundError as e:
            logger.warning(f"DNS flush command not found: {e}. Skipping DNS flush.")
        except subprocess.CalledProcessError as e:
            # Log stdout/stderr for debugging if the command fails
            stderr = e.stderr.decode() if e.stderr else "(no stderr)"
            logger.warning(f"Failed to flush DNS cache. Command '{' '.join(e.cmd)}' returned {e.returncode}. Error: {stderr}")
        except Exception as e:
            # Catch any other potential exceptions
            logger.error(f"An unexpected error occurred during DNS flush: {str(e)}", exc_info=True)
            
    def _get_random_user_agent(self) -> str:
        """Get a random user agent"""
        agent = random.choice(self.user_agents)
        logger.debug(f"Selected user agent: {agent}")
        return agent
        
    def _should_rotate_session(self, worker_id: str) -> bool:
        """Check if session should be rotated"""
        state = self.worker_states.get(worker_id)
        if not state or not state.current_session_start:
            return True
            
        session_duration = datetime.now(timezone.utc) - state.current_session_start
        should_rotate = session_duration.total_seconds() > state.session_management['session_duration_max']
        
        if should_rotate:
            logger.debug(f"Worker {worker_id} should rotate session: duration {session_duration.total_seconds():.1f}s > {state.session_management['session_duration_max']}s")
        else:
            logger.debug(f"Worker {worker_id} session duration: {session_duration.total_seconds():.1f}s")
            
        return should_rotate
        
    def _should_flush_dns(self, worker_id: str) -> bool:
        """Check if DNS should be flushed"""
        state = self.worker_states.get(worker_id)
        if not state or not state.last_dns_flush:
            return True
            
        time_since_flush = datetime.now(timezone.utc) - state.last_dns_flush
        should_flush = time_since_flush.total_seconds() > state.network_settings['dns_flush_interval']
        
        if should_flush:
            logger.debug(f"Worker {worker_id} should flush DNS: {time_since_flush.total_seconds():.1f}s since last flush")
        else:
            logger.debug(f"Worker {worker_id} DNS flush check: {time_since_flush.total_seconds():.1f}s since last flush")
            
        return should_flush
        
    def _should_change_user_agent(self, worker_id: str) -> bool:
        """Check if user agent should be changed"""
        state = self.worker_states.get(worker_id)
        if not state or not state.last_user_agent_change:
            return True
            
        time_since_change = datetime.now(timezone.utc) - state.last_user_agent_change
        should_change = time_since_change.total_seconds() > state.network_settings['user_agent_change_interval']
        
        if should_change:
            logger.debug(f"Worker {worker_id} should change user agent: {time_since_change.total_seconds():.1f}s since last change")
        else:
            logger.debug(f"Worker {worker_id} user agent check: {time_since_change.total_seconds():.1f}s since last change")
            
        return should_change
        
    def _calculate_operating_hours_wait_time(self, worker_id: str, current_time_override: Optional[datetime] = None, force_next_day: bool = False) -> float:
        """Calculate how long to wait until operating hours start. Returns 0 if within hours and not forcing next day."""
        state = self.worker_states.get(worker_id)
        if not state:
            return 0.0
            
        now = current_time_override or datetime.now(timezone.utc)
        local_now = self._convert_to_local_time(now)
            
        if not force_next_day and self._is_within_operating_hours(worker_id, current_time_override=now):
            return 0.0 # Already within hours and not forcing next day
            
        # Calculate wait time until next start
        start_h = state.operating_hours['start_hour_min']
        start_m = state.operating_hours['start_minute_min']
        
        # Calculate the next start time in local time
        next_start_today = local_now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        
        if not force_next_day and local_now < next_start_today:
            # If current time is before today's start time and not forcing next day, wait until today's start
            wait_seconds = (next_start_today - local_now).total_seconds()
            logger.debug(f"Worker {worker_id} OperatingHoursWait: Waiting {wait_seconds:.1f}s until today's start ({start_h:02}:{start_m:02}) local time")
            return wait_seconds
        else:
            # If current time is past today's start time OR we're forcing next day, wait until tomorrow's start
            next_start_tomorrow = next_start_today + timedelta(days=1)
            wait_seconds = (next_start_tomorrow - local_now).total_seconds()
            logger.debug(f"Worker {worker_id} OperatingHoursWait: Waiting {wait_seconds:.1f}s until tomorrow's start ({start_h:02}:{start_m:02}) local time")
            return wait_seconds

    def calculate_next_task_wait_time(self, worker_id: str, task_type: Optional[str] = None, current_time_override: Optional[datetime] = None) -> Tuple[float, str]:
        """
        Calculate how long to wait before assigning the next task.
        If a break should start, determine its duration and start it.
        Returns a tuple: (wait_time_seconds, reason_string)
        
        Args:
            worker_id: The worker ID
            task_type: Optional task type (e.g., 'download_youtube', 'download_rumble')
            current_time_override: Optional datetime for testing
        """
        now = current_time_override or datetime.now(timezone.utc)
        
        logger.debug(f"calculate_next_task_wait_time called for worker {worker_id} with task_type: {task_type}")
        
        # Reset daily counts if a new day has started (pass current time for consistency)
        self.reset_daily_counts(current_time_override=now)
        
        state = self.worker_states.get(worker_id)
        if not state:
            logger.warning(f"Cannot calculate wait time for unknown worker {worker_id}")
            return (0.0, "unknown_worker")
        
        # Removed verbose debug log

        # 1. Check if currently in a break
        if self._is_in_break(worker_id, task_type, current_time_override=now):
            # Get the appropriate break end time
            if task_type and task_type in state.behavior_config_by_type:
                break_end = state.break_end_time_by_type.get(task_type)
            else:
                break_end = state.break_end_time
                
            if break_end:
                wait_time = (break_end - now).total_seconds()
                reason = "in_break"
                logger.debug(f"WaitReason: {reason}" + (f" for {task_type}" if task_type else "") + f". Waiting {wait_time:.1f}s")
                return (max(0.0, wait_time), reason)
            
        # 2. Check Operating Hours
        op_hours_wait = self._calculate_operating_hours_wait_time(worker_id, current_time_override=now)
        if op_hours_wait > 0:
            reason = "outside_operating_hours"
            logger.debug(f"WaitReason: {reason}. Waiting {op_hours_wait:.1f}s")
            return (op_hours_wait, reason)
            
        # 3. Check Daily Capacity
        if not self._has_daily_capacity(worker_id, task_type):
            # If max capacity reached, wait until the *next day's* operating window starts
            wait_time = self._calculate_operating_hours_wait_time(worker_id, current_time_override=now, force_next_day=True)
            reason = "daily_capacity_reached"
            if task_type and task_type in state.behavior_config_by_type:
                task_count = state.daily_task_counts_by_type.get(task_type, 0)
                max_downloads = state.behavior_config_by_type[task_type].get('daily_limits', {}).get('max_downloads', 35)
                logger.debug(f"WaitReason: {reason} for {task_type} ({task_count}/{max_downloads}). Waiting {wait_time:.1f}s until next day's operating window.")
            else:
                # Show per-type counts if available, otherwise overall count
                if state.behavior_config_by_type:
                    type_counts = []
                    for task_t, config in state.behavior_config_by_type.items():
                        task_count = state.daily_task_counts_by_type.get(task_t, 0)
                        max_downloads = config.get('daily_limits', {}).get('max_downloads', 35)
                        type_counts.append(f"{task_t}={task_count}/{max_downloads}")
                    counts_str = ", ".join(type_counts)
                    logger.debug(f"WaitReason: {reason} ({counts_str}). Waiting {wait_time:.1f}s until next day's operating window.")
                else:
                    logger.debug(f"WaitReason: {reason} ({state.daily_task_count}/{state.daily_limits['max_downloads']}). Waiting {wait_time:.1f}s until next day's operating window.")
            return (wait_time, reason)
            
        # 4. Start session if it's the first task (no session running)
        # Check per-type session if task_type is provided
        if task_type and task_type in state.behavior_config_by_type:
            if task_type not in state.current_session_start_by_type or state.current_session_start_by_type[task_type] is None:
                logger.debug(f"Starting new session for worker {worker_id} ({task_type}) - first task of period")
                state.current_session_start_by_type[task_type] = now
                # Set target duration for the new session
                try:
                    config = state.behavior_config_by_type[task_type]
                    session_mgmt = config.get('session_management', state.session_management)
                    s_min = session_mgmt.get('session_duration_min', 3600)
                    s_max = session_mgmt.get('session_duration_max', 7200)
                    state.current_session_target_duration_by_type[task_type] = random.uniform(s_min, s_max)
                    logger.debug(f"New session target duration for {worker_id} ({task_type}): {state.current_session_target_duration_by_type[task_type]/60:.1f} minutes")
                except Exception as e:
                    logger.error(f"Error setting session target duration for {worker_id} ({task_type}): {e}. Using default 3600s.")
                    state.current_session_target_duration_by_type[task_type] = 3600
        else:
            # Use overall session tracking
            if state.current_session_start is None:
                logger.debug(f"Starting new session for worker {worker_id} - first task of period")
                state.current_session_start = now
                # Set target duration for the new session
                try:
                    s_min = state.session_management.get('session_duration_min', 3600)
                    s_max = state.session_management.get('session_duration_max', 7200)
                    state.current_session_target_duration = random.uniform(s_min, s_max)
                    logger.debug(f"New session target duration for {worker_id}: {state.current_session_target_duration/60:.1f} minutes")
                except Exception as e:
                    logger.error(f"Error setting session target duration for {worker_id}: {e}. Using default 3600s.")
                    state.current_session_target_duration = 3600
        
        # 5. Check if a break *should* start now based on session duration
        # Skip break logic if it's the first task of the day OR if we're in early operating hours
        if state.daily_task_count == 0:
            logger.debug(f"Skipping break check for worker {worker_id} - first task of day")
        elif self._is_early_in_operating_hours(worker_id, current_time_override=now):
            logger.debug(f"Skipping break check for worker {worker_id} - early in operating hours")
        else:
            should_break_flag = self._should_take_break(worker_id, task_type, current_time_override=now)
            if should_break_flag:
                # Get appropriate break duration config
                if task_type and task_type in state.behavior_config_by_type:
                    config = state.behavior_config_by_type[task_type]
                    session_mgmt = config.get('session_management', state.session_management)
                else:
                    session_mgmt = state.session_management
                    
                # Take a regular break - always use the standard break duration
                min_break = session_mgmt.get('break_duration_min', 3600) # Default normal min
                max_break = session_mgmt.get('break_duration_max', 7200) # Default normal max
                
                # Ensure min <= max
                if min_break > max_break:
                    logger.warning(f"Worker {worker_id} break duration min ({min_break}) > max ({max_break}). Swapping.")
                    min_break, max_break = max_break, min_break
                    
                current_break_duration = random.uniform(min_break, max_break)
                
                # --- STATE UPDATE: Start the break with calculated duration --- 
                self._start_break(worker_id, current_break_duration, task_type, current_time_override=now)
                # --- End STATE UPDATE --- 
                reason = "session_break_starting"
                logger.debug(f"WaitReason: {reason}" + (f" for {task_type}" if task_type else "") + f". Starting break. Waiting {current_break_duration:.1f}s")
                return (current_break_duration, reason) # Wait for the full duration of the break we just started

        # 6. Check Network related waits (DNS flush, User Agent change) - currently informational, not causing waits
        # Note: These checks use datetime.now() internally as they relate to real-world time intervals, not simulated task progression
        if self._should_flush_dns(worker_id):
            logger.debug(f"Worker {worker_id} due for DNS flush (informational)..")
        if self._should_change_user_agent(worker_id):
            logger.debug(f"Worker {worker_id} due for User Agent change (informational)..")
            
        # If none of the above conditions triggered a wait, return 0
        reason = "ready"
        logger.debug(f"No wait conditions met for worker {worker_id}. Wait time = 0.0s. Reason: {reason}")
        return (0.0, reason)

    def task_assignment_decorator(self, func: Callable) -> Callable:
        """Decorator for task assignment - NOW MINIMAL, only logging boundaries."""
        @wraps(func)
        async def wrapper(worker_id: str, *args, **kwargs):
            # Removed verbose debug log
            # Wait time logic is now handled EXTERNALLY in handle_task_completion
            result = await func(worker_id, *args, **kwargs)
            # Task count update is now handled EXTERNALLY in handle_task_completion
            # Removed verbose debug log
            return result
            
        return wrapper
        
    def task_processing_decorator(self, func: Callable) -> Callable:
        """Decorator for task processing - Handles User Agent and potentially DNS flush."""
        @wraps(func)
        async def wrapper(task: Dict, *args, **kwargs):
            worker_id = task.get('worker_id')
            task_type = task.get('task_type') # Get task type

            if not worker_id or worker_id not in self.worker_states:
                 logger.warning(f"Task processing decorator called with invalid/unknown worker_id: {worker_id}")
                 return await func(task, *args, **kwargs)

            state = self.worker_states[worker_id]
            now = datetime.now(timezone.utc)
            # Removed verbose debug log

            # --- BEGIN CONDITIONAL EXECUTION ---
            if task_type in ['download_youtube', 'download_rumble']:
                logger.debug(f"Applying download-specific behavior for worker {worker_id} (task type: {task_type})")
                # Check DNS flush
                if self._should_flush_dns(worker_id):
                    logger.debug(f"Flushing DNS for worker {worker_id}")
                    await self._flush_dns() # Perform flush before task processing potentially uses network
                    state.last_dns_flush = now
                
                # Check user agent change
                if self._should_change_user_agent(worker_id):
                    logger.debug(f"Changing user agent for worker {worker_id}")
                    state.current_user_agent = self._get_random_user_agent()
                    state.last_user_agent_change = now
                elif not state.current_user_agent:
                    # Ensure user agent is set if it hasn't been changed yet
                    state.current_user_agent = self._get_random_user_agent()
                    state.last_user_agent_change = now # Record the time it was first set
                    logger.debug(f"Assigned initial user agent for worker {worker_id}")

                # Get current user agent and add to task headers
                if state.current_user_agent:
                    if 'headers' not in task:
                        task['headers'] = {}
                    task['headers']['User-Agent'] = state.current_user_agent
                    logger.debug(f"Using User-Agent for task: {state.current_user_agent}")
                else:
                    logger.warning(f"No user agent available for worker {worker_id} during task processing")
            else:
                 logger.debug(f"Skipping download-specific behavior for task type: {task_type}")
            # --- END CONDITIONAL EXECUTION ---

            # Execute original function (the actual network call like yt-dlp)
            result = await func(task, *args, **kwargs)
            
            # Simulate watch time (remains here as it's part of processing)
            if 'duration' in task:
                # Ensure duration is a float or int
                try:
                    duration_val = float(task['duration'])
                    logger.debug(f"Simulating watch time for task (Duration: {duration_val:.1f}s)")
                    await self._simulate_watch_time(worker_id, duration_val, task_type)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid duration value found in task for watch time simulation: {task['duration']}")
            
            # Removed verbose debug log
            return result
            
        return wrapper
        
    def calculate_post_task_delay(self, worker_id: str, task_type: Optional[str] = None, current_time_override: Optional[datetime] = None) -> float:
        """Calculates a random delay duration based on dynamically calculated watch_time config."""
        state = self.worker_states.get(worker_id)
        if not state:
            logger.warning(f"Cannot calculate post-task delay for unknown worker {worker_id}")
            return 0.0

        # Get appropriate watch_time config based on task type
        if task_type and task_type in state.behavior_config_by_type:
            # Use per-type config with dynamically calculated values
            config = state.behavior_config_by_type[task_type]
            watch_time = config.get('watch_time', state.watch_time)
        else:
            # Fall back to overall watch_time
            watch_time = state.watch_time

        # Calculate a random duration based on min/max watch time config
        min_dur = watch_time['min_duration']
        max_dur = watch_time['max_duration']
        base_wait_time = random.uniform(min_dur, max_dur)

        # Add variation factor
        variation = watch_time['variation_factor']
        delay_time = base_wait_time * random.uniform(1 - variation, 1 + variation)
        
        # Ensure delay time is within absolute min/max bounds and at least 20 seconds
        delay_time = max(20.0, min(delay_time, max_dur))
        
        return delay_time

    def reset_daily_counts(self, current_time_override: Optional[datetime] = None):
        """Reset daily task counts and session state for all workers when a new day starts"""
        now = current_time_override or datetime.now(timezone.utc)
        logger.debug(f"--- Attempting Daily Count Reset at {now} ---")
        for worker_id, state in self.worker_states.items():
            if state.last_task_time:
                 # Check if the last task was completed on a previous day (UTC)
                 if state.last_task_time.date() < now.date():
                     logger.debug(f"Resetting daily state for worker {worker_id}. Old count: {state.daily_task_count}")
                     state.daily_task_count = 0
                     
                     # Reset per-type counts
                     if state.daily_task_counts_by_type:
                         for task_type, count in state.daily_task_counts_by_type.items():
                             if count > 0:
                                 logger.debug(f"  Resetting {task_type} count from {count} to 0")
                         state.daily_task_counts_by_type.clear()
                     
                     # Reset session state for new day
                     state.current_session_start = None
                     state.break_end_time = None
                     state.current_session_target_duration = None
                     
                     # Reset per-type session states
                     state.current_session_start_by_type.clear()
                     state.break_end_time_by_type.clear()
                     state.current_session_target_duration_by_type.clear()
                     
                     logger.debug(f"Session state reset for worker {worker_id} - new day")
                 else:
                     logger.debug(f"No daily count reset needed for {worker_id} (last task today)")
            else:
                # If no tasks have been run, ensure count is 0 and session state is clean
                if state.daily_task_count != 0:
                     logger.debug(f"Resetting daily task count for worker {worker_id} (no previous tasks). Old count: {state.daily_task_count}")
                     state.daily_task_count = 0
                else:
                     logger.debug(f"No daily count reset needed for {worker_id} (no tasks run yet)")
                     
                # Clear per-type counts
                if state.daily_task_counts_by_type:
                    state.daily_task_counts_by_type.clear()
                    
                # Ensure session state is clean for workers with no previous tasks
                if state.current_session_start or state.break_end_time:
                    state.current_session_start = None
                    state.break_end_time = None
                    state.current_session_target_duration = None
                    logger.info(f"Cleaned session state for worker {worker_id} (no previous tasks)")
                    
                # Clear per-type session states
                state.current_session_start_by_type.clear()
                state.break_end_time_by_type.clear()
                state.current_session_target_duration_by_type.clear()
                
    def get_worker_stats(self, worker_id: str) -> Dict:
        """Get behavior statistics for a worker"""
        state = self.worker_states.get(worker_id)
        if not state:
            logger.warning(f"No stats found for worker {worker_id}")
            return {}
            
        stats = {
            'daily_task_count': state.daily_task_count,
            'daily_limits': state.daily_limits,
            'last_task_time': state.last_task_time.isoformat() if state.last_task_time else None,
            'current_session_start': state.current_session_start.isoformat() if state.current_session_start else None,
            'break_end_time': state.break_end_time.isoformat() if state.break_end_time else None,
            'operating_hours': state.operating_hours,
            'session_management': state.session_management,
            'watch_time': state.watch_time,
            'current_user_agent': state.current_user_agent,
            'network_settings': state.network_settings
        }
        logger.debug(f"Retrieved stats for worker {worker_id}: {stats}")
        return stats 

    def _should_change_cookie_profile(self, worker_id: str) -> bool:
        """Check if worker should change cookie profile"""
        state = self.worker_states.get(worker_id)
        if not state or not state.last_cookie_change:
            return True
            
        # Change cookies after a certain number of tasks or time
        time_since_change = datetime.now(timezone.utc) - state.last_cookie_change
        should_change = (
            time_since_change.total_seconds() > 3600 or  # Change every hour
            state.daily_task_count % 10 == 0  # Or every 10 tasks
        )
        
        if should_change:
            logger.debug(f"Worker {worker_id} should change cookie profile: {time_since_change.total_seconds():.1f}s since last change")
        
        return should_change
        
    def get_cookie_profile(self, worker_id: str) -> Optional[str]:
        """Get the cookie profile to use for the next task"""
        state = self.worker_states.get(worker_id)
        if not state:
            logger.warning(f"No state found for worker {worker_id}")
            return None
            
        if not self.cookie_profiles:
            logger.warning("No cookie profiles available")
            return None
            
        # Check if we should change profiles
        if self._should_change_cookie_profile(worker_id):
            # Choose a random profile, but not the same as current
            available_profiles = [p for p in self.cookie_profiles if p != state.current_cookie_profile]
            if not available_profiles:
                available_profiles = self.cookie_profiles
                
            state.current_cookie_profile = random.choice(available_profiles)
            state.last_cookie_change = datetime.now(timezone.utc)
            logger.debug(f"Changed cookie profile for worker {worker_id} to {state.current_cookie_profile}")
            
        return state.current_cookie_profile 

    def handle_task_completion(self, worker_id: str, task_type: Optional[str] = None, current_time_override: Optional[datetime] = None):
        """Updates worker state after a task is successfully completed."""
        state = self.worker_states.get(worker_id)
        if not state:
            logger.warning(f"Cannot handle task completion for unknown worker {worker_id}")
            return

        now = current_time_override or datetime.now(timezone.utc)
        
        # Check if we need to reset daily counts BEFORE updating task completion
        # This handles the case where a task completes on a new day
        if state.last_task_time and state.last_task_time.date() < now.date():
            logger.info(f"Daily boundary crossed for worker {worker_id}. Resetting counts before task completion.")
            self.reset_daily_counts(current_time_override=now)
        
        # Increment daily count (both overall and per-type)
        state.daily_task_count += 1
        
        # Increment per-type count if task_type provided
        if task_type:
            if task_type not in state.daily_task_counts_by_type:
                state.daily_task_counts_by_type[task_type] = 0
            state.daily_task_counts_by_type[task_type] += 1
            
            # Update per-type last task time
            state.last_task_time_by_type[task_type] = now
            
            logger.debug(f"Task type {task_type} count: {state.daily_task_counts_by_type[task_type]}")
        
        # Update last task time
        state.last_task_time = now
        
        # Reset consecutive failures if applicable (assuming success here)
        state.consecutive_blocked = 0 
        
        # If this is the first task of a session (session_start might be None if just after break)
        if task_type and task_type in state.behavior_config_by_type:
            if task_type not in state.current_session_start_by_type or state.current_session_start_by_type[task_type] is None:
                state.current_session_start_by_type[task_type] = now
                logger.debug(f"Worker {worker_id} started new session for {task_type} on task completion at {now}")
        else:
            if state.current_session_start is None:
                state.current_session_start = now
                logger.debug(f"Worker {worker_id} started new session on task completion at {now}")

        # Calculate wait time for next task
        wait_time = self.calculate_post_task_delay(worker_id, task_type, now)
        
        # Simplified logging showing count and wait
        if task_type:
            type_count = state.daily_task_counts_by_type.get(task_type, 0)
            logger.info(f"Task completed. Daily count: {type_count} ({task_type}). Wait time: {wait_time:.1f}s")
        else:
            logger.info(f"Task completed. Daily count: {state.daily_task_count}. Wait time: {wait_time:.1f}s") 

    def simulate_worker_behavior(self, worker_id: str, hours: int = 24) -> Dict:
        """Simulate worker behavior for specified number of hours"""
        state = self.worker_states.get(worker_id)
        if not state:
            logger.warning(f"No state found for worker {worker_id}")
            return {}
            
        # Initialize simulation state starting at midnight
        now = datetime.now(timezone.utc)
        simulation_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        simulation_end = simulation_start + timedelta(hours=hours)
        
        current_time = simulation_start
        events = []
        total_wait_time = 0
        total_break_time = 0
        session_count = 0
        current_session_start = None
        
        # Reset daily count for simulation
        state.daily_task_count = 0
        
        logger.info(f"Starting simulation for worker {worker_id}")
        logger.info(f"Daily limits: min={state.daily_limits['min_downloads']}, max={state.daily_limits['max_downloads']}, target={state.daily_limits['target_downloads']}")
        
        while current_time < simulation_end:
            # Check if we've hit daily limits
            if state.daily_task_count >= state.daily_limits['max_downloads']:
                logger.info(f"Worker {worker_id} reached daily limit of {state.daily_limits['max_downloads']} tasks")
                # Add remaining time as wait time
                remaining_time = (simulation_end - current_time).total_seconds()
                total_wait_time += remaining_time
                events.append({
                    'time': current_time,
                    'type': 'wait',
                    'duration': remaining_time,
                    'reason': 'daily_limit_reached'
                })
                break
                
            # Calculate wait time for next task
            wait_time, reason = self.calculate_next_task_wait_time(worker_id, current_time_override=current_time)
            
            if wait_time > 0:
                if reason == "in_break":
                    total_break_time += wait_time
                    events.append({
                        'time': current_time,
                        'type': 'break',
                        'duration': wait_time,
                        'reason': reason
                    })
                else:
                    total_wait_time += wait_time
                    events.append({
                        'time': current_time,
                        'type': 'wait',
                        'duration': wait_time,
                        'reason': reason
                    })
                current_time += timedelta(seconds=wait_time)
                continue
                
            # If we're starting a new session
            if not current_session_start:
                current_session_start = current_time
                session_count += 1
                events.append({
                    'time': current_time,
                    'type': 'session_start',
                    'session_number': session_count
                })
                
            # Simulate task execution
            events.append({
                'time': current_time,
                'type': 'task',
                'task_number': state.daily_task_count + 1
            })
            
            # Update state
            state.daily_task_count += 1
            state.last_task_time = current_time
            
            logger.debug(f"Task {state.daily_task_count} completed at {current_time}")
            
            # Calculate post-task delay
            delay = self.calculate_post_task_delay(worker_id, current_time_override=current_time)
            if delay > 0:
                total_wait_time += delay
                events.append({
                    'time': current_time,
                    'type': 'delay',
                    'duration': delay
                })
                current_time += timedelta(seconds=delay)
            else:
                current_time += timedelta(seconds=1)  # Minimal increment if no delay
                
        # Calculate statistics
        total_tasks = state.daily_task_count
        total_hours = (simulation_end - simulation_start).total_seconds() / 3600
        tasks_per_hour = total_tasks / total_hours if total_hours > 0 else 0
        
        logger.info(f"Simulation completed for worker {worker_id}")
        logger.info(f"Final task count: {total_tasks} (target was {state.daily_limits['target_downloads']})")
        
        return {
            'simulation_start': simulation_start,
            'simulation_end': simulation_end,
            'total_tasks': total_tasks,
            'tasks_per_hour': tasks_per_hour,
            'total_wait_time': total_wait_time / 3600,  # Convert to hours
            'total_break_time': total_break_time / 3600,  # Convert to hours
            'session_count': session_count,
            'events': events
        }
        
    def print_simulation_report(self, worker_id: str, simulation_results: Dict):
        """Print a detailed simulation report"""
        if not simulation_results:
            logger.warning(f"No simulation results for worker {worker_id}")
            return
            
        print("\n" + "="*50)
        print(f"Simulation Report for Worker {worker_id}")
        print("="*50)
        print(f"Simulation Period: {simulation_results['simulation_start'].strftime('%Y-%m-%d %H:%M:%S')} to {simulation_results['simulation_end'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tasks: {simulation_results['total_tasks']}")
        print(f"Average Tasks per Hour: {simulation_results['tasks_per_hour']:.1f}")
        print(f"Total Wait Time: {simulation_results['total_wait_time']:.1f} hours")
        print(f"Total Break Time: {simulation_results['total_break_time']:.1f} hours")
        print(f"Number of Sessions: {simulation_results['session_count']}")
        
        # Print session details
        print("\nSession Details:")
        current_session = None
        session_start = None
        session_tasks = 0
        
        for event in simulation_results['events']:
            if event['type'] == 'session_start':
                if current_session is not None:
                    duration = (event['time'] - session_start).total_seconds() / 60
                    print(f"  Session {current_session}: {session_tasks} tasks over {duration:.1f} minutes")
                current_session = event['session_number']
                session_start = event['time']
                session_tasks = 0
            elif event['type'] == 'task':
                session_tasks += 1
                
        # Print final session
        if current_session is not None:
            duration = (simulation_results['simulation_end'] - session_start).total_seconds() / 60
            print(f"  Session {current_session}: {session_tasks} tasks over {duration:.1f} minutes")
            
        print("="*50 + "\n") 

# Example usage (for testing)
if __name__ == '__main__':
    # Configure logging for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running human_behavior.py standalone for testing...")

    # Assume config.yaml is in ../../config relative to this file if run directly
    config_file = Path(__file__).parent.parent / 'config' / 'config.yaml'
    if not config_file.exists():
        logger.error(f"Config file not found at {config_file} for standalone test.")
        sys.exit(1)

    try:
        manager = HumanBehaviorManager(config_path=str(config_file))
        
        # Manually register workers based on example config structure
        worker_configs = manager.config.get('processing', {}).get('workers', {})
        if not worker_configs:
            logger.error("No worker configurations found in config file.")
            sys.exit(1)

        registered_workers = []
        for worker_id, config in worker_configs.items():
            if config.get('enabled'):
                logger.info(f"\n--- Registering worker: {worker_id} ---")
                manager.register_worker(worker_id, config) # Pass the whole config dict
                registered_workers.append(worker_id)
            else:
                logger.info(f"Skipping disabled worker: {worker_id}")

        if not registered_workers:
            logger.error("No enabled workers found to test.")
            sys.exit(1)

        # --- Simulation --- #
        # Pick the first registered worker for simulation
        test_worker_id = registered_workers[0]
        logger.info(f"\n--- Running Simulation for worker: {test_worker_id} ---")
        simulation_results = manager.simulate_worker_behavior(test_worker_id, hours=24)
        manager.print_simulation_report(test_worker_id, simulation_results)

        # --- Test Wait Time Calculation --- #
        logger.info(f"\n--- Testing Wait Time Calculation for worker: {test_worker_id} ---")
        # Reset state for wait time test
        state = manager.worker_states[test_worker_id]
        state.break_end_time = None
        state.current_session_start = datetime.now(timezone.utc)
        state.daily_task_count = 0
        
        wait_time, reason = manager.calculate_next_task_wait_time(test_worker_id)
        logger.info(f"Initial wait time for {test_worker_id}: {wait_time:.1f}s, Reason: {reason}")
        
        # Simulate reaching session end
        logger.info(f"Simulating session end for {test_worker_id}...")
        session_duration = state.session_management.get('session_duration_max', 3600)
        fake_past_time = state.current_session_start - timedelta(seconds=session_duration + 10)
        state.current_session_start = fake_past_time # Set session start far in the past
        wait_time, reason = manager.calculate_next_task_wait_time(test_worker_id)
        logger.info(f"Wait time after session end simulation for {test_worker_id}: {wait_time:.1f}s, Reason: {reason}")
        
        # Simulate being in break
        if state.break_end_time: # Check if break was started by previous call
            logger.info(f"Simulating being in break for {test_worker_id}...")
            fake_now_in_break = state.break_end_time - timedelta(seconds=10)
            wait_time, reason = manager.calculate_next_task_wait_time(test_worker_id, current_time_override=fake_now_in_break)
            logger.info(f"Wait time during break simulation for {test_worker_id}: {wait_time:.1f}s, Reason: {reason}")
        else:
            logger.warning("Could not simulate being in break, break_end_time not set.")

        logger.info("\nStandalone test finished.")

    except Exception as e:
        logger.error(f"An error occurred during standalone test: {e}", exc_info=True) 