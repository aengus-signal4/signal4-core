import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set, Optional, Dict

class AuditTracker:
    """Tracks which content has been audited to avoid redundant checks."""
    
    def __init__(self, log_dir: Path, max_age_days: int = 7):
        """
        Initialize the audit tracker.
        
        Args:
            log_dir: Directory where audit history should be stored
            max_age_days: Maximum age of audit records to consider valid
        """
        self.log_dir = log_dir
        self.max_age_days = max_age_days
        self.history_file = log_dir / "audit_history.json"
        self.logger = logging.getLogger("audit_tracker")
        self._audit_history: Dict[str, str] = {}
        self._load_history()
    
    def _load_history(self) -> None:
        """Load audit history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    self._audit_history = json.load(f)
                self._clean_old_entries()
        except Exception as e:
            self.logger.error(f"Error loading audit history: {e}")
            self._audit_history = {}
    
    def _save_history(self) -> None:
        """Save audit history to file."""
        try:
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.history_file, 'w') as f:
                json.dump(self._audit_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving audit history: {e}")
    
    def _clean_old_entries(self) -> None:
        """Remove entries older than max_age_days."""
        cutoff_date = (datetime.now() - timedelta(days=self.max_age_days)).isoformat()
        self._audit_history = {
            content_id: audit_date 
            for content_id, audit_date in self._audit_history.items()
            if audit_date >= cutoff_date
        }
    
    def was_recently_audited(self, content_id: str) -> bool:
        """
        Check if content was audited within the max age window.
        
        Args:
            content_id: The content ID to check
            
        Returns:
            bool: True if content was recently audited, False otherwise
        """
        if content_id not in self._audit_history:
            return False
            
        audit_date = datetime.fromisoformat(self._audit_history[content_id])
        age = datetime.now() - audit_date
        return age.days <= self.max_age_days
    
    def mark_audited(self, content_id: str) -> None:
        """
        Mark content as having been audited.
        
        Args:
            content_id: The content ID that was audited
        """
        self._audit_history[content_id] = datetime.now().isoformat()
        self._save_history()
    
    def get_audit_date(self, content_id: str) -> Optional[datetime]:
        """
        Get the date when content was last audited.
        
        Args:
            content_id: The content ID to check
            
        Returns:
            Optional[datetime]: The audit date if found, None otherwise
        """
        if content_id in self._audit_history:
            return datetime.fromisoformat(self._audit_history[content_id])
        return None 