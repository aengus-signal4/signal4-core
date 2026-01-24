"""
API utility functions for the system monitoring dashboard.
"""

import requests
import streamlit as st

from ..config import ORCHESTRATOR_API_URL
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('system_monitoring')


def fetch_api(endpoint: str, base_url: str = None, timeout: int = 15) -> dict:
    """Fetch data from API with error handling"""
    url = base_url or ORCHESTRATOR_API_URL
    try:
        response = requests.get(f"{url}{endpoint}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Connection refused - Service not running"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": str(e)}


def post_api(endpoint: str, data: dict = None, base_url: str = None) -> dict:
    """POST to API with error handling"""
    url = base_url or ORCHESTRATOR_API_URL
    try:
        response = requests.post(f"{url}{endpoint}", json=data or {}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Connection refused - Service not running"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=30)
def check_service_health(url: str, endpoint: str = "/health") -> dict:
    """Check if a service is healthy"""
    try:
        response = requests.get(f"{url}{endpoint}", timeout=3)
        if response.status_code == 200:
            return {"status": "running", "response": response.json() if response.text else {}}
        else:
            return {"status": "unhealthy", "code": response.status_code}
    except requests.exceptions.ConnectionError:
        return {"status": "stopped", "error": "Connection refused"}
    except requests.exceptions.Timeout:
        return {"status": "timeout", "error": "Request timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


