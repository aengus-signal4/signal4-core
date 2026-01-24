"""Tab rendering functions for the system monitoring dashboard."""

from .quick_status import render_quick_status_tab
from .services import render_services_tab
from .tasks import render_tasks_tab
from .projects import render_projects_tab
from .logs import render_logs_tab

__all__ = [
    'render_quick_status_tab',
    'render_services_tab',
    'render_tasks_tab',
    'render_projects_tab',
    'render_logs_tab',
]
