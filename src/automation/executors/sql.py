"""
SQL Executor - Executes PostgreSQL functions

SQL function definitions are documented in:
    src/database/sql/cache_refresh_functions.sql

Currently used functions:
    - refresh_embedding_cache_7d(): Hourly cache refresh (7-day window)
    - refresh_embedding_cache_30d(): 4x daily cache refresh (30-day window)
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import text

from .base import BaseExecutor, ExecutionResult

logger = logging.getLogger(__name__)

# Thread pool for running synchronous database operations
_executor = ThreadPoolExecutor(max_workers=2)


class SQLExecutor(BaseExecutor):
    """
    Executes PostgreSQL functions.

    Config options:
    - function: Name of the PostgreSQL function to call
    - params: Optional dictionary of parameters to pass
    """

    def __init__(self, get_session_func=None):
        """
        Args:
            get_session_func: Function that returns a database session context manager
        """
        self._get_session = get_session_func

    async def execute(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
        timeout_seconds: int = 3600
    ) -> ExecutionResult:
        """Execute a PostgreSQL function"""
        start_time = datetime.now(timezone.utc)

        function_name = config.get('function')
        params = config.get('params', {})

        if not function_name:
            return ExecutionResult(
                success=False,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error="No function name specified"
            )


        try:
            # Import here to avoid circular imports
            if self._get_session is None:
                from src.database.session import get_session
                self._get_session = get_session

            # Run synchronous DB operation in thread pool
            loop = asyncio.get_event_loop()

            def run_sql():
                with self._get_session() as session:
                    # Build the function call
                    if params:
                        # Function with parameters (not implemented yet - simple case first)
                        query = text(f"SELECT {function_name}()")
                    else:
                        query = text(f"SELECT {function_name}()")

                    result = session.execute(query)
                    session.commit()

                    # Try to get scalar result
                    try:
                        scalar_result = result.scalar()
                        return {'result': scalar_result}
                    except Exception:
                        return {'result': 'completed'}

            # Run with timeout
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(_executor, run_sql),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                end_time = datetime.now(timezone.utc)
                return ExecutionResult(
                    success=False,
                    start_time=start_time,
                    end_time=end_time,
                    error=f"SQL function timed out after {timeout_seconds} seconds"
                )

            end_time = datetime.now(timezone.utc)

            return ExecutionResult(
                success=True,
                start_time=start_time,
                end_time=end_time,
                output=result
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            error_msg = f"Error executing SQL function {function_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ExecutionResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                error=error_msg
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate SQL executor configuration"""
        if 'function' not in config:
            return False

        # Function name should be a string
        if not isinstance(config['function'], str):
            return False

        return True
