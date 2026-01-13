"""
SSE Client
==========

Simulates frontend Server-Sent Events (SSE) consumption for testing.
"""

import aiohttp
import asyncio
import time
import json
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SSEEvent:
    """Single SSE event"""
    event_type: str
    timestamp_ms: int
    data: Optional[Dict[str, Any]] = None
    raw: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_type': self.event_type,
            'timestamp_ms': self.timestamp_ms,
            'data': self.data
        }


@dataclass
class SSEStream:
    """Complete SSE stream with metadata"""
    events: List[SSEEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration_ms: Optional[int] = None
    status_code: int = 200
    error: Optional[str] = None

    def complete(self):
        """Mark stream as complete"""
        self.end_time = time.time()
        self.total_duration_ms = int((self.end_time - self.start_time) * 1000)

    def get_event_by_type(self, event_type: str) -> Optional[SSEEvent]:
        """Get first event of given type"""
        for event in self.events:
            if event.event_type == event_type:
                return event
        return None

    def get_events_by_type(self, event_type: str) -> List[SSEEvent]:
        """Get all events of given type"""
        return [e for e in self.events if e.event_type == event_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'events': [
                {
                    'event_type': e.event_type,
                    'timestamp_ms': e.timestamp_ms,
                    'data': e.data
                }
                for e in self.events
            ],
            'total_duration_ms': self.total_duration_ms,
            'status_code': self.status_code,
            'error': self.error
        }


class SSEClient:
    """
    SSE client that simulates frontend streaming behavior.

    Parses SSE events and accumulates them for validation.
    """

    def __init__(self, base_url: str = "http://localhost:7999"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _parse_sse_line(self, line: str, start_time: float) -> Optional[SSEEvent]:
        """
        Parse a single SSE line.

        SSE format:
        data: {json_payload}

        Returns SSEEvent or None if not a data line.
        """
        line = line.strip()

        if not line.startswith('data: '):
            return None

        # Extract JSON payload
        json_str = line[6:]  # Remove "data: " prefix

        try:
            data = json.loads(json_str)
            event_type = data.get('event', 'unknown')

            # Calculate timestamp relative to start
            timestamp_ms = int((time.time() - start_time) * 1000)

            return SSEEvent(
                event_type=event_type,
                timestamp_ms=timestamp_ms,
                data=data,
                raw=line
            )
        except json.JSONDecodeError:
            # Not valid JSON - might be plain text event
            return SSEEvent(
                event_type='unknown',
                timestamp_ms=int((time.time() - start_time) * 1000),
                data={'text': json_str},
                raw=line
            )

    async def stream_request(
        self,
        endpoint: str,
        method: str = "POST",
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 180
    ) -> SSEStream:
        """
        Make HTTP request and parse SSE stream (or simulate if not SSE).

        For non-SSE endpoints, this wraps the response in a simulated SSE stream
        with 'started' and 'complete' events.
        """
        if not self.session:
            raise RuntimeError("SSEClient must be used as async context manager")

        url = f"{self.base_url}{endpoint}"
        stream = SSEStream()

        try:
            # Make request
            async with self.session.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                stream.status_code = response.status

                # Check if response is SSE (text/event-stream)
                content_type = response.headers.get('Content-Type', '')

                if 'text/event-stream' in content_type:
                    # Real SSE stream
                    async for line in response.content:
                        line_str = line.decode('utf-8')
                        event = self._parse_sse_line(line_str, stream.start_time)
                        if event:
                            stream.events.append(event)
                else:
                    # Regular JSON response - simulate SSE
                    # Add 'started' event
                    stream.events.append(SSEEvent(
                        event_type='started',
                        timestamp_ms=0,
                        data={'simulated': True}
                    ))

                    # Parse JSON response
                    response_data = await response.json()

                    # Add 'complete' event with response data
                    stream.events.append(SSEEvent(
                        event_type='complete',
                        timestamp_ms=int((time.time() - stream.start_time) * 1000),
                        data=response_data
                    ))

        except asyncio.TimeoutError:
            stream.error = f"Timeout after {timeout}s"
            stream.status_code = 408

        except aiohttp.ClientError as e:
            stream.error = f"Client error: {str(e)}"
            stream.status_code = 500

        except Exception as e:
            stream.error = f"Unexpected error: {str(e)}"
            stream.status_code = 500

        finally:
            stream.complete()

        return stream

    async def search(
        self,
        query: str,
        dashboard_id: str = "cprmv-practitioner",
        time_window_days: int = 30,
        max_results: int = 50,
        threshold: float = 0.43,
        generate_theme_summary: bool = False,
        extract_subthemes: bool = False,
        min_subtheme_silhouette_score: float = 0.15,
        **kwargs
    ) -> SSEStream:
        """Perform search request with SSE streaming"""
        return await self.stream_request(
            endpoint="/api/search",
            method="POST",
            json_data={
                'query': query,
                'dashboard_id': dashboard_id,
                'time_window_days': time_window_days,
                'max_results': max_results,
                'threshold': threshold,
                'generate_theme_summary': generate_theme_summary,
                'extract_subthemes': extract_subthemes,
                'min_subtheme_silhouette_score': min_subtheme_silhouette_score,
                **kwargs
            }
        )

    async def optimize_query(
        self,
        query: str,
        dashboard_id: str = "cprmv-practitioner"
    ) -> SSEStream:
        """Optimize query request"""
        return await self.stream_request(
            endpoint="/api/llm/optimize-query",
            method="POST",
            json_data={
                'query': query,
                'dashboard_id': dashboard_id
            }
        )

    async def encode_text(self, text: str) -> SSEStream:
        """Encode single text to embedding"""
        return await self.stream_request(
            endpoint="/api/embeddings/encode",
            method="POST",
            json_data={'text': text}
        )

    async def encode_batch(self, texts: List[str]) -> SSEStream:
        """Encode multiple texts to embeddings"""
        return await self.stream_request(
            endpoint="/api/embeddings/batch",
            method="POST",
            json_data={'texts': texts}
        )

    async def health_check(self) -> SSEStream:
        """Basic health check"""
        return await self.stream_request(
            endpoint="/health",
            method="GET"
        )

    async def health_db(self) -> SSEStream:
        """Database health check"""
        return await self.stream_request(
            endpoint="/health/db",
            method="GET"
        )

    async def health_models(self) -> SSEStream:
        """Models health check"""
        return await self.stream_request(
            endpoint="/health/models",
            method="GET"
        )


# Helper function for synchronous usage
def create_sse_client(base_url: str = "http://localhost:7999") -> SSEClient:
    """Create SSE client instance"""
    return SSEClient(base_url=base_url)
