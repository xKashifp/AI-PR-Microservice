from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time

_request_count = 0
_request_latencies = []

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global _request_count
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        _request_count += 1
        _request_latencies.append(elapsed)
        response.headers["X-Request-Count"] = str(_request_count)
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
        return response

def get_metrics():
    import numpy as np
    if not _request_latencies:
        return {"request_count": _request_count, "p95_latency_ms": None}
    p95 = float(np.percentile(_request_latencies, 95)) * 1000
    return {
        "request_count": _request_count,
        "p95_latency_ms": round(p95, 2)
    }
