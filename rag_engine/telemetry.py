import functools
import inspect
import time
from typing import Any, Callable
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import InMemorySpanExporter, SimpleSpanProcessor

_provider = TracerProvider()
exporter = InMemorySpanExporter()
_provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer("page.indexed.rag")


def telemetry(span_name: str, layer: str = "agent") -> Callable:
    """OpenTelemetry span decorator compatible with Arize Phoenix agent/tool/workflow tracing."""

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter_ns()
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute("openinference.span.kind", layer.upper())
                    span.set_attribute("fn.name", fn.__name__)
                    try:
                        res = await fn(*args, **kwargs)
                        span.set_attribute("status", "ok")
                        return res
                    finally:
                        span.set_attribute("latency_ms", round((time.perf_counter_ns() - t0) / 1e6, 3))

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter_ns()
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("openinference.span.kind", layer.upper())
                span.set_attribute("fn.name", fn.__name__)
                try:
                    res = fn(*args, **kwargs)
                    span.set_attribute("status", "ok")
                    return res
                finally:
                    span.set_attribute("latency_ms", round((time.perf_counter_ns() - t0) / 1e6, 3))

        return sync_wrapper

    return decorator
