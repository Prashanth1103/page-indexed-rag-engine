import os
from typing import Optional
from opentelemetry import trace


def configure_arize_phoenix_tracing(
    project_name: str = "deep-researcher-page-rag",
    collector_endpoint: Optional[str] = None,
) -> trace.Tracer:
    """
    Registers Arize Phoenix OpenTelemetry TracerProvider for visualizing
    multi-agent CrewAI execution graphs, per-stage memoization hits, and GPT-4o latency.
    """
    endpoint = collector_endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
    try:
        from phoenix.otel import register

        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
        )
        return trace.get_tracer(project_name, tracer_provider=tracer_provider)
    except Exception:
        return trace.get_tracer(project_name)
