from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in {None, ""} else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in {None, ""} else float(value)


def _get_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return None if value in {None, ""} else int(value)


@dataclass(frozen=True)
class Settings:
    azure_openai_api_key: str | None
    azure_openai_endpoint: str | None
    azure_openai_api_version: str | None
    azure_openai_deployment: str | None
    azure_openai_summary_deployment: str | None
    tavily_api_key: str | None
    model: str
    summary_model: str
    output_version: str
    context_window_tokens: int
    summary_trigger_fraction: float
    summary_keep_messages: int
    request_timeout_seconds: float
    shell_timeout_seconds: float
    shell_startup_timeout_seconds: float
    shell_termination_timeout_seconds: float
    shell_max_output_lines: int
    shell_max_output_bytes: int | None
    shell_program: str
    tavily_max_results: int

    @classmethod
    def from_env(cls) -> "Settings":
        raw_azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")
        model = os.getenv("AZURE_OPENAI_MODEL", deployment)
        summary_deployment = os.getenv("AZURE_OPENAI_SUMMARY_DEPLOYMENT", deployment)
        summary_model = os.getenv("AZURE_OPENAI_SUMMARY_MODEL", model)

        return cls(
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_openai_endpoint=_normalize_azure_endpoint(raw_azure_endpoint),
            azure_openai_api_version=(
                os.getenv("AZURE_OPENAI_API_VERSION")
                or _extract_azure_api_version(raw_azure_endpoint)
            ),
            azure_openai_deployment=deployment,
            azure_openai_summary_deployment=summary_deployment,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            model=model,
            summary_model=summary_model,
            output_version=os.getenv("OPENAI_OUTPUT_VERSION", "responses/v1"),
            context_window_tokens=_get_int("OPENAI_CONTEXT_WINDOW_TOKENS", 400_000),
            summary_trigger_fraction=_get_float("NIRVANA_SUMMARY_TRIGGER_FRACTION", 0.9),
            summary_keep_messages=_get_int("NIRVANA_SUMMARY_KEEP_MESSAGES", 12),
            request_timeout_seconds=_get_float("OPENAI_REQUEST_TIMEOUT_SECONDS", 180.0),
            shell_timeout_seconds=_get_float("NIRVANA_SHELL_TIMEOUT_SECONDS", 3600.0),
            shell_startup_timeout_seconds=_get_float(
                "NIRVANA_SHELL_STARTUP_TIMEOUT_SECONDS", 30.0
            ),
            shell_termination_timeout_seconds=_get_float(
                "NIRVANA_SHELL_TERMINATION_TIMEOUT_SECONDS", 10.0
            ),
            shell_max_output_lines=_get_int("NIRVANA_SHELL_MAX_OUTPUT_LINES", 300),
            shell_max_output_bytes=_get_optional_int("NIRVANA_SHELL_MAX_OUTPUT_BYTES"),
            shell_program=os.getenv("NIRVANA_SHELL_PROGRAM", "/bin/bash"),
            tavily_max_results=_get_int("TAVILY_MAX_RESULTS", 5),
        )


def _normalize_azure_endpoint(endpoint: str | None) -> str | None:
    if endpoint in {None, ""}:
        return None

    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"

    return endpoint


def _extract_azure_api_version(endpoint: str | None) -> str | None:
    if endpoint in {None, ""}:
        return None

    parsed = urlparse(endpoint)
    versions = parse_qs(parsed.query).get("api-version")
    return versions[0] if versions else None
