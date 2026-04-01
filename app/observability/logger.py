"""
Observability layer:
  - structlog for structured JSON logging
  - AgentCallbackHandler: captures every LLM/tool step for tracing
  - OpenTelemetry spans (optional, activated when OTEL_EXPORTER_OTLP_ENDPOINT is set)
"""
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def setup_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if os.getenv("DEBUG", "false").lower() == "true"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger()


class AgentCallbackHandler(BaseCallbackHandler):
    """Captures full agent reasoning trace for response metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: List[Dict[str, Any]] = []
        self.sources_used: List[str] = []
        self._current_tool: Optional[str] = None

    # ── LLM hooks ─────────────────────────────────────────────────────────
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        logger.debug("llm_start", model=serialized.get("name", "unknown"))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage", {})
        logger.info("llm_end", token_usage=usage)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        logger.error("llm_error", error=str(error))

    # ── Tool hooks ────────────────────────────────────────────────────────
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self._current_tool = serialized.get("name", "unknown")
        logger.info("tool_start", tool=self._current_tool, input=input_str[:300])
        self.steps.append(
            {"type": "tool_call", "tool": self._current_tool, "input": input_str}
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        # LangGraph passes a ToolMessage object; extract the text content
        if hasattr(output, "content"):
            output_str = output.content if isinstance(output.content, str) else str(output.content)
        else:
            output_str = str(output)

        logger.info("tool_end", tool=self._current_tool, output_len=len(output_str))
        if self.steps and self.steps[-1].get("type") == "tool_call":
            self.steps[-1]["output_preview"] = output_str[:400]
        # Extract source citations from tool output
        for line in output_str.split("\n"):
            if line.strip().startswith("Source:"):
                src = line.strip().replace("Source:", "").strip()
                if src not in self.sources_used:
                    self.sources_used.append(src)

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        logger.error("tool_error", tool=self._current_tool, error=str(error))

    # ── Agent hooks ───────────────────────────────────────────────────────
    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        logger.info(
            "agent_action", tool=action.tool, thought=str(action.log)[:200]
        )

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        logger.info(
            "agent_finish",
            output=finish.return_values.get("output", "")[:200],
        )

    def get_trace(self) -> Dict[str, Any]:
        return {"steps": self.steps, "sources_cited": self.sources_used}
