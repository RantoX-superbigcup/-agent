"""LangGraph workflow for terminal dialogue understanding."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, TypedDict

warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change.*",
    category=Warning,
)

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from entity_linking_agent.llm.deepseek_client import (
    DeepSeekAPIError,
    DeepSeekChatClient,
    normalize_turn_action,
)

RuleParser = Callable[[str, dict[str, Any]], dict[str, Any]]


class DialogueState(TypedDict, total=False):
    user_text: str
    current_state: dict[str, Any]
    llm_action: dict[str, Any]
    action: dict[str, Any]
    route: str
    node_events: list[dict[str, Any]]


class DialogueWorkflow:
    """Use DeepSeek first, then deterministic rules as a safe fallback."""

    def __init__(
        self,
        llm_client: Optional[DeepSeekChatClient] = None,
        use_llm: bool = True,
        rule_parser: Optional[RuleParser] = None,
    ) -> None:
        self.llm_client = llm_client or DeepSeekChatClient()
        self.use_llm = use_llm and bool(getattr(self.llm_client, "is_configured", False))
        self.rule_parser = rule_parser or self._default_rule_parser
        self.graph = self._build_graph()

    def invoke(self, user_text: str, current_state: dict[str, Any]) -> dict[str, Any]:
        final_state = self.graph.invoke(
            {
                "user_text": user_text,
                "current_state": current_state,
                "node_events": [],
            }
        )
        return {
            "action": final_state["action"],
            "route": final_state.get("route", "unknown"),
            "node_events": final_state.get("node_events", []),
        }

    def _build_graph(self):
        builder = StateGraph(DialogueState)
        builder.add_node("llm_understand", RunnableLambda(self._llm_understand, name="llm_understand"))
        builder.add_node("rule_fallback", RunnableLambda(self._rule_fallback, name="rule_fallback"))
        builder.add_node("finalize_action", RunnableLambda(self._finalize_action, name="finalize_action"))

        builder.add_edge(START, "llm_understand")
        builder.add_conditional_edges(
            "llm_understand",
            self._route_after_llm,
            {
                "accepted": "finalize_action",
                "fallback": "rule_fallback",
            },
        )
        builder.add_edge("rule_fallback", END)
        builder.add_edge("finalize_action", END)
        return builder.compile()

    def _llm_understand(self, state: DialogueState) -> dict[str, Any]:
        if not self.use_llm:
            return {
                "node_events": self._append_event(
                    state,
                    "llm_understand",
                    {"enabled": False, "accepted": False, "reason": "llm_disabled_or_missing_key"},
                )
            }

        try:
            action = self.llm_client.analyze_turn(
                user_text=state["user_text"],
                current_state=state["current_state"],
            )
        except DeepSeekAPIError as exc:
            return {
                "node_events": self._append_event(
                    state,
                    "llm_understand",
                    {"enabled": True, "accepted": False, "reason": str(exc)[:160]},
                )
            }

        accepted = action["action"] != "unknown" and action["confidence"] >= 0.45
        return {
            "llm_action": action,
            "node_events": self._append_event(
                state,
                "llm_understand",
                {
                    "enabled": True,
                    "accepted": accepted,
                    "action": action["action"],
                    "confidence": action["confidence"],
                },
            ),
        }

    def _route_after_llm(self, state: DialogueState) -> str:
        action = state.get("llm_action")
        if action and action["action"] != "unknown" and action["confidence"] >= 0.45:
            return "accepted"
        return "fallback"

    def _rule_fallback(self, state: DialogueState) -> dict[str, Any]:
        action = normalize_turn_action(self.rule_parser(state["user_text"], state["current_state"]))
        return {
            "action": action,
            "route": "rules",
            "node_events": self._append_event(
                state,
                "rule_fallback",
                {"action": action["action"], "run_requested": action["run_requested"]},
            ),
        }

    def _finalize_action(self, state: DialogueState) -> dict[str, Any]:
        action = normalize_turn_action(state["llm_action"])
        return {
            "action": action,
            "route": "deepseek",
            "node_events": self._append_event(
                state,
                "finalize_action",
                {"action": action["action"], "run_requested": action["run_requested"]},
            ),
        }

    @staticmethod
    def _append_event(state: DialogueState, node: str, detail: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            *state.get("node_events", []),
            {
                "node": node,
                "detail": detail,
            },
        ]

    @staticmethod
    def _default_rule_parser(user_text: str, current_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": "unknown",
            "kb_id": None,
            "text": None,
            "mentions": [],
            "run_requested": False,
            "reply": None,
            "confidence": 0.0,
        }
