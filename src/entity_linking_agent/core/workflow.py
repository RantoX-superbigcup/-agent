"""LangChain + LangGraph workflow for Topic 10."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, TypedDict
import warnings

from langchain_core.runnables import RunnableLambda

warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change.*",
    category=Warning,
)

from langgraph.graph import END, START, StateGraph

from entity_linking_agent.core.contracts import (
    CandidateScore,
    EvidenceRecord,
    KnowledgeBaseEntity,
    KnowledgeBaseSnapshot,
    LinkDecision,
    LinkOptions,
    MentionRecord,
)
from entity_linking_agent.core.linker import EntityLinker
from entity_linking_agent.core.retriever import CandidateRetriever
from entity_linking_agent.core.trace_store import TraceRepository
from entity_linking_agent.kb.loader import KnowledgeBaseLoader
from entity_linking_agent.utils.text import extract_context, normalize_text


class EntityLinkingState(TypedDict, total=False):
    text: str
    mentions: list[MentionRecord]
    knowledge_base_id: Optional[str]
    inline_entities: Optional[list[KnowledgeBaseEntity]]
    options: LinkOptions
    trace_id: str
    kb_snapshot: KnowledgeBaseSnapshot
    candidates_by_mention_id: dict[str, list[CandidateScore]]
    rescored_by_mention_id: dict[str, list[CandidateScore]]
    results: list[LinkDecision]
    payload: dict[str, Any]
    node_events: list[dict[str, Any]]
    validation_errors: list[str]
    route_decision: str


class EntityLinkingWorkflow:
    """Graph orchestration layer for entity linking."""

    def __init__(
        self,
        kb_loader: KnowledgeBaseLoader,
        trace_repository: TraceRepository,
        retriever: Optional[CandidateRetriever] = None,
        linker: Optional[EntityLinker] = None,
    ) -> None:
        self.kb_loader = kb_loader
        self.trace_repository = trace_repository
        self.retriever = retriever or CandidateRetriever()
        self.linker = linker or EntityLinker(self.retriever)
        self.graph = self._build_graph()

    def invoke(self, initial_state: EntityLinkingState) -> dict[str, Any]:
        final_state = self.graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": initial_state["trace_id"]}},
        )
        return final_state["payload"]

    def _build_graph(self):
        builder = StateGraph(EntityLinkingState)
        builder.add_node("validate_input", RunnableLambda(self._validate_input, name="validate_input"))
        builder.add_node("load_kb", RunnableLambda(self._load_kb, name="load_kb"))
        builder.add_node(
            "generate_candidates",
            RunnableLambda(self._generate_candidates, name="generate_candidates"),
        )
        builder.add_node("nil_fallback", RunnableLambda(self._nil_fallback, name="nil_fallback"))
        builder.add_node(
            "rerank_candidates",
            RunnableLambda(self._rerank_candidates, name="rerank_candidates"),
        )
        builder.add_node(
            "resolve_mentions",
            RunnableLambda(self._resolve_mentions, name="resolve_mentions"),
        )
        builder.add_node("human_review", RunnableLambda(self._human_review, name="human_review"))
        builder.add_node("build_response", RunnableLambda(self._build_response, name="build_response"))
        builder.add_node("persist_trace", RunnableLambda(self._persist_trace, name="persist_trace"))

        builder.add_edge(START, "validate_input")
        builder.add_conditional_edges(
            "validate_input",
            self._input_route,
            {
                "ok": "load_kb",
                "invalid": "build_response",
            },
        )
        builder.add_edge("load_kb", "generate_candidates")
        builder.add_conditional_edges(
            "generate_candidates",
            self._candidate_route,
            {
                "has_candidates": "rerank_candidates",
                "empty_candidates": "nil_fallback",
            },
        )
        builder.add_edge("nil_fallback", "build_response")
        builder.add_edge("rerank_candidates", "resolve_mentions")
        builder.add_conditional_edges(
            "resolve_mentions",
            self._review_route,
            {
                "auto_accept": "build_response",
                "needs_review": "human_review",
            },
        )
        builder.add_edge("human_review", "build_response")
        builder.add_edge("build_response", "persist_trace")
        builder.add_edge("persist_trace", END)
        return builder.compile()

    def _validate_input(self, state: EntityLinkingState) -> dict[str, Any]:
        errors: list[str] = []
        if not state.get("text", "").strip():
            errors.append("text_required")
        mentions = state.get("mentions", [])
        if not mentions:
            errors.append("mentions_required")
        for mention in mentions:
            if not mention.mention_id:
                errors.append("mention_id_required")
            if not mention.text.strip():
                errors.append(f"mention_text_required:{mention.mention_id}")
            if mention.start is not None and mention.start < 0:
                errors.append(f"mention_start_invalid:{mention.mention_id}")
            if mention.end is not None and mention.end < 0:
                errors.append(f"mention_end_invalid:{mention.mention_id}")
            if mention.start is not None and mention.end is not None and mention.start > mention.end:
                errors.append(f"mention_span_invalid:{mention.mention_id}")

        options = state["options"]
        if options.top_k_candidates < 1:
            errors.append("top_k_candidates_must_be_positive")
        if not 0 <= options.nil_threshold <= 1:
            errors.append("nil_threshold_out_of_range")
        if not 0 <= options.ambiguity_margin <= 1:
            errors.append("ambiguity_margin_out_of_range")

        return {
            "validation_errors": errors,
            "route_decision": "invalid_input" if errors else state.get("route_decision", "validated"),
            "node_events": self._append_event(
                state,
                node="validate_input",
                detail={"valid": not errors, "errors": errors},
            ),
        }

    def _input_route(self, state: EntityLinkingState) -> str:
        return "invalid" if state.get("validation_errors") else "ok"

    def _load_kb(self, state: EntityLinkingState) -> dict[str, Any]:
        snapshot = self.kb_loader.load(
            knowledge_base_id=state.get("knowledge_base_id"),
            inline_entities=state.get("inline_entities"),
        )
        return {
            "kb_snapshot": snapshot,
            "node_events": self._append_event(
                state,
                node="load_kb",
                detail={"kb_id": snapshot.kb_id, "entity_count": len(snapshot.entities)},
            ),
        }

    def _generate_candidates(self, state: EntityLinkingState) -> dict[str, Any]:
        snapshot = state["kb_snapshot"]
        options = state["options"]
        candidates_by_mention_id = {
            mention.mention_id: self.retriever.retrieve(
                mention=mention,
                entities=snapshot.entities,
                top_k=max(options.top_k_candidates, 1),
            )
            for mention in state["mentions"]
        }
        return {
            "candidates_by_mention_id": candidates_by_mention_id,
            "node_events": self._append_event(
                state,
                node="generate_candidates",
                detail={
                    "mention_count": len(state["mentions"]),
                    "candidate_count": sum(len(items) for items in candidates_by_mention_id.values()),
                    "empty_mentions": [
                        mention_id
                        for mention_id, candidates in candidates_by_mention_id.items()
                        if not candidates
                    ],
                },
            ),
        }

    def _candidate_route(self, state: EntityLinkingState) -> str:
        candidates_by_mention_id = state.get("candidates_by_mention_id", {})
        has_any_candidate = any(candidates_by_mention_id.get(mention.mention_id) for mention in state["mentions"])
        return "has_candidates" if has_any_candidate else "empty_candidates"

    def _nil_fallback(self, state: EntityLinkingState) -> dict[str, Any]:
        results = [
            LinkDecision(
                mention_id=mention.mention_id,
                text=mention.text,
                entity_type=mention.entity_type,
                linked_entity_id=None,
                canonical_name=None,
                status="nil",
                confidence=0.0,
                needs_review=False,
                candidates=[],
                evidence=EvidenceRecord(
                    normalized_mention=normalize_text(mention.text),
                    matched_alias=None,
                    context_snippet=extract_context(
                        text=state["text"],
                        start=mention.start,
                        end=mention.end,
                        fallback=mention.sentence or state["text"],
                    ),
                    rationale=["candidate_route_empty", "nil_fallback"],
                ),
            )
            for mention in state["mentions"]
        ]
        return {
            "results": results,
            "route_decision": "empty_candidates",
            "node_events": self._append_event(
                state,
                node="nil_fallback",
                detail={"nil_mentions": len(results), "reason": "no_candidates_for_all_mentions"},
            ),
        }

    def _rerank_candidates(self, state: EntityLinkingState) -> dict[str, Any]:
        snapshot = state["kb_snapshot"]
        options = state["options"]
        entity_index = {entity.entity_id: entity for entity in snapshot.entities}
        rescored_by_mention_id: dict[str, list[CandidateScore]] = {}

        for mention in state["mentions"]:
            context = extract_context(
                text=state["text"],
                start=mention.start,
                end=mention.end,
                fallback=mention.sentence or state["text"],
            )
            candidates = state["candidates_by_mention_id"].get(mention.mention_id, [])
            rescored_by_mention_id[mention.mention_id] = self.linker.rescore_candidates(
                mention=mention,
                context=context,
                candidates=candidates,
                entity_index=entity_index,
                top_k=options.top_k_candidates,
            )

        return {
            "rescored_by_mention_id": rescored_by_mention_id,
            "node_events": self._append_event(
                state,
                node="rerank_candidates",
                detail={"reranked_mentions": len(rescored_by_mention_id)},
            ),
        }

    def _resolve_mentions(self, state: EntityLinkingState) -> dict[str, Any]:
        history: list[LinkDecision] = []
        results: list[LinkDecision] = []

        for mention in state["mentions"]:
            context = extract_context(
                text=state["text"],
                start=mention.start,
                end=mention.end,
                fallback=mention.sentence or state["text"],
            )
            decision = self.linker.decide_mention(
                mention=mention,
                context=context,
                candidates=state["rescored_by_mention_id"].get(mention.mention_id, []),
                options=state["options"],
                history=history,
            )
            results.append(decision)
            if decision.linked_entity_id:
                history.append(decision)

        return {
            "results": results,
            "route_decision": "resolved",
            "node_events": self._append_event(
                state,
                node="resolve_mentions",
                detail={
                    "linked": sum(1 for item in results if item.status == "linked"),
                    "nil": sum(1 for item in results if item.status == "nil"),
                    "ambiguous": sum(1 for item in results if item.status == "ambiguous"),
                },
            ),
        }

    def _review_route(self, state: EntityLinkingState) -> str:
        results = state.get("results", [])
        options = state["options"]
        needs_review = any(
            item.needs_review
            or item.status == "ambiguous"
            or (bool(item.candidates) and item.confidence < options.nil_threshold)
            for item in results
        )
        return "needs_review" if needs_review else "auto_accept"

    def _human_review(self, state: EntityLinkingState) -> dict[str, Any]:
        reviewed_results: list[LinkDecision] = []
        review_items: list[dict[str, Any]] = []

        for item in state["results"]:
            if item.needs_review or item.status == "ambiguous":
                item.needs_review = True
                item.evidence.rationale.append("human_review_required")
                review_items.append(
                    {
                        "mention_id": item.mention_id,
                        "status": item.status,
                        "confidence": item.confidence,
                    }
                )
            reviewed_results.append(item)

        return {
            "results": reviewed_results,
            "route_decision": "needs_review",
            "node_events": self._append_event(
                state,
                node="human_review",
                detail={
                    "review_required": len(review_items),
                    "items": review_items,
                    "policy": "low_confidence_or_ambiguous",
                },
            ),
        }

    def _build_response(self, state: EntityLinkingState) -> dict[str, Any]:
        snapshot = state.get("kb_snapshot")
        results = state.get("results", [])
        node_events = self._append_event(
            state,
            node="build_response",
            detail={"total_mentions": len(results), "route_decision": state.get("route_decision", "unknown")},
        )
        payload = {
            "trace_id": state["trace_id"],
            "kb_id": snapshot.kb_id if snapshot is not None else state.get("knowledge_base_id") or "unknown",
            "kb_version": snapshot.version if snapshot is not None else "unloaded",
            "workflow_engine": "langgraph",
            "graph_nodes": [event["node"] for event in node_events],
            "route_decision": state.get("route_decision", "unknown"),
            "validation_errors": state.get("validation_errors", []),
            "node_events": node_events,
            "results": [asdict(item) for item in results],
            "summary": self._build_summary(results),
            "decision_log": self._build_decision_log(results, state["trace_id"]),
        }
        return {
            "payload": payload,
            "node_events": node_events,
        }

    def _persist_trace(self, state: EntityLinkingState) -> dict[str, Any]:
        payload = dict(state["payload"])
        node_events = self._append_event(
            state,
            node="persist_trace",
            detail={"trace_persisted": True},
        )
        payload["node_events"] = node_events
        payload["graph_nodes"] = [event["node"] for event in node_events]
        payload["trace_persisted"] = True
        if not self.trace_repository.write_trace(payload):
            payload["trace_persisted"] = False
            node_events[-1]["detail"]["trace_persisted"] = False
            payload["node_events"] = node_events
        return {
            "payload": payload,
            "node_events": node_events,
        }

    @staticmethod
    def _append_event(state: EntityLinkingState, node: str, detail: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            *state.get("node_events", []),
            {
                "node": node,
                "detail": detail,
            },
        ]

    @staticmethod
    def _build_summary(results: list[LinkDecision]) -> dict[str, int]:
        return {
            "total_mentions": len(results),
            "linked": sum(1 for item in results if item.status == "linked"),
            "ambiguous": sum(1 for item in results if item.status == "ambiguous"),
            "nil": sum(1 for item in results if item.status == "nil"),
            "review_required": sum(1 for item in results if item.needs_review),
        }

    @staticmethod
    def _build_decision_log(results: list[LinkDecision], trace_id: str) -> list[dict]:
        return [
            {
                "trace_id": trace_id,
                "mention_id": item.mention_id,
                "status": item.status,
                "linked_entity_id": item.linked_entity_id,
                "confidence": item.confidence,
                "needs_review": item.needs_review,
            }
            for item in results
        ]
