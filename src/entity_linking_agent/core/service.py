"""Application service for Topic 10."""

from __future__ import annotations

from typing import Optional

from entity_linking_agent.core.alias_prior import AliasPrior
from entity_linking_agent.config import AppConfig, load_config
from entity_linking_agent.core.contracts import KnowledgeBaseEntity, LinkOptions, MentionRecord
from entity_linking_agent.core.linker import EntityLinker
from entity_linking_agent.core.retriever import CandidateRetriever
from entity_linking_agent.core.trace_store import TraceRepository
from entity_linking_agent.core.workflow import EntityLinkingWorkflow
from entity_linking_agent.kb.loader import KnowledgeBaseLoader
from entity_linking_agent.utils.tracing import build_trace_id


class Topic10EntityLinkingService:
    """High-level orchestration service."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()
        self.kb_loader = KnowledgeBaseLoader(self.config)
        self.trace_repository = TraceRepository(self.config.traces_dir)
        self.retriever = CandidateRetriever()
        self.linker = EntityLinker(
            retriever=self.retriever,
            alias_prior=AliasPrior.load(self.config.alias_prior_path),
        )
        self.workflow = EntityLinkingWorkflow(
            kb_loader=self.kb_loader,
            trace_repository=self.trace_repository,
            retriever=self.retriever,
            linker=self.linker,
        )

    def link(
        self,
        text: str,
        mentions: list[MentionRecord],
        knowledge_base_id: Optional[str] = None,
        inline_entities: Optional[list[KnowledgeBaseEntity]] = None,
        options: Optional[LinkOptions] = None,
        trace_id: Optional[str] = None,
    ) -> dict:
        effective_options = options or LinkOptions(top_k_candidates=self.config.default_top_k_candidates)
        effective_trace_id = trace_id or build_trace_id(self.config.trace_prefix)
        return self.workflow.invoke(
            {
                "text": text,
                "mentions": mentions,
                "knowledge_base_id": knowledge_base_id,
                "inline_entities": inline_entities,
                "options": effective_options,
                "trace_id": effective_trace_id,
                "node_events": [],
            }
        )

    def list_builtin_kbs(self) -> dict[str, str]:
        return self.kb_loader.list_builtin_kbs()

    def get_trace(self, trace_id: str) -> dict:
        return self.trace_repository.read_trace(trace_id)
