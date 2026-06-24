from __future__ import annotations

import warnings
from typing import Any, Optional, TypedDict

warnings.filterwarnings("ignore", message=r"The default value of `allowed_objects` will change.*", category=Warning)

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from app.config import AppConfig
from app.core import candidate as candidate_mod
from app.core import coreference as coref_mod
from app.core import evidence as evidence_mod
from app.core import nil_detector
from app.core.scorer import AliasPrior, rescore
from app.models.enums import EvidenceType, LinkStatus
from app.models.request import LinkOptions, LinkRequest, MentionInput
from app.models.response import (
    CandidateItem,
    CoreferenceChain,
    EvidenceItem,
    EntityRef,
    LinkResponse,
    LinkResult,
    LinkSummary,
    LinkTrace,
)
from app.storage.index import NameIndex
from app.storage.kb_store import KBStore


class LinkState(TypedDict, total=False):
    request: LinkRequest
    entities_index: NameIndex
    candidates_by_id: dict[str, list[candidate_mod.CandidateResult]]
    results: list[LinkResult]
    coref_chains: list[CoreferenceChain]
    validation_error: Optional[str]


class LinkService:
    def __init__(self, store: KBStore, config: AppConfig, alias_prior: Optional[AliasPrior] = None) -> None:
        self.store = store
        self.config = config
        self.alias_prior = alias_prior
        self.graph = self._build_graph()

    def link(self, request: LinkRequest) -> LinkResponse:
        state = self.graph.invoke({"request": request})
        if state.get("validation_error"):
            from app.models.response import ErrorDetail, ErrorResponse
            raise ValueError(state["validation_error"])
        return self._build_response(request, state)

    def _build_graph(self):
        builder = StateGraph(LinkState)
        builder.add_node("validate", RunnableLambda(self._validate, name="validate"))
        builder.add_node("load_kb", RunnableLambda(self._load_kb, name="load_kb"))
        builder.add_node("generate_candidates", RunnableLambda(self._generate_candidates, name="generate_candidates"))
        builder.add_node("nil_fallback", RunnableLambda(self._nil_fallback, name="nil_fallback"))
        builder.add_node("rerank", RunnableLambda(self._rerank, name="rerank"))
        builder.add_node("resolve", RunnableLambda(self._resolve, name="resolve"))
        builder.add_node("coreference", RunnableLambda(self._coreference, name="coreference"))

        builder.add_edge(START, "validate")
        builder.add_conditional_edges("validate", self._route_validate, {"ok": "load_kb", "error": END})
        builder.add_edge("load_kb", "generate_candidates")
        builder.add_conditional_edges(
            "generate_candidates",
            self._route_candidates,
            {"has_candidates": "rerank", "empty": "nil_fallback"},
        )
        builder.add_edge("nil_fallback", END)
        builder.add_edge("rerank", "resolve")
        builder.add_edge("resolve", "coreference")
        builder.add_edge("coreference", END)
        return builder.compile()

    def _validate(self, state: LinkState) -> dict:
        req = state["request"]
        if not req.text.content.strip():
            return {"validation_error": "MISSING_FIELD:text.content cannot be empty"}
        if not req.mentions:
            return {"validation_error": "EMPTY_MENTIONS:mentions cannot be empty"}
        kb_ref = req.knowledge_base
        if not self.store.exists(kb_ref.kb_id):
            return {"validation_error": f"KB_NOT_FOUND:{kb_ref.kb_id}"}
        meta = self.store.get_meta(kb_ref.kb_id)
        if meta and meta.kb_version != kb_ref.kb_version:
            return {"validation_error": f"KB_VERSION_MISMATCH:expected {meta.kb_version}"}
        return {}

    def _route_validate(self, state: LinkState) -> str:
        return "error" if state.get("validation_error") else "ok"

    def _load_kb(self, state: LinkState) -> dict:
        kb_id = state["request"].knowledge_base.kb_id
        entities = self.store.load_entities(kb_id)
        return {"entities_index": NameIndex(entities)}

    def _generate_candidates(self, state: LinkState) -> dict:
        req = state["request"]
        index: NameIndex = state["entities_index"]
        entities = index.all_entities()
        result = {
            m.mention_id: candidate_mod.retrieve(m, index, entities, req.options.top_k)
            for m in req.mentions
        }
        return {"candidates_by_id": result}

    def _route_candidates(self, state: LinkState) -> str:
        return "has_candidates" if any(state["candidates_by_id"].values()) else "empty"

    def _nil_fallback(self, state: LinkState) -> dict:
        results = [
            LinkResult(
                mention_id=m.mention_id,
                surface_form=m.surface_form,
                link_status=LinkStatus.nil,
            )
            for m in state["request"].mentions
        ]
        return {"results": results, "coref_chains": []}

    def _rerank(self, state: LinkState) -> dict:
        req = state["request"]
        context = req.text.content
        rescored: dict[str, list[candidate_mod.CandidateResult]] = {}
        for m in req.mentions:
            cands = state["candidates_by_id"].get(m.mention_id, [])
            rescored[m.mention_id] = sorted(
                [rescore(c, m, context, self.alias_prior) for c in cands],
                key=lambda c: c.score, reverse=True,
            )[:req.options.top_k]
        return {"candidates_by_id": rescored}

    def _resolve(self, state: LinkState) -> dict:
        req = state["request"]
        results: list[LinkResult] = []
        for m in req.mentions:
            cands = state["candidates_by_id"].get(m.mention_id, [])
            status, top = nil_detector.decide(cands, req.options)
            if status == "nil" or top is None:
                results.append(LinkResult(
                    mention_id=m.mention_id,
                    surface_form=m.surface_form,
                    link_status=LinkStatus.nil,
                    candidates=self._fmt_candidates(cands, req.options),
                ))
            else:
                evid = evidence_mod.build_evidence(top, req.text.content) if req.options.return_evidence else []
                results.append(LinkResult(
                    mention_id=m.mention_id,
                    surface_form=m.surface_form,
                    link_status=LinkStatus(status),
                    entity=EntityRef(
                        entity_id=top.entity.entity_id,
                        canonical_name=top.entity.canonical_name,
                        entity_type=top.entity.entity_type,
                    ),
                    confidence=top.score,
                    candidates=self._fmt_candidates(cands, req.options),
                    evidence=evid,
                ))
        return {"results": results}

    def _coreference(self, state: LinkState) -> dict:
        req = state["request"]
        if not req.options.enable_coreference:
            return {"coref_chains": []}
        updated, chains = coref_mod.resolve(
            state["results"], req.mentions, self.config.coreference_terms
        )
        # attach coreference evidence
        final = []
        for r in updated:
            if r.coreference and req.options.return_evidence:
                evid = list(r.evidence) + [EvidenceItem(
                    evidence_type=EvidenceType.coreference,
                    detail=f"该 mention 回指前文 mention {r.coreference.resolved_from}",
                )]
                r = r.model_copy(update={"evidence": evid})
            final.append(r)
        return {"results": final, "coref_chains": chains}

    @staticmethod
    def _fmt_candidates(cands: list[candidate_mod.CandidateResult], options: LinkOptions) -> list[CandidateItem]:
        if not options.return_candidates:
            return []
        return [CandidateItem(entity_id=c.entity.entity_id, canonical_name=c.entity.canonical_name, score=c.score) for c in cands]

    def _build_response(self, request: LinkRequest, state: LinkState) -> LinkResponse:
        results = state.get("results", [])
        chains = state.get("coref_chains", [])
        linked = sum(1 for r in results if r.link_status == LinkStatus.linked)
        nil = sum(1 for r in results if r.link_status == LinkStatus.nil)
        opts = request.options
        return LinkResponse(
            request_id=request.request_id,
            status="success",
            results=results,
            coreference_chains=chains,
            summary=LinkSummary(total_mentions=len(results), linked_count=linked, nil_count=nil),
            trace=LinkTrace(
                linker_version=opts.linker_version,
                kb_id=request.knowledge_base.kb_id,
                kb_version=request.knowledge_base.kb_version,
                options_used={"top_k": opts.top_k, "nil_threshold": opts.nil_threshold,
                              "enable_nil": opts.enable_nil, "enable_coreference": opts.enable_coreference},
            ),
        )
