from __future__ import annotations
import logging
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
    CandidateItem, CoreferenceChain, EvidenceItem, EntityRef,
    LinkResponse, LinkResult, LinkSummary, LinkTrace,
)
from app.storage.index import NameIndex
from app.storage.kb_store import KBStore
from app.storage.vector_index import VectorIndex

logger = logging.getLogger("entity_link_agent")


class LinkState(TypedDict, total=False):
    request: LinkRequest
    entities_index: NameIndex
    vector_index: Optional[VectorIndex]
    candidates_by_id: dict[str, list[candidate_mod.CandidateResult]]
    results: list[LinkResult]
    coref_chains: list[CoreferenceChain]
    validation_error: Optional[str]


class LinkService:
    def __init__(self, store: KBStore, config: AppConfig, alias_prior: Optional[AliasPrior] = None, embedder=None) -> None:
        self.store = store
        self.config = config
        self.alias_prior = alias_prior
        self.embedder = embedder
        self.graph = self._build_graph()

    def link(self, request: LinkRequest) -> LinkResponse:
        logger.info("=" * 60)
        logger.info("▶ 实体链接开始  request_id=%s  kb=%s/%s  mentions=%d",
                     request.request_id, request.knowledge_base.kb_id,
                     request.knowledge_base.kb_version, len(request.mentions))
        state = self.graph.invoke({"request": request})
        if state.get("validation_error"):
            logger.error("✘ 链接失败: %s", state["validation_error"])
            raise ValueError(state["validation_error"])
        resp = self._build_response(request, state)
        logger.info("✔ 实体链接完成  linked=%d  nil=%d",
                     resp.summary.linked_count, resp.summary.nil_count)
        logger.info("=" * 60)
        return resp

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
        builder.add_conditional_edges("generate_candidates", self._route_candidates, {"has_candidates": "rerank", "empty": "nil_fallback"})
        builder.add_edge("nil_fallback", END)
        builder.add_edge("rerank", "resolve")
        builder.add_edge("resolve", "coreference")
        builder.add_edge("coreference", END)
        return builder.compile()

    def _validate(self, state: LinkState) -> dict:
        req = state["request"]
        logger.info("  [1/6] 校验请求...")
        if not req.text.content.strip():
            return {"validation_error": "MISSING_FIELD:text.content cannot be empty"}
        if not req.mentions:
            return {"validation_error": "EMPTY_MENTIONS:mentions cannot be empty"}
        if not self.store.exists(req.knowledge_base.kb_id):
            return {"validation_error": f"KB_NOT_FOUND:{req.knowledge_base.kb_id}"}
        meta = self.store.get_meta(req.knowledge_base.kb_id)
        if meta and meta.kb_version != req.knowledge_base.kb_version:
            return {"validation_error": f"KB_VERSION_MISMATCH:expected {meta.kb_version}"}
        logger.info("  [1/6] 校验通过")
        return {}

    def _route_validate(self, state: LinkState) -> str:
        return "error" if state.get("validation_error") else "ok"

    def _load_kb(self, state: LinkState) -> dict:
        kb_id = state["request"].knowledge_base.kb_id
        logger.info("  [2/6] 加载知识库 %s...", kb_id)
        entities = self.store.load_entities(kb_id)
        logger.info("  [2/6] 已加载 %d 个实体", len(entities))
        vec_idx = None
        if self.embedder and self.config.index_dir:
            vec_idx = VectorIndex(kb_id, self.config.index_dir)
            if vec_idx.load():
                logger.info("  [2/6] 向量索引已加载 (FAISS)")
            else:
                logger.info("  [2/6] 向量索引不存在，正在构建...")
                self.store._rebuild_index(kb_id, entities)
                vec_idx.load()
                logger.info("  [2/6] 向量索引构建完成")
        else:
            logger.info("  [2/6] 无向量索引，将使用规则召回")
        return {"entities_index": NameIndex(entities), "vector_index": vec_idx}

    def _generate_candidates(self, state: LinkState) -> dict:
        req = state["request"]
        index: NameIndex = state["entities_index"]
        entities = index.all_entities()
        trigger_terms = self.config.coreference_terms
        logger.info("  [3/6] 候选召回  mentions=%d  kb_entities=%d", len(req.mentions), len(entities))
        result = {}
        for m in req.mentions:
            if m.surface_form in trigger_terms:
                logger.info("    mention=%s → 共指触发词，跳过候选召回", m.mention_id)
                result[m.mention_id] = []
            else:
                cands = candidate_mod.retrieve(
                    m, index, entities, req.options.top_k,
                    context=req.text.content,
                    embedder=self.embedder,
                    vector_index=state.get("vector_index"),
                    vector_top_k=self.config.top_k_retrieve,
                    semantic_min_score=self.config.semantic_min_score,
                )
                result[m.mention_id] = cands
                if cands:
                    logger.info("    mention=%s → %d 个候选, top=%.3f(%s) [%s]",
                               m.mention_id, len(cands), cands[0].score,
                               cands[0].entity.canonical_name, cands[0].match_source)
                else:
                    logger.info("    mention=%s → 0 个候选", m.mention_id)
        return {"candidates_by_id": result}

    def _route_candidates(self, state: LinkState) -> str:
        has = any(state["candidates_by_id"].values())
        if not has:
            logger.info("  [3/6] 所有 mention 均无候选，转入 NIL 回退")
        return "has_candidates" if has else "empty"

    def _nil_fallback(self, state: LinkState) -> dict:
        logger.info("  [✘] NIL 回退：所有 mention 标记为 nil")
        results = [LinkResult(mention_id=m.mention_id, surface_form=m.surface_form, link_status=LinkStatus.nil) for m in state["request"].mentions]
        return {"results": results, "coref_chains": []}

    def _rerank(self, state: LinkState) -> dict:
        req = state["request"]
        context = req.text.content
        logger.info("  [4/6] 重排序...")
        rescored: dict[str, list[candidate_mod.CandidateResult]] = {}
        for m in req.mentions:
            cands = state["candidates_by_id"].get(m.mention_id, [])
            rescored[m.mention_id] = sorted(
                [rescore(c, m, context, self.alias_prior) for c in cands],
                key=lambda c: c.score, reverse=True,
            )[:req.options.top_k]
            if rescored[m.mention_id]:
                t = rescored[m.mention_id][0]
                logger.info("    mention=%s top=%.3f(%s)", m.mention_id, t.score, t.entity.canonical_name)
        return {"candidates_by_id": rescored}

    def _resolve(self, state: LinkState) -> dict:
        req = state["request"]
        logger.info("  [5/6] 决策解析 (nil_threshold=%.2f)...", req.options.nil_threshold)
        results: list[LinkResult] = []
        for m in req.mentions:
            cands = state["candidates_by_id"].get(m.mention_id, [])
            status, top = nil_detector.decide(cands, req.options)
            if status == "nil" or top is None:
                logger.info("    mention=%s → NIL (最高分=%.3f < %.2f)",
                           m.mention_id, cands[0].score if cands else 0, req.options.nil_threshold)
                results.append(LinkResult(mention_id=m.mention_id, surface_form=m.surface_form, link_status=LinkStatus.nil, candidates=self._fmt_candidates(cands, req.options)))
            else:
                logger.info("    mention=%s → %s → %s (置信度=%.3f) [%s]",
                           m.mention_id, status.upper(), top.entity.canonical_name,
                           top.score, top.match_source)
                evid = evidence_mod.build_evidence(top, req.text.content) if req.options.return_evidence else []
                results.append(LinkResult(
                    mention_id=m.mention_id, surface_form=m.surface_form,
                    link_status=LinkStatus(status),
                    entity=EntityRef(entity_id=top.entity.entity_id, canonical_name=top.entity.canonical_name, entity_type=top.entity.entity_type),
                    confidence=top.score,
                    candidates=self._fmt_candidates(cands, req.options),
                    evidence=evid,
                ))
        return {"results": results}

    def _coreference(self, state: LinkState) -> dict:
        req = state["request"]
        if not req.options.enable_coreference:
            logger.info("  [6/6] 共指消解已禁用，跳过")
            return {"coref_chains": []}
        logger.info("  [6/6] 共指消解...")
        updated, chains = coref_mod.resolve(state["results"], req.mentions, self.config.coreference_terms)
        for r in updated:
            if r.coreference:
                logger.info("    mention=%s → 共指解析 → %s (chain=%s, confidence=%.3f)",
                           r.mention_id, r.coreference.resolved_from,
                           r.coreference.chain_id, r.confidence)
            elif r.link_status == LinkStatus.nil and req.mentions:
                # Check if this was a trigger term that couldn't be resolved
                for m in req.mentions:
                    if m.mention_id == r.mention_id and m.surface_form in self.config.coreference_terms:
                        logger.info("    mention=%s → 共指触发词，未找到前驱 → NIL", r.mention_id)
                        break
        if chains:
            logger.info("    共指链: %d 个", len(chains))
        final = []
        for r in updated:
            if r.coreference and req.options.return_evidence:
                r = r.model_copy(update={"evidence": list(r.evidence) + [EvidenceItem(evidence_type=EvidenceType.coreference, detail=f"该 mention 回指前文 mention {r.coreference.resolved_from}")]})
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
            request_id=request.request_id, status="success",
            results=results, coreference_chains=chains,
            summary=LinkSummary(total_mentions=len(results), linked_count=linked, nil_count=nil),
            trace=LinkTrace(
                linker_version=opts.linker_version,
                kb_id=request.knowledge_base.kb_id, kb_version=request.knowledge_base.kb_version,
                options_used={"top_k": opts.top_k, "nil_threshold": opts.nil_threshold, "enable_nil": opts.enable_nil, "enable_coreference": opts.enable_coreference},
            ),
        )
