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
from app.core.kb_profile import (
    KBProfile,
    ScoreWeights,
    build_kb_profile,
    calibrate_options,
    score_weights_for,
)
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
    effective_options: LinkOptions
    kb_profile: KBProfile
    score_weights: ScoreWeights
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
        builder.add_node("human_review", RunnableLambda(self._human_review, name="human_review"))
        builder.add_edge(START, "validate")
        builder.add_conditional_edges("validate", self._route_validate, {"ok": "load_kb", "error": END})
        builder.add_edge("load_kb", "generate_candidates")
        builder.add_conditional_edges("generate_candidates", self._route_candidates, {"has_candidates": "rerank", "empty": "nil_fallback"})
        builder.add_edge("nil_fallback", END)
        builder.add_edge("rerank", "resolve")
        builder.add_edge("resolve", "coreference")
        builder.add_conditional_edges("coreference", self._review_route, {"auto_accept": END, "needs_review": "human_review"})
        builder.add_edge("human_review", END)
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
        profile = build_kb_profile(entities)
        effective_options = calibrate_options(state["request"].options, profile)
        score_weights = score_weights_for(profile, effective_options.auto_calibrate)
        logger.info(
            "  [2/6] KB画像: aliases/entity=%.3f, keywords/entity=%.3f, desc_coverage=%.3f, homonym_rate=%.3f",
            profile.alias_density,
            profile.keyword_density,
            profile.description_coverage,
            profile.homonym_rate,
        )
        logger.info(
            "  [2/6] 自适应参数: top_k=%d, nil_threshold=%.3f, ambiguity_margin=%.3f, alias_w=%.3f, context_w=%.3f",
            effective_options.top_k,
            effective_options.nil_threshold,
            effective_options.ambiguity_margin,
            score_weights.alias_weight,
            score_weights.context_weight,
        )
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
        return {
            "entities_index": NameIndex(entities),
            "vector_index": vec_idx,
            "effective_options": effective_options,
            "kb_profile": profile,
            "score_weights": score_weights,
        }

    def _generate_candidates(self, state: LinkState) -> dict:
        req = state["request"]
        options = state.get("effective_options", req.options)
        index: NameIndex = state["entities_index"]
        entities = index.all_entities()
        trigger_terms = self.config.coreference_terms
        logger.info("  [3/6] 候选召回  mentions=%d  kb_entities=%d", len(req.mentions), len(entities))
        result = {}
        previous_mentions: list[MentionInput] = []
        for m in req.mentions:
            if coref_mod.should_skip_candidate_retrieval(m, previous_mentions, trigger_terms):
                logger.info("    mention=%s → 共指触发词，跳过候选召回", m.mention_id)
                result[m.mention_id] = []
            else:
                cands = candidate_mod.retrieve(
                    m, index, entities, options.top_k,
                    context=req.text.content,
                    embedder=self.embedder,
                    vector_index=state.get("vector_index"),
                )
                result[m.mention_id] = cands
                if cands:
                    logger.info("    mention=%s → %d 个候选, top=%.3f(%s) [%s]",
                               m.mention_id, len(cands), cands[0].score,
                               cands[0].entity.canonical_name, cands[0].match_source)
                else:
                    logger.info("    mention=%s → 0 个候选", m.mention_id)
            previous_mentions.append(m)
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
        options = state.get("effective_options", req.options)
        weights = state.get("score_weights", ScoreWeights())
        context = req.text.content
        logger.info("  [4/6] 重排序...")
        rescored: dict[str, list[candidate_mod.CandidateResult]] = {}
        for m in req.mentions:
            cands = state["candidates_by_id"].get(m.mention_id, [])
            rescored[m.mention_id] = sorted(
                [rescore(c, m, context, self.alias_prior, weights) for c in cands],
                key=candidate_mod.rank_key, reverse=True,
            )[:options.top_k]
            if rescored[m.mention_id]:
                t = rescored[m.mention_id][0]
                logger.info("    mention=%s top=%.3f(%s)", m.mention_id, t.score, t.entity.canonical_name)
        return {"candidates_by_id": rescored}

    def _resolve(self, state: LinkState) -> dict:
        req = state["request"]
        options = state.get("effective_options", req.options)
        logger.info("  [5/6] 决策解析 (nil_threshold=%.3f)...", options.nil_threshold)
        results: list[LinkResult] = []
        for m in req.mentions:
            cands = state["candidates_by_id"].get(m.mention_id, [])
            status, top = nil_detector.decide(cands, options)
            if status == "nil" or top is None:
                logger.info("    mention=%s → NIL (最高分=%.3f < %.2f)",
                           m.mention_id, cands[0].score if cands else 0, options.nil_threshold)
                results.append(LinkResult(mention_id=m.mention_id, surface_form=m.surface_form, link_status=LinkStatus.nil, candidates=self._fmt_candidates(cands, options)))
            else:
                logger.info("    mention=%s → %s → %s (置信度=%.3f) [%s]",
                           m.mention_id, status.upper(), top.entity.canonical_name,
                           top.score, top.match_source)
                evid = evidence_mod.build_evidence(top, req.text.content) if options.return_evidence else []
                results.append(LinkResult(
                    mention_id=m.mention_id, surface_form=m.surface_form,
                    link_status=LinkStatus(status),
                    entity=EntityRef(entity_id=top.entity.entity_id, canonical_name=top.entity.canonical_name, entity_type=top.entity.entity_type),
                    confidence=top.score,
                    candidates=self._fmt_candidates(cands, options),
                    evidence=evid,
                ))
        return {"results": results}

    def _coreference(self, state: LinkState) -> dict:
        req = state["request"]
        options = state.get("effective_options", req.options)
        if not options.enable_coreference:
            logger.info("  [6/6] 共指消解已禁用，跳过")
            return {"coref_chains": []}
        logger.info("  [6/6] 共指消解...")
        index: NameIndex | None = state.get("entities_index")
        entities_by_id = {entity.entity_id: entity for entity in index.all_entities()} if index else {}
        updated, chains = coref_mod.resolve(
            state["results"],
            req.mentions,
            self.config.coreference_terms,
            entities_by_id=entities_by_id,
            context=req.text.content,
        )
        for r in updated:
            if r.coreference:
                logger.info("    mention=%s → 共指解析 → %s (chain=%s, confidence=%.3f)",
                           r.mention_id, r.coreference.resolved_from,
                           r.coreference.chain_id, r.confidence)
            elif r.link_status == LinkStatus.nil and req.mentions:
                # Check if this was a trigger term that couldn't be resolved
                for m in req.mentions:
                    if m.mention_id == r.mention_id and coref_mod.is_trigger_surface(m.surface_form, self.config.coreference_terms):
                        logger.info("    mention=%s → 共指触发词，未找到前驱 → NIL", r.mention_id)
                        break
        if chains:
            logger.info("    共指链: %d 个", len(chains))
        final = []
        for r in updated:
            if r.coreference and options.return_evidence:
                r = r.model_copy(update={"evidence": list(r.evidence) + [EvidenceItem(evidence_type=EvidenceType.coreference, detail=f"该 mention 与 mention {r.coreference.resolved_from} 共指")]})
            final.append(r)
        return {"results": final, "coref_chains": chains}

    def _review_route(self, state: LinkState) -> str:
        req = state["request"]
        options = state.get("effective_options", req.options)
        needs_review = any(self._needs_review(result, options) for result in state.get("results", []))
        if needs_review:
            logger.info("  [review] 检测到低置信或歧义结果，进入人工复核分支")
        return "needs_review" if needs_review else "auto_accept"

    def _human_review(self, state: LinkState) -> dict:
        logger.info("  [review] 标记需要人工复核的 mention...")
        req = state["request"]
        options = state.get("effective_options", req.options)
        reviewed: list[LinkResult] = []
        for result in state.get("results", []):
            if self._needs_review(result, options):
                evidence = list(result.evidence)
                evidence.append(
                    EvidenceItem(
                        evidence_type=EvidenceType.model_inference,
                        detail="human_review_required: low_confidence_or_ambiguous",
                    )
                )
                reviewed.append(result.model_copy(update={"evidence": evidence}))
            else:
                reviewed.append(result)
        return {"results": reviewed}

    @staticmethod
    def _needs_review(result: LinkResult, options: LinkOptions) -> bool:
        if result.link_status == LinkStatus.ambiguous:
            return True
        if result.link_status == LinkStatus.nil and result.candidates:
            return True
        if result.link_status == LinkStatus.nil and result.coreference:
            return True
        if result.entity and result.confidence < options.nil_threshold:
            return True
        return False

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
        opts = state.get("effective_options", request.options)
        profile = state.get("kb_profile")
        weights = state.get("score_weights", ScoreWeights())
        ambiguous = sum(1 for r in results if r.link_status == LinkStatus.ambiguous)
        review = sum(1 for r in results if self._needs_review(r, opts))
        return LinkResponse(
            request_id=request.request_id, status="success",
            results=results, coreference_chains=chains,
            summary=LinkSummary(
                total_mentions=len(results),
                linked_count=linked,
                nil_count=nil,
                ambiguous_count=ambiguous,
                review_count=review,
            ),
            trace=LinkTrace(
                linker_version=opts.linker_version,
                kb_id=request.knowledge_base.kb_id, kb_version=request.knowledge_base.kb_version,
                options_used={
                    "top_k": opts.top_k,
                    "nil_threshold": opts.nil_threshold,
                    "ambiguity_margin": opts.ambiguity_margin,
                    "auto_calibrate": opts.auto_calibrate,
                    "enable_nil": opts.enable_nil,
                    "enable_coreference": opts.enable_coreference,
                    "kb_profile": profile.to_dict() if profile else None,
                    "scoring_weights": weights.to_dict(),
                },
            ),
        )
