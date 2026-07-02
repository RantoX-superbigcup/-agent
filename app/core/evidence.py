from __future__ import annotations
from app.core.candidate import CandidateResult
from app.models.response import EvidenceItem
from app.models.enums import EvidenceType
from app.storage.index import normalize


def build_evidence(candidate: CandidateResult, context: str) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    source = candidate.match_source
    if source == "canonical_match":
        items.append(EvidenceItem(
            evidence_type=EvidenceType.canonical_match,
            detail=f"surface_form 命中标准名：{candidate.matched_name}",
        ))
    elif source == "alias_match":
        items.append(EvidenceItem(
            evidence_type=EvidenceType.alias_match,
            detail=f"surface_form 命中别名：{candidate.matched_name}",
        ))
    elif source == "former_name_match":
        items.append(EvidenceItem(
            evidence_type=EvidenceType.former_name_match,
            detail=f"surface_form 命中曾用名：{candidate.matched_name}",
        ))
    else:
        items.append(EvidenceItem(
            evidence_type=EvidenceType.similarity_match,
            detail=(
                f"描述性指称召回：{candidate.matched_name}，score={candidate.score}"
                if source == "descriptive_match"
                else f"字符相似度命中：{candidate.matched_name}，score={candidate.score}"
            ),
        ))
    entity = candidate.entity
    kw_hits = [kw for kw in entity.keywords if kw and kw in context]
    if kw_hits:
        items.append(EvidenceItem(
            evidence_type=EvidenceType.context_match,
            detail=f"上下文包含关键词：{'、'.join(kw_hits[:5])}",
        ))
    for reason in candidate.reasons:
        if reason in {"llm_alias_expansion", "alias_prior_support"}:
            items.append(EvidenceItem(
                evidence_type=EvidenceType.model_inference,
                detail=f"模型/先验加权依据：{reason}",
            ))
        elif reason == "llm_rerank_support":
            items.append(EvidenceItem(
                evidence_type=EvidenceType.model_inference,
                detail="大模型复核在候选列表中选择该实体",
            ))
        elif reason == "llm_rerank_reason":
            items.append(EvidenceItem(
                evidence_type=EvidenceType.model_inference,
                detail="大模型复核理由已用于重排",
            ))
        elif reason == "contextual_alias_expansion":
            items.append(EvidenceItem(
                evidence_type=EvidenceType.model_inference,
                detail="上下文规则触发别名扩展",
            ))
        elif reason == "expansion_context_validated":
            items.append(EvidenceItem(
                evidence_type=EvidenceType.model_inference,
                detail="别名扩展已通过上下文验证",
            ))
        elif reason == "duplicate_canonical_auto_accept":
            items.append(EvidenceItem(
                evidence_type=EvidenceType.model_inference,
                detail="近分候选均为同名重复实体，已按知识库去噪策略自动接受",
            ))
        elif reason == "descriptive_reference":
            items.append(EvidenceItem(
                evidence_type=EvidenceType.model_inference,
                detail="通过 mention 周边描述与实体描述/关键词匹配召回",
            ))
        elif reason == "descriptive_reference_auto_accept":
            items.append(EvidenceItem(
                evidence_type=EvidenceType.model_inference,
                detail="描述性指称候选证据充分，已自动接受",
            ))
        elif reason == "entity_type_aligned":
            items.append(EvidenceItem(
                evidence_type=EvidenceType.type_match,
                detail="mention 类型与候选实体类型一致",
            ))
    return items
