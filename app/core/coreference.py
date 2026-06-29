from __future__ import annotations
from app.models.request import MentionInput
from app.models.response import LinkResult, CoreferenceInfo, CoreferenceChain
from app.models.enums import LinkStatus

_ORG_COREF_TERMS = {"该公司", "该企业", "该机构", "该集团"}
_PERSON_COREF_TERMS = {"他", "她"}
_ORG_SURFACE_HINTS = ("公司", "企业", "集团", "机构", "科技")


def resolve(
    results: list[LinkResult],
    mentions: list[MentionInput],
    trigger_terms: set[str],
) -> tuple[list[LinkResult], list[CoreferenceChain]]:
    """共指消解 —— 处理触发词，并构建共指链。"""
    chains: dict[str, list[str]] = {}
    chain_entity: dict[str, str] = {}
    chain_counter = 0
    entity_to_chain: dict[str, str] = {}

    updated = list(results)

    for i, result in enumerate(updated):
        mention = mentions[i]

        # ── 先检查触发词（优先级高于 link_status） ──
        if mention.surface_form in trigger_terms:
            resolved = False
            previous_items = list(zip(mentions[:i], updated[:i]))
            for prev_mention, prev in reversed(previous_items):
                if not _antecedent_matches(mention.surface_form, prev_mention, prev):
                    continue
                if prev.link_status == LinkStatus.linked and prev.entity:
                    eid = prev.entity.entity_id
                    chain_id = entity_to_chain.get(eid)
                    coref = CoreferenceInfo(
                        resolved_from=prev.mention_id,
                        chain_id=chain_id or "c_unknown",
                    )
                    updated[i] = result.model_copy(update={
                        "link_status": LinkStatus.linked,
                        "entity": prev.entity,
                        "confidence": round(max(0.5, prev.confidence - 0.15), 3),
                        "coreference": coref,
                        "evidence": result.evidence,
                    })
                    if chain_id:
                        chains[chain_id].append(result.mention_id)
                    resolved = True
                    break
                if prev.link_status == LinkStatus.nil:
                    updated[i] = result.model_copy(update={
                        "link_status": LinkStatus.nil,
                        "entity": None,
                        "confidence": 0.0,
                        "coreference": CoreferenceInfo(
                            resolved_from=prev.mention_id,
                            chain_id="nil_coref",
                        ),
                    })
                    resolved = True
                    break
            if not resolved:
                # 找不到前驱 → 标记为 nil
                updated[i] = result.model_copy(update={"link_status": LinkStatus.nil})
            continue

        # ── 正常链接的 mention → 建链 ──
        if result.link_status == LinkStatus.linked and result.entity:
            eid = result.entity.entity_id
            if eid not in entity_to_chain:
                chain_id = f"c{chain_counter}"
                chain_counter += 1
                entity_to_chain[eid] = chain_id
                chains[chain_id] = [result.mention_id]
                chain_entity[chain_id] = eid
            else:
                chains[entity_to_chain[eid]].append(result.mention_id)

    coref_chains = [
        CoreferenceChain(chain_id=cid, mention_ids=mids, entity_id=chain_entity[cid])
        for cid, mids in chains.items()
        if len(mids) > 1
    ]
    return updated, coref_chains


def _antecedent_matches(trigger: str, previous_mention: MentionInput, previous: LinkResult) -> bool:
    if trigger in _ORG_COREF_TERMS:
        if previous.entity and previous.entity.entity_type.value == "ORG":
            return True
        return any(hint in previous_mention.surface_form for hint in _ORG_SURFACE_HINTS)

    if trigger in _PERSON_COREF_TERMS:
        return bool(previous.entity and previous.entity.entity_type.value == "PERSON")

    return previous.link_status == LinkStatus.linked and previous.entity is not None
