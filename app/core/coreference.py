from __future__ import annotations
from app.core.candidate import CandidateResult
from app.models.request import MentionInput
from app.models.response import LinkResult, CoreferenceInfo, CoreferenceChain
from app.models.enums import LinkStatus


def resolve(
    results: list[LinkResult],
    mentions: list[MentionInput],
    trigger_terms: set[str],
) -> tuple[list[LinkResult], list[CoreferenceChain]]:
    """Fill coreference mentions and build coreference chains."""
    chains: dict[str, list[str]] = {}  # chain_id -> mention_ids
    chain_entity: dict[str, str] = {}
    chain_counter = 0
    linked_by_mention: dict[str, LinkResult] = {
        r.mention_id: r for r in results if r.link_status == LinkStatus.linked
    }
    # build initial chains from linked mentions
    entity_to_chain: dict[str, str] = {}

    updated = list(results)
    for i, result in enumerate(updated):
        mention = mentions[i]
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
        elif mention.surface_form in trigger_terms:
            # find most recent linked result for coreference
            for prev in reversed(updated[:i]):
                if prev.link_status == LinkStatus.linked and prev.entity:
                    eid = prev.entity.entity_id
                    chain_id = entity_to_chain.get(eid)
                    coref = CoreferenceInfo(resolved_from=prev.mention_id, chain_id=chain_id or "c_unknown")
                    updated[i] = result.model_copy(update={
                        "link_status": LinkStatus.linked,
                        "entity": prev.entity,
                        "confidence": round(max(0.5, prev.confidence - 0.15), 3),
                        "coreference": coref,
                        "evidence": result.evidence,
                    })
                    if chain_id:
                        chains[chain_id].append(result.mention_id)
                    break

    coref_chains = [
        CoreferenceChain(chain_id=cid, mention_ids=mids, entity_id=chain_entity[cid])
        for cid, mids in chains.items()
        if len(mids) > 1
    ]
    return updated, coref_chains
