"""Baseline entity linking and knowledge alignment logic."""

from __future__ import annotations

from typing import Optional

from entity_linking_agent.core.alias_prior import AliasPrior
from entity_linking_agent.core.contracts import (
    CandidateScore,
    EvidenceRecord,
    KnowledgeBaseEntity,
    LinkDecision,
    LinkOptions,
    MentionRecord,
)
from entity_linking_agent.core.retriever import CandidateRetriever
from entity_linking_agent.utils.text import clamp, extract_context, keyword_hits, normalize_text

_COREFERENCE_TERMS = {"该公司", "该企业", "该机构", "该集团", "其", "他", "她", "它"}


class EntityLinker:
    """Service baseline for mention-to-entity linking."""

    def __init__(
        self,
        retriever: Optional[CandidateRetriever] = None,
        alias_prior: Optional[AliasPrior] = None,
    ) -> None:
        self.retriever = retriever or CandidateRetriever()
        self.alias_prior = alias_prior or AliasPrior({})

    def link_document(
        self,
        text: str,
        mentions: list[MentionRecord],
        entities: list[KnowledgeBaseEntity],
        options: LinkOptions,
    ) -> list[LinkDecision]:
        entity_index = {entity.entity_id: entity for entity in entities}
        history: list[LinkDecision] = []
        results: list[LinkDecision] = []

        for mention in mentions:
            context = extract_context(
                text=text,
                start=mention.start,
                end=mention.end,
                fallback=mention.sentence or text,
            )

            candidates = self.retriever.retrieve(
                mention=mention,
                entities=entities,
                top_k=max(options.top_k_candidates, 1),
            )
            trimmed_candidates = self.rescore_candidates(
                mention=mention,
                context=context,
                candidates=candidates,
                entity_index=entity_index,
                top_k=options.top_k_candidates,
            )

            decision = self.decide_mention(
                mention=mention,
                context=context,
                candidates=trimmed_candidates,
                options=options,
                history=history,
            )
            results.append(decision)

            if decision.linked_entity_id:
                history.append(decision)

        return results

    def rescore_candidates(
        self,
        mention: MentionRecord,
        context: str,
        candidates: list[CandidateScore],
        entity_index: dict[str, KnowledgeBaseEntity],
        top_k: int,
    ) -> list[CandidateScore]:
        rescored = [
            self._rescore_candidate(
                mention=mention,
                context=context,
                candidate=candidate,
                entity=entity_index[candidate.entity_id],
            )
            for candidate in candidates
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]

    def decide_mention(
        self,
        mention: MentionRecord,
        context: str,
        candidates: list[CandidateScore],
        options: LinkOptions,
        history: list[LinkDecision],
    ) -> LinkDecision:
        return self._decide(
            mention=mention,
            context=context,
            candidates=candidates,
            options=options,
            history=history,
        )

    def _rescore_candidate(
        self,
        mention: MentionRecord,
        context: str,
        candidate: CandidateScore,
        entity: KnowledgeBaseEntity,
    ) -> CandidateScore:
        overlaps = keyword_hits(context, entity.keywords)
        context_score = min(1.0, len(overlaps) / max(1, min(len(entity.keywords), 4)))
        description_score = self._description_overlap_score(context, entity)
        context_score = max(context_score, description_score)
        type_bonus = 0.05 if mention.entity_type and mention.entity_type == entity.entity_type else 0.0
        canonical_bonus = 0.03 if normalize_text(mention.text) == normalize_text(entity.canonical_name) else 0.0
        expansion_bonus = 0.08 if "llm_alias_expansion" in candidate.reasons else 0.0
        prior_score = self.alias_prior.score(mention.text, entity.entity_id)
        prior_bonus = 0.22 * prior_score
        final_score = clamp(
            0.62 * candidate.alias_similarity
            + 0.23 * context_score
            + type_bonus
            + canonical_bonus
            + expansion_bonus
            + prior_bonus
        )

        reasons = list(candidate.reasons)
        if overlaps:
            reasons.append("context_keyword_support")
        if description_score > 0:
            reasons.append("description_overlap_support")
        if prior_score > 0:
            reasons.append("alias_prior_support")

        return CandidateScore(
            entity_id=candidate.entity_id,
            canonical_name=candidate.canonical_name,
            entity_type=candidate.entity_type,
            score=round(final_score, 3),
            alias_similarity=candidate.alias_similarity,
            matched_alias=candidate.matched_alias,
            overlapping_keywords=overlaps,
            reasons=reasons,
        )

    @staticmethod
    def _description_overlap_score(context: str, entity: KnowledgeBaseEntity) -> float:
        description = str(entity.metadata.get("description", ""))
        normalized_context = normalize_text(context)
        normalized_description = normalize_text(description)
        if not normalized_context or not normalized_description:
            return 0.0

        context_chars = {char for char in normalized_context if "\u4e00" <= char <= "\u9fff"}
        if not context_chars:
            return 0.0
        matched = sum(1 for char in context_chars if char in normalized_description)
        return min(1.0, matched / max(4, len(context_chars)))

    def _decide(
        self,
        mention: MentionRecord,
        context: str,
        candidates: list[CandidateScore],
        options: LinkOptions,
        history: list[LinkDecision],
    ) -> LinkDecision:
        coreference = self._try_coreference(mention=mention, context=context, history=history)

        if not candidates:
            if coreference is not None:
                return coreference
            return self._nil_decision(mention=mention, context=context, candidates=[])

        top_candidate = candidates[0]
        second_candidate = candidates[1] if len(candidates) > 1 else None

        if top_candidate.score < options.nil_threshold:
            if coreference is not None:
                return coreference
            return self._nil_decision(mention=mention, context=context, candidates=candidates)

        status = "linked"
        needs_review = False
        rationale = list(top_candidate.reasons)

        if second_candidate and (top_candidate.score - second_candidate.score) < options.ambiguity_margin:
            status = "ambiguous"
            needs_review = True
            rationale.append("margin_below_ambiguity_threshold")
        else:
            rationale.append("score_above_link_threshold")

        evidence = EvidenceRecord(
            normalized_mention=normalize_text(mention.text),
            matched_alias=top_candidate.matched_alias,
            context_snippet=context,
            overlapping_keywords=top_candidate.overlapping_keywords,
            rationale=rationale,
        )

        return LinkDecision(
            mention_id=mention.mention_id,
            text=mention.text,
            entity_type=mention.entity_type,
            linked_entity_id=top_candidate.entity_id,
            canonical_name=top_candidate.canonical_name,
            status=status,
            confidence=top_candidate.score,
            needs_review=needs_review,
            candidates=candidates,
            evidence=evidence,
        )

    def _nil_decision(
        self,
        mention: MentionRecord,
        context: str,
        candidates: list[CandidateScore],
    ) -> LinkDecision:
        rationale = ["nil_detected_due_to_low_score"]
        if not candidates:
            rationale.append("no_candidate_retrieved")

        evidence = EvidenceRecord(
            normalized_mention=normalize_text(mention.text),
            matched_alias=candidates[0].matched_alias if candidates else None,
            context_snippet=context,
            overlapping_keywords=candidates[0].overlapping_keywords if candidates else [],
            rationale=rationale,
        )

        return LinkDecision(
            mention_id=mention.mention_id,
            text=mention.text,
            entity_type=mention.entity_type,
            linked_entity_id=None,
            canonical_name=None,
            status="nil",
            confidence=candidates[0].score if candidates else 0.0,
            needs_review=bool(candidates),
            candidates=candidates,
            evidence=evidence,
        )

    def _try_coreference(
        self,
        mention: MentionRecord,
        context: str,
        history: list[LinkDecision],
    ) -> Optional[LinkDecision]:
        coreference_hint = bool(mention.metadata.get("coreference_hint"))
        if mention.text not in _COREFERENCE_TERMS and not coreference_hint:
            return None

        for previous in reversed(history):
            if mention.entity_type and previous.entity_type and mention.entity_type != previous.entity_type:
                continue

            evidence = EvidenceRecord(
                normalized_mention=normalize_text(mention.text),
                matched_alias=None,
                context_snippet=context,
                rationale=[
                    "coreference_fallback",
                    f"inherits_recent_link:{previous.mention_id}",
                ],
            )
            confidence = round(max(0.50, previous.confidence - 0.15), 3)
            return LinkDecision(
                mention_id=mention.mention_id,
                text=mention.text,
                entity_type=mention.entity_type,
                linked_entity_id=previous.linked_entity_id,
                canonical_name=previous.canonical_name,
                status="linked",
                confidence=confidence,
                needs_review=True,
                candidates=[],
                evidence=evidence,
                coreference_source_mention_id=previous.mention_id,
            )

        return None
