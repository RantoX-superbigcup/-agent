from __future__ import annotations

from dataclasses import asdict, dataclass

from app.models.entity import Entity
from app.models.request import LinkOptions
from app.storage.index import normalize


@dataclass(frozen=True)
class KBProfile:
    entity_count: int
    alias_density: float
    keyword_density: float
    description_coverage: float
    avg_description_chars: float
    homonym_rate: float
    type_coverage: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScoreWeights:
    alias_weight: float = 0.62
    context_weight: float = 0.23
    type_bonus: float = 0.05
    inferred_type_bonus: float = 0.04
    canonical_bonus: float = 0.03
    expansion_bonus: float = 0.08
    expansion_canonical_bonus: float = 0.08
    prior_weight: float = 0.22
    dirty_expansion_penalty: float = 0.04

    def to_dict(self) -> dict:
        return asdict(self)


def build_kb_profile(entities: list[Entity]) -> KBProfile:
    entity_count = len(entities)
    if not entities:
        return KBProfile(
            entity_count=0,
            alias_density=0.0,
            keyword_density=0.0,
            description_coverage=0.0,
            avg_description_chars=0.0,
            homonym_rate=0.0,
            type_coverage=0.0,
        )

    alias_count = sum(len(entity.aliases) + len(entity.former_names) for entity in entities)
    keyword_count = sum(len(entity.keywords) for entity in entities)
    descriptions = [entity.description.strip() for entity in entities if entity.description.strip()]
    description_chars = sum(len(description) for description in descriptions)
    typed_count = sum(1 for entity in entities if entity.entity_type.value != "OTHER")

    name_buckets: dict[str, set[str]] = {}
    for entity in entities:
        for name in [entity.canonical_name, *entity.aliases, *entity.former_names]:
            normalized = normalize(name)
            if not normalized:
                continue
            name_buckets.setdefault(normalized, set()).add(entity.entity_id)

    ambiguous_names = sum(1 for ids in name_buckets.values() if len(ids) > 1)
    homonym_rate = ambiguous_names / max(1, len(name_buckets))

    return KBProfile(
        entity_count=entity_count,
        alias_density=round(alias_count / entity_count, 3),
        keyword_density=round(keyword_count / entity_count, 3),
        description_coverage=round(len(descriptions) / entity_count, 3),
        avg_description_chars=round(description_chars / max(1, len(descriptions)), 3),
        homonym_rate=round(homonym_rate, 3),
        type_coverage=round(typed_count / entity_count, 3),
    )


def calibrate_options(options: LinkOptions, profile: KBProfile) -> LinkOptions:
    if not options.auto_calibrate:
        return options

    context_quality = _context_quality(profile)
    ambiguity_pressure = _ambiguity_pressure(profile)

    nil_threshold = options.nil_threshold
    nil_threshold += 0.04 * (1.0 - context_quality)
    nil_threshold -= 0.03 * ambiguity_pressure
    nil_threshold = _clamp(nil_threshold, 0.52, 0.72)

    ambiguity_margin = options.ambiguity_margin
    ambiguity_margin += 0.035 * ambiguity_pressure
    ambiguity_margin += 0.015 * (1.0 - context_quality)
    ambiguity_margin = _clamp(ambiguity_margin, 0.05, 0.15)

    top_k = options.top_k
    if profile.entity_count >= 100_000 or ambiguity_pressure >= 0.45:
        top_k = max(top_k, 8)

    return options.model_copy(
        update={
            "top_k": top_k,
            "nil_threshold": round(nil_threshold, 3),
            "ambiguity_margin": round(ambiguity_margin, 3),
        }
    )


def score_weights_for(profile: KBProfile, auto_calibrate: bool = True) -> ScoreWeights:
    if not auto_calibrate:
        return ScoreWeights()

    context_quality = _context_quality(profile)
    ambiguity_pressure = _ambiguity_pressure(profile)

    alias_weight = 0.62
    alias_weight -= 0.045 * (context_quality - 0.5)
    alias_weight -= 0.035 * ambiguity_pressure

    context_weight = 0.23
    context_weight += 0.06 * (context_quality - 0.5)
    context_weight += 0.04 * ambiguity_pressure

    type_bonus = 0.05 + 0.015 * profile.type_coverage

    return ScoreWeights(
        alias_weight=round(_clamp(alias_weight, 0.54, 0.68), 3),
        context_weight=round(_clamp(context_weight, 0.18, 0.32), 3),
        type_bonus=round(_clamp(type_bonus, 0.05, 0.065), 3),
    )


def _context_quality(profile: KBProfile) -> float:
    keyword_signal = _clamp(profile.keyword_density / 4.0, 0.0, 1.0)
    length_signal = _clamp(profile.avg_description_chars / 80.0, 0.0, 1.0)
    return _clamp(
        0.45 * profile.description_coverage
        + 0.35 * keyword_signal
        + 0.20 * length_signal,
        0.0,
        1.0,
    )


def _ambiguity_pressure(profile: KBProfile) -> float:
    alias_signal = _clamp(profile.alias_density / 3.0, 0.0, 1.0)
    homonym_signal = _clamp(profile.homonym_rate * 6.0, 0.0, 1.0)
    size_signal = _clamp(profile.entity_count / 300_000, 0.0, 1.0)
    return _clamp(0.55 * homonym_signal + 0.30 * alias_signal + 0.15 * size_signal, 0.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
