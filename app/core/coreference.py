from __future__ import annotations
from dataclasses import dataclass
import re
from app.models.entity import Entity
from app.models.request import WorkflowMentionInput as MentionInput
from app.models.response import LinkResult, CoreferenceInfo, CoreferenceChain
from app.models.enums import LinkStatus

@dataclass(frozen=True)
class CoreferenceRule:
    trigger_terms: set[str]
    entity_types: set[str]
    surface_hints: tuple[str, ...] = ()
    allow_nil_antecedent: bool = False


_ALL_ENTITY_TYPES = {"PERSON", "ORG", "LOC", "OTHER"}
_DEICTIC_PATTERN = re.compile(r"^(?:这|该|此|那)(?:个|位|座|家|部|本|条|口|片|处|所|名|只|间|艘|棵|辆|眼)?(.+)$")
_APPOSITIVE_SEPARATOR_PATTERN = re.compile(r"^[\s,，、:：;；—\-]*$")
_APPOSITIVE_SUFFIXES = ("之人", "之王", "者", "人物", "霸主", "领袖", "冠军", "称号", "角色", "化身")
_PREDICATIVE_TITLE_PREFIXES = ("我们的", "咱们的", "大家的", "真正的", "所谓的")
_HEAD_TYPE_RULES: tuple[tuple[tuple[str, ...], set[str], bool], ...] = (
    (("人", "人物", "导演", "演员", "作者", "先生", "女士", "主席", "总统"), {"PERSON"}, False),
    (("公司", "企业", "集团", "机构", "学校", "大学", "医院", "银行", "政府"), {"ORG"}, True),
    (("城市", "地方", "景区", "湖", "河", "山", "井", "桥", "站", "机场", "港口", "国家", "省", "市", "县", "区", "镇", "村"), {"LOC", "OTHER"}, False),
    (("作品", "电影", "影片", "书", "小说", "歌曲", "理念", "方略", "事件", "项目", "产品", "技术", "系统", "模型"), {"OTHER"}, False),
)


_COREFERENCE_RULES = [
    CoreferenceRule(
        trigger_terms={"他", "她", "这个人", "该人", "此人", "这位", "该人物"},
        entity_types={"PERSON"},
    ),
    CoreferenceRule(
        trigger_terms={"该公司", "该企业", "该机构", "该集团", "这家公司", "这家企业", "这个机构"},
        entity_types={"ORG"},
        surface_hints=("公司", "企业", "集团", "机构", "科技"),
        allow_nil_antecedent=True,
    ),
    CoreferenceRule(
        trigger_terms={"这座湖", "该湖", "这片湖", "此湖"},
        entity_types={"LOC"},
        surface_hints=("湖", "西湖"),
    ),
    CoreferenceRule(
        trigger_terms={"这口井", "该井", "此井", "这眼井"},
        entity_types={"LOC", "OTHER"},
        surface_hints=("井",),
    ),
    CoreferenceRule(
        trigger_terms={"这里", "那里", "此地", "该地", "这个地方", "这座城市", "该城市", "这处景区", "该景区"},
        entity_types={"LOC"},
    ),
    CoreferenceRule(
        trigger_terms={"它", "该作品", "这部作品", "这部电影", "该电影", "这本书", "该书"},
        entity_types={"OTHER"},
    ),
]


def is_trigger_surface(surface_form: str, trigger_terms: set[str]) -> bool:
    return surface_form in trigger_terms or _build_cue(surface_form) is not None


def should_skip_candidate_retrieval(
    mention: MentionInput,
    previous_mentions: list[MentionInput],
    trigger_terms: set[str],
) -> bool:
    if mention.surface_form in trigger_terms or _find_exact_rule(mention.surface_form):
        return True
    cue = _build_cue(mention.surface_form)
    if cue is None:
        return False
    return any(_surface_matches_hints(previous.surface_form, cue.surface_hints) for previous in previous_mentions)


def resolve(
    results: list[LinkResult],
    mentions: list[MentionInput],
    trigger_terms: set[str],
    entities_by_id: dict[str, Entity] | None = None,
    context: str = "",
) -> tuple[list[LinkResult], list[CoreferenceChain]]:
    """共指消解 —— 处理触发词，并构建共指链。"""
    chains: dict[str, list[str]] = {}
    chain_entity: dict[str, str] = {}
    chain_counter = 0
    entity_to_chain: dict[str, str] = {}

    updated = list(results)

    for i, result in enumerate(updated):
        mention = mentions[i]

        appositive_antecedent = _find_appositive_antecedent(i, mention, mentions, updated, context, trigger_terms)
        if appositive_antecedent:
            prev_mention, prev = appositive_antecedent
            _copy_linked_coreference(
                updated,
                i,
                result,
                prev_mention,
                prev,
                entity_to_chain,
                chains,
            )
            continue

        # ── 先检查触发词（优先级高于 link_status） ──
        if is_trigger_surface(mention.surface_form, trigger_terms):
            if result.link_status == LinkStatus.linked and result.entity:
                if _ensure_chain(result, entity_to_chain, chains, chain_entity, chain_counter):
                    chain_counter += 1
                continue
            resolved = False
            previous_items = list(zip(mentions[:i], updated[:i]))
            for prev_mention, prev in reversed(previous_items):
                if not _antecedent_matches(mention.surface_form, prev_mention, prev, entities_by_id or {}):
                    continue
                if prev.link_status == LinkStatus.linked and prev.entity:
                    _copy_linked_coreference(
                        updated,
                        i,
                        result,
                        prev_mention,
                        prev,
                        entity_to_chain,
                        chains,
                    )
                    resolved = True
                    break
                rule = _build_cue(mention.surface_form)
                if prev.link_status == LinkStatus.nil and (rule is None or rule.allow_nil_antecedent):
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
            if _ensure_chain(result, entity_to_chain, chains, chain_entity, chain_counter):
                chain_counter += 1

    chain_counter = _resolve_cataphora(
        updated,
        mentions,
        trigger_terms,
        entity_to_chain,
        chains,
        chain_entity,
        chain_counter,
        context,
    )

    mention_order = {mention.mention_id: order for order, mention in enumerate(mentions)}
    for mention_ids in chains.values():
        mention_ids.sort(key=lambda mention_id: mention_order.get(mention_id, len(mention_order)))

    coref_chains = [
        CoreferenceChain(chain_id=cid, mention_ids=mids, entity_id=chain_entity[cid])
        for cid, mids in chains.items()
        if len(mids) > 1
    ]
    return updated, coref_chains


def _resolve_cataphora(
    updated: list[LinkResult],
    mentions: list[MentionInput],
    trigger_terms: set[str],
    entity_to_chain: dict[str, str],
    chains: dict[str, list[str]],
    chain_entity: dict[str, str],
    chain_counter: int,
    context: str,
) -> int:
    for i, result in enumerate(updated):
        mention = mentions[i]
        if _has_previous_appositive_antecedent(i, mention, mentions, updated, context, trigger_terms):
            continue
        if result.link_status == LinkStatus.linked and result.entity and not _is_forward_rewritable_title(mention.surface_form, result):
            continue
        if not _can_resolve_forward(mention.surface_form, trigger_terms):
            continue
        target = _find_forward_target(mention.surface_form, mentions[i + 1:], updated[i + 1:])
        if target:
            next_mention, nxt = target
            if _ensure_chain(nxt, entity_to_chain, chains, chain_entity, chain_counter):
                chain_counter += 1
            _copy_linked_coreference(
                updated,
                i,
                result,
                next_mention,
                nxt,
                entity_to_chain,
                chains,
            )
    return chain_counter


def _find_forward_target(
    surface_form: str,
    next_mentions: list[MentionInput],
    next_results: list[LinkResult],
) -> tuple[MentionInput, LinkResult] | None:
    fallback: tuple[MentionInput, LinkResult] | None = None
    for next_mention, nxt in zip(next_mentions, next_results):
        if nxt.link_status != LinkStatus.linked or not nxt.entity:
            continue
        if not _forward_target_matches(surface_form, nxt):
            continue
        if fallback is None:
            fallback = (next_mention, nxt)
        if not any(e.detail.startswith("描述性指称召回") for e in nxt.evidence):
            return next_mention, nxt
    return fallback


def _can_resolve_forward(surface_form: str, trigger_terms: set[str]) -> bool:
    if surface_form in trigger_terms or _build_cue(surface_form) is not None:
        return True
    return _looks_like_predicative_title(surface_form)


def _is_forward_rewritable_title(surface_form: str, result: LinkResult) -> bool:
    return (
        _looks_like_predicative_title(surface_form)
        and result.entity is not None
        and any(e.detail.startswith("描述性指称召回") for e in result.evidence)
    )


def _forward_target_matches(surface_form: str, target: LinkResult) -> bool:
    if not target.entity:
        return False
    cue = _build_cue(surface_form)
    if cue:
        return target.entity.entity_type.value in cue.entity_types
    if _looks_like_predicative_title(surface_form):
        return target.entity.entity_type.value in {"PERSON", "OTHER"}
    return False


def _ensure_chain(
    result: LinkResult,
    entity_to_chain: dict[str, str],
    chains: dict[str, list[str]],
    chain_entity: dict[str, str],
    chain_counter: int,
) -> bool:
    if not result.entity:
        return False
    eid = result.entity.entity_id
    if eid not in entity_to_chain:
        chain_id = f"c{chain_counter}"
        entity_to_chain[eid] = chain_id
        chains[chain_id] = [result.mention_id]
        chain_entity[chain_id] = eid
        return True
    chain_id = entity_to_chain[eid]
    if result.mention_id not in chains[chain_id]:
        chains[chain_id].append(result.mention_id)
    return False


def _copy_linked_coreference(
    updated: list[LinkResult],
    index: int,
    result: LinkResult,
    antecedent_mention: MentionInput,
    antecedent: LinkResult,
    entity_to_chain: dict[str, str],
    chains: dict[str, list[str]],
) -> None:
    if not antecedent.entity:
        return
    eid = antecedent.entity.entity_id
    chain_id = entity_to_chain.get(eid)
    coref = CoreferenceInfo(
        resolved_from=antecedent_mention.mention_id,
        chain_id=chain_id or "c_unknown",
    )
    updated[index] = result.model_copy(update={
        "link_status": LinkStatus.linked,
        "entity": antecedent.entity,
        "confidence": round(max(0.5, antecedent.confidence - 0.15), 3),
        "coreference": coref,
        "evidence": result.evidence,
    })
    if chain_id and result.mention_id not in chains[chain_id]:
        chains[chain_id].append(result.mention_id)


def _find_appositive_antecedent(
    current_index: int,
    mention: MentionInput,
    mentions: list[MentionInput],
    updated: list[LinkResult],
    context: str,
    trigger_terms: set[str],
) -> tuple[MentionInput, LinkResult] | None:
    if current_index <= 0 or not context:
        return None
    if is_trigger_surface(mention.surface_form, trigger_terms):
        return None
    if not _looks_like_appositive_surface(mention.surface_form):
        return None

    prev_mention = mentions[current_index - 1]
    if is_trigger_surface(prev_mention.surface_form, trigger_terms):
        return None
    prev = updated[current_index - 1]
    if prev.link_status != LinkStatus.linked or not prev.entity:
        return None
    if not _has_appositive_boundary(context, prev_mention, mention):
        return None
    return prev_mention, prev


def _has_previous_appositive_antecedent(
    current_index: int,
    mention: MentionInput,
    mentions: list[MentionInput],
    updated: list[LinkResult],
    context: str,
    trigger_terms: set[str],
) -> bool:
    return _find_appositive_antecedent(current_index, mention, mentions, updated, context, trigger_terms) is not None


def _has_appositive_boundary(context: str, previous_mention: MentionInput, mention: MentionInput) -> bool:
    if (
        previous_mention.end_offset < 0
        or mention.start_offset < previous_mention.end_offset
        or mention.start_offset > len(context)
    ):
        return False
    between = context[previous_mention.end_offset:mention.start_offset]
    return bool(between) and _APPOSITIVE_SEPARATOR_PATTERN.match(between) is not None


def _looks_like_appositive_surface(surface_form: str) -> bool:
    text = surface_form.strip()
    if not 2 <= len(text) <= 16:
        return False
    return "的" in text or "之" in text or text.endswith(_APPOSITIVE_SUFFIXES)


def _looks_like_predicative_title(surface_form: str) -> bool:
    text = surface_form.strip()
    if not 2 <= len(text) <= 16:
        return False
    return text.startswith(_PREDICATIVE_TITLE_PREFIXES) or text.endswith(_APPOSITIVE_SUFFIXES)


def _antecedent_matches(
    trigger: str,
    previous_mention: MentionInput,
    previous: LinkResult,
    entities_by_id: dict[str, Entity],
) -> bool:
    rule = _build_cue(trigger)
    if rule is None:
        return previous.link_status == LinkStatus.linked and previous.entity is not None

    if previous.entity:
        type_matches = previous.entity.entity_type.value in rule.entity_types
        if not type_matches:
            return False
        if not rule.surface_hints:
            return True
        full_entity = entities_by_id.get(previous.entity.entity_id)
        return any(
            hint in previous_mention.surface_form or hint in previous.entity.canonical_name
            or (full_entity is not None and hint in _entity_text(full_entity))
            for hint in rule.surface_hints
        )

    if previous.link_status == LinkStatus.nil and rule.allow_nil_antecedent:
        return _surface_matches_hints(previous_mention.surface_form, rule.surface_hints)

    return False


def _build_cue(surface_form: str) -> CoreferenceRule | None:
    exact_rule = _find_exact_rule(surface_form)
    if exact_rule:
        return exact_rule

    head = _extract_deictic_head(surface_form)
    if not head:
        return None

    for keywords, entity_types, allow_nil in _HEAD_TYPE_RULES:
        if any(keyword in head for keyword in keywords):
            return CoreferenceRule(
                trigger_terms={surface_form},
                entity_types=entity_types,
                surface_hints=(head,),
                allow_nil_antecedent=allow_nil,
            )

    return CoreferenceRule(
        trigger_terms={surface_form},
        entity_types=_ALL_ENTITY_TYPES,
        surface_hints=(head,),
    )


def _find_exact_rule(surface_form: str) -> CoreferenceRule | None:
    for rule in _COREFERENCE_RULES:
        if surface_form in rule.trigger_terms:
            return rule
    return None


def _extract_deictic_head(surface_form: str) -> str:
    matched = _DEICTIC_PATTERN.match(surface_form.strip())
    if not matched:
        return ""
    return matched.group(1).strip()


def _surface_matches_hints(surface_form: str, hints: tuple[str, ...]) -> bool:
    return bool(hints) and any(hint in surface_form for hint in hints)


def _entity_text(entity: Entity) -> str:
    return " ".join(
        [
            entity.canonical_name,
            *entity.aliases,
            *entity.former_names,
            entity.description,
            *entity.keywords,
        ]
    )
