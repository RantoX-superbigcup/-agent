from enum import Enum


class LinkStatus(str, Enum):
    linked = "linked"
    nil = "nil"
    ambiguous = "ambiguous"
    error = "error"


class EvidenceType(str, Enum):
    canonical_match = "canonical_match"
    alias_match = "alias_match"
    short_name_match = "short_name_match"
    context_match = "context_match"
    type_match = "type_match"
    semantic_match = "semantic_match"
    coreference = "coreference"
    model_inference = "model_inference"


class EntityType(str, Enum):
    ORG = "ORG"
    PERSON = "PERSON"
    LOC = "LOC"
    OTHER = "OTHER"
