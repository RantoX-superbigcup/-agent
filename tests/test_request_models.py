import pytest
from pydantic import ValidationError

from app.models.request import MentionHint, MentionInput, WorkflowMentionInput


def test_public_mention_input_forbids_internal_fields():
    with pytest.raises(ValidationError):
        MentionInput(
            mention_id="m1",
            surface_form="国网",
            start_offset=0,
            end_offset=2,
            entity_type="ORG",
        )


def test_workflow_mention_input_accepts_internal_type_alias():
    mention = WorkflowMentionInput(
        mention_id="m1",
        surface_form="李安",
        start_offset=0,
        end_offset=2,
        entity_type="人物",
    )

    assert mention.mention_type.value == "PERSON"


def test_mention_hint_normalizes_type_only():
    hint = MentionHint(entity_type="公司")

    assert hint.mention_type.value == "ORG"
