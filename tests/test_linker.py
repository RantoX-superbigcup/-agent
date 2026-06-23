from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from entity_linking_agent.core.contracts import KnowledgeBaseEntity, MentionRecord  # noqa: E402
from entity_linking_agent.core.service import Topic10EntityLinkingService  # noqa: E402


class Topic10EntityLinkingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Topic10EntityLinkingService()

    def test_alias_linking(self) -> None:
        response = self.service.link(
            text="国家电网与南方电网联合发布标准。国网表示将推进配电数字化。",
            mentions=[
                MentionRecord(
                    mention_id="m1",
                    text="国家电网",
                    entity_type="organization",
                    sentence="国家电网与南方电网联合发布标准。",
                ),
                MentionRecord(
                    mention_id="m2",
                    text="国网",
                    entity_type="organization",
                    sentence="国网表示将推进配电数字化。",
                ),
            ],
        )

        predictions = {item["mention_id"]: item for item in response["results"]}
        self.assertEqual(response["workflow_engine"], "langgraph")
        self.assertEqual(response["graph_nodes"][0], "validate_input")
        self.assertIn("resolve_mentions", response["graph_nodes"])
        self.assertEqual(response["node_events"][0]["node"], "validate_input")
        self.assertEqual(predictions["m1"]["linked_entity_id"], "org:sgcc")
        self.assertEqual(predictions["m2"]["linked_entity_id"], "org:sgcc")

    def test_nil_detection(self) -> None:
        response = self.service.link(
            text="未知研究院提交了新的方案。",
            mentions=[
                MentionRecord(
                    mention_id="m_nil",
                    text="未知研究院",
                    entity_type="organization",
                    sentence="未知研究院提交了新的方案。",
                )
            ],
        )

        self.assertEqual(response["results"][0]["status"], "nil")

    def test_invalid_input_stops_before_kb_loading(self) -> None:
        response = self.service.link(text="", mentions=[])

        self.assertEqual(response["route_decision"], "invalid_input")
        self.assertIn("text_required", response["validation_errors"])
        self.assertIn("mentions_required", response["validation_errors"])
        self.assertIn("validate_input", response["graph_nodes"])
        self.assertNotIn("load_kb", response["graph_nodes"])
        self.assertEqual(response["summary"]["total_mentions"], 0)

    def test_candidate_route_empty_goes_to_nil_fallback(self) -> None:
        response = self.service.link(
            text="qzxvbnm 没有对应知识库实体。",
            mentions=[
                MentionRecord(
                    mention_id="m_empty",
                    text="qzxvbnm",
                    entity_type="organization",
                    sentence="qzxvbnm 没有对应知识库实体。",
                )
            ],
            knowledge_base_id="inline-route-test",
            inline_entities=[
                KnowledgeBaseEntity(
                    entity_id="org:sgcc",
                    canonical_name="国家电网",
                    aliases=["国网"],
                    entity_type="organization",
                )
            ],
        )

        result = response["results"][0]
        self.assertEqual(response["route_decision"], "empty_candidates")
        self.assertIn("nil_fallback", response["graph_nodes"])
        self.assertNotIn("rerank_candidates", response["graph_nodes"])
        self.assertEqual(result["status"], "nil")
        self.assertFalse(result["needs_review"])
        self.assertIn("nil_fallback", result["evidence"]["rationale"])

    def test_review_route_handles_ambiguous_candidates(self) -> None:
        response = self.service.link(
            text="苹果发布了新的产品。",
            mentions=[
                MentionRecord(
                    mention_id="m_apple",
                    text="苹果",
                    entity_type="organization",
                    sentence="苹果发布了新的产品。",
                )
            ],
            knowledge_base_id="inline-ambiguous-test",
            inline_entities=[
                KnowledgeBaseEntity(
                    entity_id="org:apple_company",
                    canonical_name="苹果公司",
                    aliases=["苹果"],
                    entity_type="organization",
                ),
                KnowledgeBaseEntity(
                    entity_id="org:apple_lab",
                    canonical_name="苹果实验室",
                    aliases=["苹果"],
                    entity_type="organization",
                ),
            ],
        )

        result = response["results"][0]
        self.assertEqual(response["route_decision"], "needs_review")
        self.assertIn("human_review", response["graph_nodes"])
        self.assertEqual(result["status"], "ambiguous")
        self.assertTrue(result["needs_review"])
        self.assertIn("human_review_required", result["evidence"]["rationale"])

    def test_llm_alias_expansion_links_surface_mention(self) -> None:
        response = self.service.link(
            text="李导演的《断背山》真是令人动人。",
            mentions=[
                MentionRecord(
                    mention_id="m_director",
                    text="李导演",
                    entity_type="Human",
                    sentence="李导演的《断背山》真是令人动人。",
                    metadata={"candidate_aliases": ["李安"]},
                )
            ],
            knowledge_base_id="inline-alias-expansion-test",
            inline_entities=[
                KnowledgeBaseEntity(
                    entity_id="person:ang_lee",
                    canonical_name="李安",
                    aliases=["李安", "Ang Lee"],
                    entity_type="Human",
                    keywords=["导演", "断背山", "电影"],
                )
            ],
        )

        result = response["results"][0]
        self.assertEqual(result["status"], "linked")
        self.assertEqual(result["linked_entity_id"], "person:ang_lee")
        self.assertIn("llm_alias_expansion", result["candidates"][0]["reasons"])

    def test_coreference_fallback(self) -> None:
        response = self.service.link(
            text="国网智能科技正在建设算法平台。该公司强调数据治理能力。",
            mentions=[
                MentionRecord(
                    mention_id="m_org",
                    text="国网智能科技",
                    entity_type="organization",
                    sentence="国网智能科技正在建设算法平台。",
                ),
                MentionRecord(
                    mention_id="m_coref",
                    text="该公司",
                    entity_type="organization",
                    sentence="该公司强调数据治理能力。",
                    metadata={"coreference_hint": True},
                ),
            ],
        )

        predictions = {item["mention_id"]: item for item in response["results"]}
        self.assertEqual(predictions["m_org"]["linked_entity_id"], "org:sgit")
        self.assertEqual(predictions["m_coref"]["linked_entity_id"], "org:sgit")
        self.assertEqual(predictions["m_coref"]["coreference_source_mention_id"], "m_org")


if __name__ == "__main__":
    unittest.main()
