from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from entity_linking_agent.cli import ConversationalAgent, parse_user_payload  # noqa: E402


class FakeDeepSeekClient:
    is_configured = True

    def analyze_turn(self, user_text: str, current_state: dict) -> dict:
        return {
            "action": "update",
            "kb_id": "ccks2019-v1",
            "text": "南京南站:坐高铁在南京南站下。南京南站",
            "mentions": ["南京南站", "高铁"],
            "mention_aliases": {},
            "run_requested": False,
            "reply": None,
            "confidence": 0.95,
        }


class FakeMovieDeepSeekClient:
    is_configured = True

    def analyze_turn(self, user_text: str, current_state: dict) -> dict:
        return {
            "action": "update",
            "kb_id": None,
            "text": "李导演的《断背山》真是令人动人",
            "mentions": ["李导演", "断背山"],
            "mention_aliases": {"李导演": ["李安"]},
            "run_requested": False,
            "reply": None,
            "confidence": 0.95,
        }


class ConversationalCliTests(unittest.TestCase):
    def test_parse_text_and_mentions_in_one_turn(self) -> None:
        text, mentions = parse_user_payload("链接：南京南站:坐高铁在南京南站下。南京南站，实体是 南京南站,高铁")

        self.assertEqual(text, "南京南站:坐高铁在南京南站下。南京南站")
        self.assertEqual(mentions, ["南京南站", "高铁"])

    def test_dialogue_keeps_kb_context(self) -> None:
        agent = ConversationalAgent(use_llm=False)

        reply = agent.handle_turn("换成CCKS知识库")

        self.assertIn("ccks2019-v1", reply)
        self.assertEqual(agent.state.kb_id, "ccks2019-v1")
        self.assertEqual(agent.last_dialogue_route, "rules")
        self.assertEqual(agent.last_dialogue_nodes, ["llm_understand", "rule_fallback"])

    def test_long_text_question_does_not_replace_current_text(self) -> None:
        agent = ConversationalAgent(use_llm=False)
        agent.handle_turn("文本：南京南站:坐高铁在南京南站下。南京南站")

        reply = agent.handle_turn("不能是长文本吗")

        self.assertIn("可以是长文本", reply)
        self.assertEqual(agent.state.text, "南京南站:坐高铁在南京南站下。南京南站")

    def test_deepseek_intent_updates_dialogue_state(self) -> None:
        agent = ConversationalAgent(llm_client=FakeDeepSeekClient())

        reply = agent.handle_turn("用比赛知识库，把这句话里的站点和交通方式链接一下")

        self.assertIn("信息已经齐了", reply)
        self.assertEqual(agent.state.kb_id, "ccks2019-v1")
        self.assertEqual(agent.state.text, "南京南站:坐高铁在南京南站下。南京南站")
        self.assertEqual(agent.state.mention_texts, ["南京南站", "高铁"])
        self.assertEqual(agent.last_dialogue_route, "deepseek")
        self.assertEqual(agent.last_dialogue_nodes, ["llm_understand", "finalize_action"])

    def test_movie_intent_infers_ccks_and_keeps_alias_expansion(self) -> None:
        agent = ConversationalAgent(llm_client=FakeMovieDeepSeekClient())

        reply = agent.handle_turn("李导演的《断背山》真是令人动人 其中实体是李导演和断背山")

        self.assertIn("信息已经齐了", reply)
        self.assertEqual(agent.state.kb_id, "ccks2019-v1")
        self.assertEqual(agent.state.mention_texts, ["李导演", "断背山"])
        self.assertEqual(agent.state.mention_aliases, {"李导演": ["李安"]})


if __name__ == "__main__":
    unittest.main()
