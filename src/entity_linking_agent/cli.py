"""Conversational Chinese terminal agent for Topic 10 entity linking."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from entity_linking_agent.config import load_config
from entity_linking_agent.core.contracts import KnowledgeBaseEntity, MentionRecord
from entity_linking_agent.core.dialogue_workflow import DialogueWorkflow
from entity_linking_agent.core.service import Topic10EntityLinkingService
from entity_linking_agent.kb.ccks2019 import load_ccks2019_entities
from entity_linking_agent.llm.deepseek_client import DeepSeekChatClient


EXIT_WORDS = {"q", "quit", "exit", "退出", "结束", "再见"}
RUN_WORDS = {"运行", "开始", "链接", "执行", "分析", "识别"}
RESET_WORDS = {"清空", "重置", "重新来", "重新开始"}


@dataclass
class DialogueState:
    kb_id: str = "sample-energy-v1"
    text: str = ""
    mention_texts: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="课题10实体链接对话式终端智能体")
    parser.add_argument("--text", help="待链接文本，长文本也可以")
    parser.add_argument("--mentions", help="mention列表，用逗号分隔，例如：南京南站,高铁")
    parser.add_argument("--kb", default="sample-energy-v1", help="sample-energy-v1 或 ccks2019-v1")
    parser.add_argument("--json", action="store_true", help="以JSON格式输出完整结果")
    parser.add_argument("--demo", action="store_true", help="运行内置演示样例")
    parser.add_argument("--no-llm", action="store_true", help="禁用DeepSeek对话理解增强")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.demo:
        run_once(
            text="南京南站:坐高铁在南京南站下。南京南站",
            mention_texts=["南京南站", "高铁"],
            kb_id="ccks2019-v1",
            as_json=args.json,
        )
        return 0

    if args.text and args.mentions:
        run_once(
            text=args.text,
            mention_texts=parse_mentions_arg(args.mentions),
            kb_id=args.kb,
            as_json=args.json,
        )
        return 0

    ConversationalAgent(use_llm=not args.no_llm).run()
    return 0


class ConversationalAgent:
    """Dialogue manager for entity linking demos."""

    def __init__(
        self,
        llm_client: Optional[DeepSeekChatClient] = None,
        use_llm: bool = True,
    ) -> None:
        self.state = DialogueState()
        self.llm_client = llm_client or DeepSeekChatClient()
        self.use_llm = use_llm and bool(getattr(self.llm_client, "is_configured", False))
        self.dialogue_workflow = DialogueWorkflow(
            llm_client=self.llm_client,
            use_llm=self.use_llm,
            rule_parser=build_rule_action,
        )
        self.last_dialogue_route = "unknown"
        self.last_dialogue_nodes: list[str] = []

    def run(self) -> None:
        print_banner()
        if self.use_llm:
            model = getattr(getattr(self.llm_client, "config", None), "model", "deepseek")
            print_agent(f"DeepSeek对话理解已启用：{model}。")
        else:
            print_agent("未检测到 DEEPSEEK_API_KEY，当前使用本地规则对话；设置后会更聪明。")
        print_agent("我已经准备好了。你可以直接说：链接“南京南站:坐高铁在南京南站下”，实体是 南京南站, 高铁。")
        print_agent("也可以先说“换成CCKS知识库”，再分步告诉我文本和mention。长文本也支持，输入“帮助”查看示例。")

        while True:
            try:
                user_text = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAgent：收到，已退出。")
                break

            if not user_text:
                print_agent("我在。你可以给我一段短文本，或者告诉我 mention。")
                continue

            if user_text.lower() in EXIT_WORDS:
                print_agent("好的，课题10实体链接对话结束。")
                break

            reply = self.handle_turn(user_text)
            if reply:
                print_agent(reply)

    def handle_turn(self, user_text: str) -> str:
        workflow_result = self.dialogue_workflow.invoke(
            user_text=user_text,
            current_state={
                "kb_id": self.state.kb_id,
                "has_text": bool(self.state.text),
                "text_preview": self.state.text[:300],
                "mentions": self.state.mention_texts,
            },
        )
        self.last_dialogue_route = workflow_result.get("route", "unknown")
        self.last_dialogue_nodes = [event["node"] for event in workflow_result.get("node_events", [])]
        return self._apply_action(workflow_result["action"])

    def _apply_action(self, action: dict) -> str:
        if action["action"] == "help":
            return help_text()

        if action["action"] == "reset":
            self.state = DialogueState()
            return "已清空上下文。现在请告诉我知识库、文本和 mention。"

        if action["kb_id"]:
            self.state.kb_id = action["kb_id"]
        if action["text"]:
            self.state.text = action["text"]
        if action["mentions"]:
            self.state.mention_texts = action["mentions"]

        if action["action"] == "reply" and not action["kb_id"] and not action["text"] and not action["mentions"]:
            return action["reply"] or "可以，我会结合当前上下文继续处理。"

        if action["run_requested"]:
            missing = self._missing_fields()
            if missing:
                return action["reply"] or missing
            if self.last_dialogue_nodes:
                print_agent(
                    "对话工作流: "
                    + " -> ".join(self.last_dialogue_nodes)
                    + f"，route={self.last_dialogue_route}"
                )
            run_once(
                text=self.state.text,
                mention_texts=self.state.mention_texts,
                kb_id=self.state.kb_id,
                as_json=False,
            )
            return action["reply"] or "本轮链接完成。你可以继续换文本、换mention，或者输入“清空”开始新任务。"

        if action["reply"]:
            return action["reply"]

        missing = self._missing_fields()
        if missing:
            return missing
        return "信息已经齐了。输入“运行”我就开始实体链接。"

    def _missing_fields(self) -> str:
        if not self.state.text:
            return f"当前知识库是 {self.state.kb_id}。请给我一条要链接的文本，长文本也可以。"
        if not self.state.mention_texts:
            guessed = guess_mentions(self.state.text)
            if guessed:
                return "我看到文本里可能有这些mention：" + "、".join(guessed) + "。请确认：实体是 " + ",".join(guessed)
            return "文本已收到。请告诉我要链接的 mention，用逗号分隔。"
        return ""


def run_once(text: str, mention_texts: list[str], kb_id: str, as_json: bool) -> None:
    service = Topic10EntityLinkingService()
    mentions = build_mentions(text, mention_texts)
    inline_entities: Optional[list[KnowledgeBaseEntity]] = None
    request_kb_id = kb_id

    if kb_id == "ccks2019-v1":
        inline_entities = load_ccks_subset(mention_texts)
        request_kb_id = "ccks2019-inline"

    print("\nAgent：正在执行 LangGraph 实体链接流程...")
    response = service.link(
        text=text,
        mentions=mentions,
        knowledge_base_id=request_kb_id,
        inline_entities=inline_entities,
    )

    if as_json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return

    print_response(response)


def build_mentions(text: str, mention_texts: list[str]) -> list[MentionRecord]:
    mentions: list[MentionRecord] = []
    cursor = 0
    for index, mention_text in enumerate(mention_texts, start=1):
        start = text.find(mention_text, cursor)
        if start < 0:
            start = text.find(mention_text)
        end = start + len(mention_text) if start >= 0 else None
        mentions.append(
            MentionRecord(
                mention_id=f"m{index}",
                text=mention_text,
                start=start if start >= 0 else None,
                end=end,
                sentence=text,
            )
        )
        if start >= 0:
            cursor = start + len(mention_text)
    return mentions


def load_ccks_subset(mention_texts: list[str]) -> list[KnowledgeBaseEntity]:
    config = load_config()
    print("Agent：正在从 CCKS2019 kb_data 按 mention 筛选候选实体...")
    entities = load_ccks2019_entities(
        kb_path=config.ccks2019_kb_path,
        alias_texts=set(mention_texts),
    )
    print(f"Agent：已筛选候选实体 {len(entities)} 个。")
    return entities


def build_rule_action(user_text: str, current_state: dict) -> dict:
    normalized = user_text.lower()

    if "帮助" in user_text or normalized == "help":
        return _rule_action(action="help", confidence=1.0)

    if is_long_text_question(user_text):
        return _rule_action(action="reply", reply=long_text_help(), confidence=1.0)

    if any(word in user_text for word in RESET_WORDS):
        return _rule_action(action="reset", confidence=1.0)

    kb_id = detect_kb_id(user_text)
    extracted_text, extracted_mentions = parse_user_payload(user_text)
    text = extracted_text
    if not text and looks_like_plain_text(user_text):
        text = user_text

    has_text_after_update = bool(text) or bool(current_state.get("has_text"))
    has_mentions_after_update = bool(extracted_mentions) or bool(current_state.get("mentions"))
    explicit_run = any(word in user_text for word in RUN_WORDS)
    auto_run = has_text_after_update and has_mentions_after_update and (bool(text) or bool(extracted_mentions))
    run_requested = explicit_run or auto_run

    if not kb_id and not text and not extracted_mentions and not run_requested:
        return _rule_action(action="unknown", confidence=0.0)

    reply = None
    if kb_id and not text and not extracted_mentions and not run_requested:
        reply = f"已切换到 {kb_id}。现在请给我文本和 mention。"

    return _rule_action(
        action="run" if run_requested else "update",
        kb_id=kb_id,
        text=text,
        mentions=extracted_mentions,
        run_requested=run_requested,
        reply=reply,
        confidence=1.0,
    )


def detect_kb_id(user_text: str) -> Optional[str]:
    if "ccks" in user_text.lower() or "2019" in user_text or "比赛" in user_text:
        return "ccks2019-v1"
    if "示例" in user_text or "能源" in user_text or "sample" in user_text.lower():
        return "sample-energy-v1"
    return None


def _rule_action(
    action: str,
    kb_id: Optional[str] = None,
    text: Optional[str] = None,
    mentions: Optional[list[str]] = None,
    run_requested: bool = False,
    reply: Optional[str] = None,
    confidence: float = 1.0,
) -> dict:
    return {
        "action": action,
        "kb_id": kb_id,
        "text": text,
        "mentions": mentions or [],
        "run_requested": run_requested,
        "reply": reply,
        "confidence": confidence,
    }


def print_response(response: dict) -> None:
    summary = response.get("summary", {})
    print("\n========== 运行摘要 ==========")
    print(f"trace_id: {response.get('trace_id')}")
    print(f"workflow_engine: {response.get('workflow_engine')}")
    print(f"route_decision: {response.get('route_decision', 'unknown')}")
    validation_errors = response.get("validation_errors") or []
    if validation_errors:
        print("validation_errors: " + ", ".join(validation_errors))
    print("graph_nodes: " + " -> ".join(response.get("graph_nodes", [])))
    print(
        "统计: "
        f"mention={summary.get('total_mentions', 0)}, "
        f"linked={summary.get('linked', 0)}, "
        f"ambiguous={summary.get('ambiguous', 0)}, "
        f"nil={summary.get('nil', 0)}, "
        f"review={summary.get('review_required', 0)}"
    )

    print("\n========== 链接结果 ==========")
    for item in response.get("results", []):
        print(f"\n[{item['mention_id']}] {item['text']}")
        print(f"  状态: {item['status']}    置信度: {item['confidence']}")
        print(f"  标准实体: {item.get('canonical_name') or '-'}")
        print(f"  实体ID: {item.get('linked_entity_id') or '-'}")
        rationale = item.get("evidence", {}).get("rationale", [])
        print(f"  依据: {' / '.join(rationale) if rationale else '-'}")
        print("  Top候选:")
        for candidate in item.get("candidates", [])[:5]:
            print(
                "    - "
                f"{candidate['canonical_name']} "
                f"({candidate['entity_id']}) "
                f"score={candidate['score']}"
            )

    print("\n========== 节点事件 ==========")
    for event in response.get("node_events", []):
        print(f"- {event['node']}: {event.get('detail', {})}")


def parse_user_payload(user_text: str) -> tuple[str, list[str]]:
    before, after = split_mention_clause(user_text)
    text = extract_text(before)
    mentions = parse_mentions_arg(after) if after else extract_mentions(user_text)
    return text, mentions


def split_mention_clause(user_text: str) -> tuple[str, str]:
    markers = [
        "mentions是",
        "mentions为",
        "mention是",
        "mention为",
        "mention:",
        "mention：",
        "实体是",
        "实体为",
        "实体:",
        "实体：",
        "指称是",
        "指称为",
        "指称:",
        "指称：",
    ]
    for marker in markers:
        if marker in user_text:
            before, after = user_text.split(marker, 1)
            return before.strip(" ，,。"), after.strip()
    return user_text, ""


def extract_text(user_text: str) -> str:
    patterns = [
        r"(?:文本|句子|短句|短文本|链接|分析)[：:]\s*(.+)$",
        r"[“\"](.+?)[”\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_text)
        if match:
            value = match.group(1).strip()
            value = re.split(r"(?:，|,)?\s*(?:实体|mention|mentions)[是为:：]", value, maxsplit=1)[0].strip()
            if value:
                return value
    return ""


def extract_mentions(user_text: str) -> list[str]:
    patterns = [
        r"(?:实体|mention|mentions|指称)[是为:：]\s*(.+)$",
        r"(?:实体|mention|mentions|指称)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match:
            return parse_mentions_arg(match.group(1))
    return []


def guess_mentions(text: str) -> list[str]:
    quoted = re.findall(r"《(.+?)》|“(.+?)”|\"(.+?)\"", text)
    guesses = [next(item for item in group if item) for group in quoted if any(group)]
    return list(dict.fromkeys(guesses))[:5]


def looks_like_plain_text(user_text: str) -> bool:
    command_words = ["知识库", "ccks", "sample", "实体", "mention", "运行", "帮助", "长文本"]
    if any(word.lower() in user_text.lower() for word in command_words):
        return False
    return len(user_text) >= 4


def parse_mentions_arg(value: str) -> list[str]:
    cleaned = (
        value.replace("，", ",")
        .replace("、", ",")
        .replace("；", ",")
        .replace(";", ",")
        .replace("和", ",")
    )
    return [item.strip(" 　。.!！?？") for item in cleaned.split(",") if item.strip(" 　。.!！?？")]


def help_text() -> str:
    return (
        "你可以这样和我说：\n"
        "1. 换成CCKS知识库\n"
        "2. 链接：南京南站:坐高铁在南京南站下。南京南站，实体是 南京南站,高铁\n"
        "3. 文本：这里可以粘贴一整段长文本。\n"
        "4. 实体是 国家电网,南方电网\n"
        "5. 运行\n"
        "其他命令：清空、退出。"
    )


def is_long_text_question(user_text: str) -> bool:
    normalized = user_text.lower()
    mentions_text_length = "长文本" in user_text or ("long" in normalized and "text" in normalized)
    asks_capability = any(word in user_text for word in ["吗", "能", "可以", "支持", "行不行", "只能", "不能"])
    return mentions_text_length and asks_capability


def long_text_help() -> str:
    return (
        "可以是长文本，不限于短句。当前 Agent 做的是实体链接，所以最好同时告诉我要链接哪些 mention；"
        "例如：文本：<粘贴长文本>，实体是 南京南站,高铁。"
        "如果只给长文本不给 mention，我会先保存文本，再请你补充 mention。"
    )


def print_agent(message: str) -> None:
    print(f"Agent：{message}")


def print_banner() -> None:
    print("=" * 66)
    print("课题10：实体链接与知识对齐对话式 Agent")
    print("我会多轮记住：知识库、文本、mention，然后调用 LangGraph 链接流程。")
    print("=" * 66)


if __name__ == "__main__":
    raise SystemExit(main())
