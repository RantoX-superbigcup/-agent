# Badcase 模板

## 样本信息

- `doc_id`:
- `mention_id`:
- `mention_text`:
- `gold_entity_id`:
- `predicted_entity_id`:
- `predicted_status`:

## 错误类型

- 候选未召回 / 候选排序错误 / NIL 误判 / 共指错误 / 词库缺失 / 其他

## 触发原因

- 上下文不足：
- 别名冲突：
- 领域词缺失：
- 规则阈值问题：

## 修复建议

- 补充别名：
- 增加关键词：
- 调整阈值：
- 更换召回或重排策略：
