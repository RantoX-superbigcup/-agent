# 评测方案

## 对齐任务书指标

任务书建议指标包括：

- 链接准确率 >= 85%
- 同名异指消歧准确率 >= 85%
- NIL 检测 F1 >= 0.80
- 别名标准化召回率 >= 85%
- 共指消解准确率 >= 80%
- 可追溯性完整

## 当前框架内置评测

`scripts/evaluate.py` 当前提供以下基础指标：

- `overall_accuracy`
- `link_accuracy`
- `nil_precision`
- `nil_recall`
- `nil_f1`
- `alias_recall`

## 推荐后续补充

- 单独统计“同名异指”子集上的消歧准确率。
- 单独统计“行业新词 / 缺词库”场景下的 NIL 质量。
- 扩充 `gold` 字段，为共指样本增加 `is_coreference` 标签。
- 把错误样本自动归档到 `reports/` 目录，便于论文或技术报告整理。

## 运行方式

```powershell
python .\scripts\evaluate.py .\data\examples\sample_benchmark.json
```
