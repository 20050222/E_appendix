# 附件4输入审计报告

本报告由 `code/audit_attachment4.py` 只读生成；未训练模型、未修改原始 PKL 或视频。

- 对齐版本 PKL：20 个；未对齐版本 PKL：20 个。
- 当前审计确认：样本 ID 与文件名一致性、字段/shape、NaN/Inf、全零行、文本 attention mask 和视频文件存在性。
- 对齐版本没有显式 `audio_lengths`/`vision_lengths` 字段；其有效位置需要沿用第二问的 mask 恢复规则，不能仅凭全零值断言缺失。
- 附件4无真实标签，因此本阶段不计算 Accuracy、F1、MAE 或 Pearson。
- 最终推理仍依赖第二问的 `final_model.pt`、`normalization.npz`、模型配置和推理代码；这些文件当前未在 `04_robust_model` 目录发现。
