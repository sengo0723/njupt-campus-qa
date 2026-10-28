# 参与开发

先阅读 README、docs/ARCHITECTURE.md 和数据说明。提交前运行 pytest、前端 pnpm build 和 scripts/check_public_repository.py。无本地模型或 MLX 时，相应检查会明确跳过。

故障反馈请包含复现问题、模式、模型 revision/digest、年份筛选、引用证据和日志中的错误阶段。请删除私人信息，不提交会话数据库、凭据、模型或整个运行环境。

新增数据保留来源、年份、事实组及构造方法。固定测试和验收答案不加入训练，不根据测试分数反复选 checkpoint。训练、检索、计算与生成的效果分别报告，负结果也应保留。
