# 复现实验

## 环境与模型

原实验为 Apple M5、16GB、macOS 26.4.1、Python 3.12.14、MLX 0.32.4、MLX-LM 0.32.2、Ollama 0.40.11。安装以 uv.lock 和前端锁文件为准。requirements-lock.txt 为原环境快照。

按 README 安装 `train` extra 后运行 prepare_runtime.sh。MLX 训练只在 Apple Silicon macOS 验证；Linux 可执行非 MLX 检查。模型从 config/model_sources.json 指定的官方 revision 下载并核验 SHA-256，llama.cpp 转换版本在 scripts/vendor/llama.cpp/SOURCE.json。Git 不分发权重或 adapter。

## 原文与索引

可用仓库冻结正文直接 `index` 建索引；本地原文查看和完整离线验收还需 data/raw：

```bash
uv run --frozen --extra train python scripts/restore_sources.py
```

此命令只补取清单原文件，不覆盖冻结正文，并核验原始 SHA-256。学校网页可能变化；不一致文件不被冒充原快照，不覆盖已有合格文件，结果写入 results/source_restore.json。若原快照不可取，仍可看清洗正文与 URL，但不可声称已重现原文离线验收。

`python -m njupt_qa.cli crawl` 会刷新正文，需在独立副本中运行，之后重建数据、索引和评测，不能沿用旧分数。

## 训练

首版为全部 28 层 q_proj/v_proj、rank 16、scale 2、dropout 0.05、学习率 1e-4、batch 1、梯度累积 8、最大长度 1536、seed 42。仅最终 assistant 回答计损失；20 步试训后运行 3600 微步，按验证 loss 选 adapter，再合并、同精度转换和注册。

原实验最佳日志点 799、验证 loss 0.240224，耗时与数值可能因环境变化，不保证逐位一致。20 步预检若超过 10GB，正式配置启用检查点、降至 1024；应重新核验数据长度，防止答案截断。

后续调优脚本为可选研究入口，不随默认启动执行。搜索已停止，未用调优候选替换原基线。已完成运行从基座开始，没有完成断点续训；仅 adapter 不保证可恢复优化器动量。

## 评测与离线

历史结果在 experiments/2026-10-26，新运行在 results。主对照统一来源、Q8_0、提示和生成参数，关闭计算器与网页规则拒答。计算专项与四道验收单列，工具收益包含完整表格取证和确定性计算。

同配置评测支持续跑中断的问答，但拒绝合并不同协议或模型 digest。改变代码、提示或数据前，先另存本地旧结果或使用新副本。自动规则分不是人工盲评，引用命中不代表逐句语义蕴含。

pytest 不调用真实 Ollama。BGE tokenizer 缺失时跳过一个模型集成检查，MLX 未安装时跳过小模型检查。CI 不证明真实训练和推理已完成。

离线脚本依赖 macOS sandbox-exec，同时禁止 Ollama 和后端非回环外联，保留回环，并探测外联确实被拒绝。原实验为 16GB，没有在 8GB 整机另行验收。
