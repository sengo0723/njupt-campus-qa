# 开源组件与资料来源

本项目独立实现校园数据处理、问答编排、计算器、评测和聊天界面。Harman RAG 仅用于理解混合召回、融合排序、证据引用与诊断的设计思路，未导入其代码或业务资料。

- Qwen3-0.6B：Qwen 官方模型，Apache-2.0；权重不随 Git 分发，下载时保留上游许可，版本和校验值见 config/model_sources.json。
- bge-small-zh-v1.5、bge-reranker-base：BAAI 官方公开模型，模型说明随资源保留。重排使用经过 ONNX 加载验证的官方文件。
- MLX、MLX-LM：Apple 开源训练工具，MIT。
- llama.cpp：MIT；本项目固定版本的 Python 转换工具位于 scripts/vendor/llama.cpp，保留上游 LICENSE 与来源版本记录。
- React、Vite、TypeScript、Lucide、react-markdown、remark-gfm 等组件采用各自开源许可证；licenses/ 保留安装包的许可证副本，前端锁文件记录精确版本。
- FastAPI、Uvicorn、PyTorch、Transformers、ONNX Runtime 等后端工具采用各自许可证；Python 锁文件记录精确版本。

校园资料来自南京邮电大学官方公开网站。保存原文网址、文件、采集时间及校验值，不表示学校对本项目背书。原始资料的权利归相应权利人；测试和教学使用应保留来源。
