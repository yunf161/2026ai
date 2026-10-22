# LangGraph 实验报告生成系统

AI课程实验，使用Python、LangGraph、LangChain与Chroma，将提示词、代码、数据和参考资料组织为报告生成流程。主实现位于 `agents/src`。

## 流程

```text
输入解析与材料读取
→ 理论检索 / 方法分析
→ 章节生成
→ 检查与有界重写
→ Markdown 或 DOCX/PDF 导出
```

`graph/report_graph.py` 注册18个节点，以条件路由组织决策与执行。执行Agent的ReAct循环最多10步；检查失败后的默认重写上限为三轮。角色分工不等于多个模型或并行分布式执行。

课件RAG与长期记忆分离；长期记忆保存报告摘要、用户偏好与模板。文件解析、代码读取、CSV分析、网络检索与图像理解等作为工具接入。

## 环境与启动

使用独立Python环境，依赖见 `agents/requirements.txt`。该文件使用范围版本，是原实验依赖清单，不是完整可复现锁文件；请按自己的Python和平台核对兼容性。

```bash
python -m pip install -r agents/requirements.txt
cd agents/src
python main.py --prompt "根据提供的资料生成实验报告" --output-format markdown
```

程序默认从环境读取 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`；视觉与embedding分别可使用自己的环境配置。仓库根 `.env.example` 是空凭据模板。为避免工作目录影响，优先设置进程环境变量；`.env`、索引和输出均不提交。

主模型可用 `LLM_MODEL`、`LLM_BASE_URL` 配置。模板保留原实验模型名称，不保证该名称在所有账号或平台中可用。自行配置 `COURSEWARE_DIR`；缺省使用源码下的 `data/courseware`。需要特定中文字体时配置 `PDF_FONT_PATH`。

可选参数：`--data-files`、`--references`、`--code-file`、`--output-format`、`--output-dir`、`--verbose`。PDF导出可能依赖字体、文档转换工具及平台配置，不保证未安装这些组件的系统可直接导出。

## 知识库

课件与知识索引不随仓库发布。使用自己的PDF/Markdown资料，参考 `agents/src/tools/rag_search.py` 中的 `build_knowledge_base` 构建知识库。默认云端embedding可能产生费用；本地模型模式也需要独立准备模型与依赖。

## 使用边界

- `run_code` 只是临时目录中的子进程并带120秒超时，不具备容器或操作系统级安全隔离。只用于可信课程代码，不接受公网不可信输入。
- 格式检查节点目前只截取正文前500字符；不能当作整篇报告的完整事实或格式审查。
- 本次没有验证付费模型、视觉生成、完整报告生成或历史11项工具测试；仓库测试不应被解释成这些能力已在线验收。
- 不承诺断点续跑、并行提速、论文真实性或生产可用性。保留课程学习与实验性质。
