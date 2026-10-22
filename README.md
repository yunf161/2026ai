# 人工智能课程实验

本仓库整理 lab1—lab7 的课程实验源码，涵盖 Python 基础、一阶逻辑归结、启发式搜索、遗传算法、回归、CNN、DQN，以及 LangGraph 实验报告生成系统。

各实验独立运行，历史版本保留用于比较学习过程，不代表每个变体都已完成或经过全面测试。原始实验目录不作修改；不携带原 Git 历史、本机凭据、环境目录、编译结果或重复打包文件。

## 实验索引

| 实验 | 主题 | 建议入口 |
| --- | --- | --- |
| lab1 | Python、二分查找与矩阵练习 | `lab1/binary_search.py`、`lab1/matrix.py` |
| lab2 | 最一般合一与一阶逻辑归结 | `lab2/yibanheyi.py`、`lab2/yi_jie_luo_ji_v1.py` |
| lab3 | 15数码 A*/IDA*、TSP 遗传算法与局部搜索 | `lab3/15-puzzle_a.py`、`lab3/15-puzzle-ida.py`、`lab3/yichuan_v4.py` |
| lab4 | 房价回归、手写网络及特征处理 | `lab4/house_final-v2.py` |
| lab5 | 基于 PyTorch 的图像分类 | `lab5/cnn-final.py` |
| lab6 | CartPole DQN、经验回放、多步回报及对比 | `lab6/DQN/main.py`、`lab6/DQN/agent_dir/agent_dqn.py` |
| lab7 | LangGraph 报告生成、RAG、记忆与工具调用 | [lab7/README.md](lab7/README.md) |

## 运行方式

使用独立 Python/Conda 环境，按实验安装依赖。lab2 使用 `match`，需要 Python 3.10 或更新版本；本次离线检查使用 Python 3.12。

经典算法示例：

```bash
cd lab1
python binary_search.py
cd ../lab2
python yibanheyi.py
python yi_jie_luo_ji_v1.py
cd ../lab3
python 15-puzzle_a.py
python 15-puzzle-ida.py
```

lab2、lab3、lab4 多处使用当前目录下的相对输入路径，应先进入对应实验目录。TSP 遗传算法的默认实例和迭代量较大，运行前先查看 `main()` 的实例路径和迭代参数。

lab4 需要 NumPy、Matplotlib；lab5 需要 PyTorch、torchvision、Matplotlib 与自行准备的分类图片集，见 [数据说明](lab5/DATASET.md)。建议将训练类实验放在单独环境，不直接覆盖已有环境。

lab6 保留课程给定的历史 `requirements.txt` 作为环境参考，其包含较旧的 TensorFlow、PyTorch 与 Gym 版本，不建议在现代 Python 中直接全量安装。本实验主线使用 PyTorch、NumPy、Gym、tensorboardX 和 Matplotlib；代码主要采用旧 Gym `reset/step` 接口，需选择兼容环境或先做 API 迁移。`--train_pg` 分支引用的 AgentPG 不在当前目录，不作为支持功能。

lab7 的云端模型与 embedding 调用可能产生费用。请先配置自己的凭据、确认模型名称可用，并阅读其运行说明。

## 数据与发布范围

包含：54个 Python 文件、一个 C++ 练习、lab2/3 小型输入、TSP 文件、lab4 CSV，以及依赖和正式说明。lab7 只保留主实现，不重复附带课堂示例解压包、私有课件、向量数据库和历史生成报告。

未包含：约240 MB 的CNN图片及其ZIP、课程讲义/作业PDF、实验报告PDF/DOCX、模型权重、Chroma索引、历史输出、个人调试记录、API Key。小型输入仅作为实验复现材料；原有注释与课程来源信息保留，不对上游数据或框架重新指定许可证。

## 验证与已知边界

```bash
python -m unittest discover -s tests -v
```

验证包含所有发布 Python 文件的语法，以及二分查找、矩阵相加、A*/IDA* 小实例和 DQN 多步经验回放等轻量检查。不会训练神经网络、调用模型、下载权重或创建向量索引。

lab1 的 `matrix_mul` 仍为未完成练习；部分历史版本存在已知局限。语法通过不代表训练收敛、算法最优性或报告事实正确。详细范围见 [验证说明](docs/verification.md)。
