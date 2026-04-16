## 4. SoMe - 社交媒体平台

### 4.1 概述

**SoMe** (Social Media) 是一个评估 LLM Agent 在真实社交媒体场景中表现的 benchmark，发表于 AAAI 2026。涵盖帖子检测、用户行为预测、信息验证等任务。

| 属性 | 值 |
|------|-----|
| 帖子数量 | 9,164,284 |
| 用户画像 | 6,591 |
| 标注查询 | 17,869 |
| 任务类型 | 8 种 |
| 难度 | 中 |

### 4.2 任务分类

| 类别 | 任务 | 描述 |
|-----|------|------|
| 帖子中心 | 实时事件检测 (RED) | 从流数据中检测新兴事件 |
| 帖子中心 | 流事件摘要 (SES) | 对特定话题生成摘要 |
| 帖子中心 | 虚假信息检测 (MID) | 判断声明的真实性 |
| 用户中心 | 用户行为预测 (UBP) | 预测用户是否会点赞/评论/转发 |
| 用户中心 | 用户情绪分析 (UEA) | 分析用户对帖子的情绪反应 |
| 用户中心 | 用户评论模拟 (UCS) | 判断评论是否由特定用户撰写 |
| 综合 | 内容推荐 (MCR) | 预测用户是否对内容感兴趣 |
| 综合 | 社媒问答 (SMQ) | 回答关于社媒内容的问题 |

### 4.3 工具（动作空间）

| 工具 | 功能 | 参数 |
|-----|------|------|
| `post_search` | 按地点/时间搜索帖子 | location, start_time, end_time |
| `user_search` | 搜索用户画像和历史帖子 | uid |
| `topic_search` | 搜索特定话题的帖子 | topic_name |
| `post_retrieve` | 语义检索相关帖子 | query, topk |
| `topic_clustering` | 聚类帖子 | folder_name |
| `topic_summarization` | 生成聚类摘要 | folder_name |
| `knowledge_retrieve` | 检索外部知识（用于辟谣） | query, topk |

### 4.4 帖子数据格式（微博）

```json
{
  "id": "5120482505917816",
  "内容": "今天北京的天气真好，适合出去走走！",
  "发布时间": "2024-03-15 10:30:00",
  "发布地点": "北京",
  "转发数": 156,
  "评论数": 89,
  "点赞数": 1024,
  "图片数量": 2,
  "来自": "iPhone 15 Pro"
}
```

### 4.5 代表性样例

**任务**: 用户行为预测 (UBP)

```
查询:
请判断ID为5288580817的用户是否对以下帖子进行点赞：
{
  "id": "5120482505917816",
  "内容": "分享一个超实用的Python技巧...",
  "点赞数": 500
}

Agent 工作流:
1. 调用 user_search(uid="5288580817") 获取用户画像
   返回: {
     "用户名": "Python爱好者",
     "关注数": 200,
     "粉丝数": 1500,
     "简介": "程序员 | 技术分享",
     "历史帖子": [...]
   }

2. 调用 data_folder 读取用户历史帖子（最多100条）
   分析: 用户经常发布和互动技术类内容

3. 比对目标帖子与用户兴趣
   结论: 帖子是Python技术内容，符合用户兴趣

输出: "是"
真实标签: "是"
```

**任务**: 虚假信息检测 (MID)

```
查询:
Please determine if the claim "Building a wall on the U.S.-Mexico border
will take literally years." is true.

Agent 工作流:
1. 调用 knowledge_retrieve(query="US Mexico border wall construction time", topk=10)
   返回: [相关新闻报道、专家分析、政府文件...]

2. 分析检索到的事实:
   - 边境墙全长约1,954英里
   - 2017-2020年建设进度显示确实需要多年
   - 专家估计完整建设需要10+年

输出: "true"
真实标签: "true"
```

### 4.6 本仓库中的 ASP 实现方式

本仓库没有直接复用 SoMe 原始的静态评测管线，而是在 `datasets/some/` 下实现了一个
**ASP-native 工具调用 sandbox**。它和原始 benchmark 的关系是：

- 原始 SoMe：完整 900 万帖子数据库、MCP 工具服务、单次静态评测
- 本仓库实现：受 SoMe 启发的自包含工具调用环境，可交互、可生成、可训练

核心接口被重写成一个闭环：

`query -> tool_list + agent_state -> agent tool_call -> sandbox tool_execution -> tool_response + reward -> next agent_state`

和 ALFWorld（改变物理世界）、Mind2Web（改变网页状态）不同，SoMe 的世界是 **只读数据库**：
agent 通过工具调用获取信息，但不改变世界本身。核心训练信号是 **工具链规划** 和 **长上下文推理**。

#### 4.6.1 状态拆分

实现里严格区分两层状态：

- `world_state`
  - 隐藏数据库切片（帖子、用户画像、知识库报告）
  - 真实标签（ground truth）
  - oracle tool chain（最优工具调用序列）
  - 工具执行状态：已创建的 data_folder 及其内容
- `agent_state`
  - 暴露给 agent 的任务查询
  - 可用工具列表及参数 schema
  - 工具调用历史（tool_name + params + tool_response）
  - 已创建的 data_folder 名称列表

这样设计使得 agent 必须通过工具逐步获取信息，ground truth 完全隐藏在 world_state 中。

#### 4.6.2 当前支持的任务类型

8 种任务类型分为三类，和原始 SoMe 对齐：

| 类别 | 任务 | answer_type | 评估方式 |
|-----|------|-------------|---------|
| 帖子中心 | RED, SES, MID | generation / 6-class | LLM judge / ACC |
| 用户中心 | UBP, UEA, UCS | binary / 6-class | ACC |
| 综合 | MCR, SMQ | binary / generation | ACC / LLM judge |

统一动作空间为 9 种（8 个 SoMe 工具 + 终止动作）：

- `search_post`
- `search_topic`
- `search_user`
- `data_folder`
- `retrieve_post`
- `retrieve_knowledge`
- `post_clustering`
- `post_summarization`
- `submit_answer`（终止动作）

#### 4.6.3 数据生成和 Agent-as-a-Sandbox

`task_generator.py` 负责为每个任务切出自包含的 database slice（几百到几千条帖子 + 相关用户），
使得每个 `SandboxSpec` 可独立序列化、并行执行。`llm_task_agent.py` 可进一步用
OpenRouter 做文案增强。为了让"环境生成"本身也能 agent 化，`agent_tools.py` 暴露了以下工具：

- `slice_database` — 按任务类型从源数据切出子集
- `inject_noise` — 控制噪声帖子比例调整难度
- `annotate_oracle_chain` — 标注最优工具调用序列
- `validate_solvability` — 执行 oracle chain 确认可解
- `set_difficulty` — 设置 database_size / noise_ratio / max_steps
- `finalize_spec` — 导出 `SandboxSpec`

每个 SandboxSpec 必须附带 `oracle_tool_chain`，用于验证可解性和计算 step reward：

```json
{
  "task_type": "UBP",
  "oracle_tool_chain": [
    {"tool": "search_user", "params": {"uid": "5288580817"}},
    {"tool": "data_folder", "params": {"folder_name": "user_posts", "start_idx": 0, "end_idx": 50}},
    {"tool": "submit_answer", "params": {"answer": "Yes"}}
  ],
  "oracle_steps": 3
}
```

#### 4.6.4 Transition Engine

和 Mind2Web 的 metadata-driven transition 类似，SoMe 的 transition engine 是 **工具执行驱动** 的。
每个工具调用查询 database_slice，更新 `data_folders` 状态，返回 `tool_response` 作为 observation。

关键区别：

| | ALFWorld | Mind2Web | SoMe |
|---|----------|----------|------|
| Transition 驱动 | 硬编码 Python | 元素 `on_click` 元数据 | 工具执行引擎 |
| 世界变化 | 是（物体移动） | 是（表单/flag） | 否（数据库只读） |
| 核心训练信号 | 空间推理 | DOM 理解 | 工具链规划 |

#### 4.6.5 Reward 设计

`datasets/some/reward.py` 按任务类别采用不同 reward，统一成 `RewardBreakdown` 格式：

**分类任务（MID, UBP, UEA, UCS, MCR）— 规则为主**：

| 维度 | 范围 | 信号 |
|-----|------|------|
| goal_completion | {0, 1} | 答案和 ground truth 精确匹配 |
| tool_chain_quality | [0, 1] | oracle chain 覆盖率 |
| efficiency | [-1, 0] | 步数 / max_steps |
| format_compliance | {0, 1} | 输出可解析为有效工具调用 |
| hallucination_penalty | [-1, 0] | 虚构工具响应或格式错误 |

**生成任务（RED, SES, SMQ）— 混合 rule + LLM-judge**：

| 维度 | 范围 | 信号 |
|-----|------|------|
| accuracy | 0-5 | LLM judge: 信息准确性 |
| completeness | 0-5 | LLM judge: 关键信息覆盖度 |
| relevance | 0-5 | LLM judge: 与查询相关性 |
| goal_completion | [0, 1] | avg(scores) / 5 归一化 |

聚合方式和 ALFWorld 一致：success 主导（90%），shaped terms 上限 10%。

#### 4.6.6 Validation

`datasets/some/validator.py` 当前会检查：

- schema 完整性（SandboxSpec 字段齐全、database_slice 非空）
- solvability（oracle tool chain 可执行且答案正确）
- non-triviality（随机工具调用不能解出）
- database 一致性（tool 参数引用的 uid/topic 在 slice 中存在）
- batch 内去重（query + database fingerprint 不重复）

#### 4.6.7 建议新增模块

保留两个模块：`Tool-Chain Curriculum Scheduler` 和 `Tool Hallucination Detector`。

1. **Tool-Chain Curriculum Scheduler（工具链课程调度）**
   - 和 SOTOPIA 的 Curriculum Scheduler 同理，但调度维度为：task_type 混合比例、noise_ratio、database_size、max_steps
   - 由 Scheduler Agent 根据 `answer_accuracy_mean`、`tool_chain_match_rate`、`hallucination_rate` 做小步调整

2. **Tool Hallucination Detector（工具幻觉检测）**
   - SoMe 论文明确指出的核心问题：agent 虚构工具响应或工具调用格式
   - 重点检测：response hallucination（编造工具返回）、format hallucination（格式错误）、shortcut answering（不调工具直接答）、over-retrieval（反复同一工具消耗步数）
   - 接入方式同 SOTOPIA Anti-Hacking：reward 聚合前调用，`risk_score > τ` 时施加惩罚

#### 4.6.8 运行方式

查看样例：

```bash
python -m datasets.some.demo --mode episode --task_type UBP
```

训练入口：

```bash
python -m datasets.some.train --model_name Qwen/Qwen3-8B
```

如果只想让任务生成阶段使用 OpenRouter：

```bash
python -m datasets.some.train \
  --use_llm_task_generation \
  --openrouter_model openai/gpt-5-mini
```

---
