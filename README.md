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

### 4.6 ASP 训练环境设计（SoMe）

1. **Data Synthesis / Curation 层**：生成 persona、时间线、帖子语料、任务查询与 ground truth
2. **LLM-backed Tool Server 层**：用冻结 LLM agent 做工具后端，模拟社媒平台的动态反馈
3. **Reward / Judge 层**：规则指标 + 多维 LLM-as-a-Judge，支持过程级奖励

#### A. Spec 生成模块

目标：每条 episode 合成一个自洽的"小型社媒世界"——一批用户、一段时间线、一批帖子、一道任务查询和带证据链的标准答案。

**输入**：
- 任务类型（RED / SES / MID / UBP / UEA / UCS / MCR / SMQ）
- 难度参数（证据稀疏度、干扰帖子密度、persona 异质度、时间线深度）

**参与的 Agent 角色**：

| Agent | 职责 |
|-------|------|
| `Persona Agent` | 合成用户 profile：兴趣、写作风格、发帖节奏、关注关系 |
| `Timeline Agent` | 围绕一个中心话题生成时间序列事件（新兴事件 / 持续话题 / 谣言扩散） |
| `Post Generator Agent` | 按 persona + timeline 合成帖子正文、图片描述、互动计数 |
| `Distractor Agent` | 注入主题不相关帖子、相似但错误的声明、重复噪声 |
| `Query Agent` | 按任务类型生成测试查询（claim / user_id / topic / folder） |
| `Label Oracle Agent` | 基于 world_state 给出标准答案与证据链 |

**State 拆分（关键）**：
- `world_state`（隐藏）：persona 的私密字段、timeline 的真实事件、每条帖子是否为 distractor、虚假信息的真相证据
- `platform_state`（暴露）：工具调用能拿到的用户/帖子/话题视图、带噪声的 metadata、可见的互动计数

**建议字段**：

```json
{
  "episode_id": "some-000412",
  "task_type": "MID",
  "difficulty": {"distractor_density": 0.4, "evidence_sparsity": 0.6},
  "world_state": {
    "central_topic": "flight MH370 new debris found",
    "ground_truth": "false",
    "evidence_sources": ["news_ref_2024_03_11", "expert_analysis_12"],
    "timeline": [{"t": "2024-03-10", "event": "rumor starts"}],
    "personas": [{"uid": "u_001", "private_bias": "conspiracy"}],
    "posts": [{"post_id": "p_001", "uid": "u_001", "is_distractor": false}]
  },
  "query": {
    "prompt": "Please determine if the claim ... is true.",
    "gold_label": "false",
    "gold_evidence": ["news_ref_2024_03_11"]
  }
}
```

**6 阶段 Validator**：
1. Schema 完整性
2. Persona 多样性去重
3. 时间线因果一致性
4. Oracle 可解性（Label Oracle 基于 world_state 能给出稳定答案）
5. 非平凡性（naive baseline 不能直接命中，对应 SOTOPIA 的 `exploit_resistance`）
6. 反记忆扰动（post_id / uid / topic_id 哈希化，避免 pretrain 数据泄漏）

#### B. LLM-backed Tool Server 模块

目标：不是"查一个静态索引",而是让工具后端动态、可控、可复现。每个工具背后都有冻结 LLM agent 或规则引擎,整条 episode 内工具绑定不变,避免轨迹中工具行为漂移。

| 工具 | 后端 | 行为 |
|------|------|------|
| `post_search(location, time_range)` | 规则 + `Feed Synth Agent` | 从 world_state.posts 过滤,按时间/地点噪声重排 |
| `user_search(uid)` | `User Profile Agent` | 渲染公开资料(隐藏 private_bias) |
| `topic_search(topic)` | `Feed Synth Agent` | 合成带干扰的 topic feed |
| `post_retrieve(query, topk)` | `Retrieval Agent`(LLM semantic ranker) | 对 world_state.posts 做语义排序 |
| `topic_clustering(folder)` | `Clusterer Agent` | 对帖子生成聚类标签与分组 |
| `topic_summarization(folder)` | `Summarizer Agent` | 生成摘要,会被 Faithfulness Judge 评估 |
| `knowledge_retrieve(query, topk)` | `Knowledge Oracle Agent` | 从 world_state.evidence_sources 中检索真相片段 |

**核心规则**：
- 工具 agent **必须是冻结模型**,不能等于训练中模型,防止"工具-策略同步漂移"形成自证环
- episode 创建阶段一次性抽样所有工具 agent,写入元数据,中途不更换
- 所有工具输入输出记录在 `trajectory.tool_trace`,供 reward 与 anti-hacking 分析
- 可按权重、能力标签、成本上限抽样

**建议元数据**：

```json
{
  "episode_id": "some-000412",
  "training_agent_model": "Qwen/Qwen3-8B",
  "tool_agents": {
    "retrieval": "openai/gpt-4o-mini",
    "summarizer": "openai/gpt-4o-mini",
    "knowledge_oracle": "openai/gpt-4.1"
  },
  "tool_binding_policy": "fixed_per_episode",
  "seed": 20260415
}
```

#### C. Reward / Judge 模块（对应 SOTOPIA 的 Reward / Judge）

目标：不只给"答案对不对"的二元信号,而是对轨迹做可训练的细粒度奖励分解。

建议 reward 结构：
- `answer_correctness`：最终答案对比 gold_label（0-10）
- `evidence_grounding`：引用的 post_id / evidence_source 是否真实存在（0-5）
- `tool_use_efficiency`：工具调用数量 vs 最短可行路径的比值（0-5）
- `faithfulness`：摘要/回答中是否出现 world_state 以外内容（-10 ~ 0）
- `hallucination_penalty`：捏造不存在的用户/帖子/事实（-10 ~ 0）
- `privacy_violation`：是否读取或泄露了 private_profile 字段（-10 ~ 0）
- `reasoning_quality`：推理链与证据一致性（0-10）
- `format_compliance`：结构化输出可解析性（0-1）

混合奖励：
- **Rule-based（硬约束）**：答案字符串比对、引用 id 存在性、工具调用格式、隐私红线
- **LLM-as-a-Judge（软维度）**：Faithfulness / Reasoning / Hallucination / Privacy 独立 judge,多判打分再聚合

建议 judge 输出：

```json
{
  "scores": {
    "answer_correctness": 8.0,
    "evidence_grounding": 4.5,
    "tool_use_efficiency": 3.0,
    "faithfulness": 0.0,
    "hallucination_penalty": -1.0,
    "privacy_violation": 0.0,
    "reasoning_quality": 7.2,
    "format_compliance": 1.0
  },
  "rationale": "Agent reached correct conclusion with 2 valid evidence sources but cited one nonexistent post_id.",
  "confidence": 0.84
}
```

---

#### E. 推荐工作流（和 ASP 功能描述对齐）

```text
[1] Spec Agent 组生成 persona + timeline + posts + query + gold_label
    -> 6 阶段 validator:schema / 多样性 / 因果 / oracle / 非平凡 / 反记忆

[2] 从 tool-agent 池经 OpenRouter 抽样工具后端
    -> 将 tool_agent_ids 与 seed 固定绑定到 episode

[3] 训练 agent 通过工具接口与 sandbox 多轮交互
    -> 记录 tool_trace / reasoning_chain / final_answer

[4] Reward 模块打分
    -> rule-based(答案 + 引用 id + 隐私) + multi-judge(faithfulness / reasoning / privacy)

[5] Anti-Hacking Detector 审核
    -> hack_flags 与 risk_score,决定样本是否进入训练

[6] Curriculum Scheduler 根据任务粒度指标调整下一批 spec 配置
    -> 任务混合比 / 干扰密度 / 证据稀疏度 / 工具预算
```

---


#### G. Agent 在三大模块中的参与一览

| 模块 | SOTOPIA | SoMe |
|------|---------|------|
| Spec 生成 | Profile + Scenario Agent | Persona / Timeline / Post Generator / Distractor / Query / Label Oracle |
| 环境交互 | 对手模型池(固定绑定) | Feed Synth / User Profile / Retrieval / Clusterer / Summarizer / Knowledge Oracle(固定绑定) |
| Reward / Judge | Rule + 多维 LLM Judge | Rule(答案 + 引用 + 隐私) + Faithfulness / Reasoning / Hallucination / Privacy Judges |
| 课程 / 反作弊 | Scheduler + Detector Agent | Scheduler + Detector Agent(新增 SoMe 特有作弊模式) |

---
