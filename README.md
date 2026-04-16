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

与 ALFWorld（物理操作）和 Mind2Web（网页交互）不同，SoMe 的核心能力是
**工具链规划 + 信息检索**——agent 不改变世界状态，而是通过调用工具从只读数据库中
提取信息，最终提交答案。

#### 4.6.1 World State（隐藏状态）

```python
world_state = {
    "database_slice": {
        "posts": [...],          # 该任务相关的帖子子集（几百条）
        "users": {...},          # uid -> profile + 历史帖子
        "knowledge_base": [...]  # 外部知识条目（MID 任务用）
    },
    "ground_truth": "Yes",       # 正确答案（agent 不可见）
    "oracle_tool_chain": [       # 参考工具调用序列
        "user_search", "data_folder", "submit_answer"
    ],
    "task_type": "UBP",
    "data_folders": {}           # 工具运行时产生的临时文件夹
}
```

关键：`database_slice` 是从 9M 帖子库中为该任务切出的**自包含子集**，
包含解题所需的全部数据，不依赖外部数据库。

#### 4.6.2 Observation Space（agent 可见）

每步 agent 看到：

- `query`：自然语言问题
- `available_tools`：当前可调用的工具列表
- `last_tool_result`：上一步工具返回的结果（帖子列表 / 用户画像 / 摘要等）
- `history`：已调用的工具序列
- `step_index` / `remaining_steps`

agent 看不到 `ground_truth`、`oracle_tool_chain`、也无法直接读取 `database_slice`
（必须通过工具间接查询）。

#### 4.6.3 Action Space

8 个 SoMe 工具 + 1 个终止动作：

| Action | Payload | 功能 |
|--------|---------|------|
| `post_search` | `{location, start_time, end_time}` | 按地点/时间检索帖子 |
| `user_search` | `{uid}` | 查询用户画像 + 历史帖子索引 |
| `topic_search` | `{topic_name}` | 按话题检索帖子 |
| `post_retrieve` | `{query, topk}` | 语义检索相关帖子 |
| `topic_clustering` | `{folder_name}` | 对已检索帖子做聚类 |
| `topic_summarization` | `{folder_name}` | 对聚类结果生成摘要 |
| `knowledge_retrieve` | `{query, topk}` | 检索外部知识库 |
| `data_folder` | `{folder_name, start_idx, end_idx}` | 分页读取临时数据 |
| `submit_answer` | `{answer}` | 提交最终答案，结束 episode |

参数缺失 → 返回错误提示 + 负奖励；对只读数据库执行查询，**不修改 world_state**
（与 ALFWorld 的状态可变、Mind2Web 的表单可写形成对比）。

#### 4.6.4 Transition Engine

```
agent 发出 Action(tool_name, params)
  → sandbox 在 database_slice 上执行该工具
  → 返回工具结果作为新 observation
  → 若 tool_name == "submit_answer"：与 ground_truth 比对，episode 结束
```

核心特征：
- **只读**：工具执行不修改数据库，只产生查询结果
- **确定性**：相同参数 → 相同结果（无随机性）
- **中间存储**：`post_search` 等工具会将结果存入 `data_folders`，
  后续 `topic_clustering` / `data_folder` 可引用

与其他 benchmark 的对比：
- ALFWorld：`take apple` → `apple.location = inventory`（状态可变）
- Mind2Web：`TYPE "Boston"` → `form_values["destination"] = "Boston"`（状态可变）
- SoMe：`user_search(uid)` → 返回用户数据（状态不变，只增加 observation）

#### 4.6.5 Goal / Success Criterion

按 `task_type` 区分：

| 任务类别 | 答案类型 | 成功判定 |
|---------|---------|---------|
| UBP / MID / MCR | 分类（Yes/No/True/False） | `answer == ground_truth` |
| UEA | 分类（情绪标签） | `answer == ground_truth` |
| UCS | 分类（Yes/No） | `answer == ground_truth` |
| RED / SES | 生成（摘要文本） | LLM-as-a-Judge 评分 |
| SMQ | 生成（开放回答） | LLM-as-a-Judge 评分 |

分类任务用精确匹配，生成任务用 LLM judge（预留接口，可替换为 ROUGE/BERTScore）。

#### 4.6.6 Reward Design

**Outcome / Process 分割**：90% outcome + 10% shaping

**Outcome（二元/连续）**：
- 分类任务：答案正确 = 1.0，错误 = 0.0
- 生成任务：LLM judge 评分 ∈ [0, 1]

**Process shaping（在 10% 预算内）**：

| 维度 | 范围 | 信号 |
|------|------|------|
| `tool_chain_quality` | [0, 1] | 与 oracle_tool_chain 的编辑距离归一化 |
| `efficiency` | [-1, 0] | 超出 oracle 长度的步数惩罚 |
| `format_compliance` | {0, 1} | LLM 输出是否可解析为合法 Action JSON |
| `hallucination_penalty` | [-1, 0] | 引用不存在的 uid/post_id 扣分 |
| `redundancy_penalty` | [-1, 0] | 重复调用相同工具+相同参数扣分 |

**Per-step credit**：触发首次有效工具调用（如首次 `user_search` 返回非空）的
step 获得 milestone credit，用于 GRPO per-step advantage。

#### 4.6.7 Task Synthesis（SandboxSpec 生成）

三阶段流水线：

**Stage 1 — Database Slicing**
- 输入：完整 SoMe 数据库 + 一条标注查询
- 根据 oracle_tool_chain 追踪该查询涉及的 uid、post_id、topic
- 收集所有被引用的帖子 + 用户 + 知识条目
- 加入少量干扰项（同时间段/同地点的无关帖子）
- 输出：自包含 `database_slice`（几百条记录，而非 9M）

**Stage 2 — Oracle Annotation**
- 对每条查询记录解题所需的最短工具链
- 标注预期中间结果（用于 milestone 检测）

**Stage 3 — Difficulty Control**
- Easy：oracle 链 ≤ 3 步，分类任务，单工具即可解答
- Medium：oracle 链 4-6 步，需组合 2-3 个工具
- Hard：oracle 链 > 6 步，需聚类/摘要，或 LLM judge 任务

每条 SandboxSpec 包含完整 `database_slice` + `query` + `ground_truth` +
`oracle_tool_chain`，可独立运行。

#### 4.6.8 Validation

| 检查项 | 通过条件 | 捕获问题 |
|--------|---------|---------|
| Schema 完整性 | 必填字段非空，database_slice 非空 | 残缺 spec |
| 可解性 | Oracle policy 按 oracle_tool_chain 执行后得到正确答案 | 数据切片遗漏 |
| 非平凡性 | 不调工具直接 submit 的正确率 < 30% | 猜测即可的任务 |
| 数据库一致性 | oracle_tool_chain 引用的 uid/post_id 都存在于 slice 中 | 悬空引用 |
| 去重 | 同 batch 内无重复 query + database 指纹 | 冗余 spec |

只有通过全部检查的 spec 进入训练池。

---
