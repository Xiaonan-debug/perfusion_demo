## 5. InterCode - 交互式代码环境

### 5.1 概述

**InterCode** 是 Princeton NLP 开发的交互式代码环境框架，支持多种编程领域（Bash、SQL、Python、CTF）。Agent 在 Docker 容器中执行代码，根据执行反馈进行多轮交互。

| 属性 | 值 |
|------|-----|
| 环境类型 | 5 种 (Bash, SQL, Python, CTF, SWE-Bench) |
| 接口 | Gymnasium API |
| 隔离方式 | Docker 容器 |
| 难度 | 中 |

### 5.2 环境详情

| 环境 | 数据来源 | 任务示例 |
|-----|---------|---------|
| Bash | NL2Bash | 文件系统操作、命令行任务 |
| SQL | Spider | 数据库查询 |
| Python | MBPP | 函数实现 |
| CTF | picoCTF | 安全挑战、逆向工程 |

### 5.3 数据格式

**Bash 格式**:
```json
{
  "query": "Count the number of files and directories in the testbed directory",
  "gold": "find /testbed | wc -l"
}
```

**SQL 格式**:
```json
{
  "db": "poker_player",
  "query": "What are the names of poker players, ordered by final tables made?",
  "gold": "SELECT T1.Name FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID ORDER BY T2.Final_Table_Made",
  "hardness": "medium",
  "db_tables": {
    "poker_player": ["Poker_Player_ID", "People_ID", "Final_Table_Made", "Earnings"],
    "people": ["People_ID", "Name", "Nationality", "Birth_Date"]
  }
}
```

**Python 格式**:
```json
{
  "query": "Write a function to find similar elements from two tuple lists.",
  "gold": "def similar_elements(test_tup1, test_tup2):\n  return tuple(set(test_tup1) & set(test_tup2))",
  "tests": [
    "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
    "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)"
  ]
}
```

**CTF 格式**:
```json
{
  "task_id": 0,
  "query": "You have 'read' access to the file 'unpackme.flag.py' that contains the hidden flag",
  "gold": "picoCTF{175_chr157m45_85f5d0ac}",
  "tags": ["Reverse Engineering"]
}
```

### 5.4 奖励计算

| 环境 | 奖励计算方式 |
|-----|------------|
| Bash | 文件系统差异 (0.33) + 文件内容 MD5 (0.33) + 输出相似度 (0.33) |
| SQL | IoU (结果集交集/并集) + 排序准确度 |
| Python | 通过测试数 / 总测试数 |
| CTF | 二元奖励：flag 正确=1.0，否则=0.0 |

### 5.5 代表性样例

**SQL 任务**:

```
任务: "What is the average GNP and total population in nations
      whose government is US territory?"

数据库: world_1

交互:
Turn 1:
  Action: SELECT AVG(GNP), SUM(Population) FROM Country
          WHERE GovernmentForm LIKE '%US territory%'
  Observation: [(510.333333, Decimal('329000'))]
  Reward: 1.0 ✓
```

**CTF 任务 (逆向工程)**:

```
任务: "You have 'read' access to 'unpackme.flag.py' containing the hidden flag"

交互:
Turn 1:
  Action: cat unpackme.flag.py
  Observation:
    import base64
    from cryptography.fernet import Fernet
    key = b'VUovTmFxSnhOZ3dSSFlqRkFlJHM5OHlmPjxdK11TWG4='
    cipher = Fernet(key)
    payload = b'gAAAAABi...'
    # (encrypted flag)

Turn 2:
  Action: pip3 install cryptography
  Observation: Successfully installed cryptography-3.4.8

Turn 3:
  Action: python3 -c "from cryptography.fernet import Fernet;
          key=b'VUovTmFxSnhOZ3dSSFlqRkFlJHM5OHlmPjxdK11TWG4=';
          cipher=Fernet(key);
          print(cipher.decrypt(b'gAAAAABi...').decode())"
  Observation: picoCTF{175_chr157m45_85f5d0ac}

Turn 4:
  Action: submit picoCTF{175_chr157m45_85f5d0ac}
  Reward: 1.0 ✓
```

### 5.6 本仓库中的 ASP 实现方式

本仓库不直接调用 Docker 执行任意代码，而是实现一个
`datasets/intercode/` 的 **ASP-native 代码交互 sandbox**。它和原始 benchmark 的关系是：

- 原始 InterCode：真实 Docker 容器、真实代码执行、静态数据集（NL2Bash / Spider / MBPP / picoCTF）
- 本仓库实现：受 InterCode 启发的**模拟代码执行环境**，可交互、可生成、可训练

核心接口被重写成一个闭环：

`task_instruction -> terminal_state -> agent code_action -> sandbox execution -> reward -> next terminal_state`

**为什么不直接跑 Docker？**
GRPO 训练需要在每个 iteration 对 batch_size × group_size 条轨迹并行 rollout。如果每条轨迹都启动 Docker 容器执行任意代码，会带来三个问题：(1) 延迟——容器启动 + 代码执行的墙钟时间远大于模拟；(2) 安全——训练期间 agent 输出不可控，真实执行有安全风险；(3) 不可确定性——网络、文件系统竞态等导致不可复现。因此 ASP 的做法和 Mind2Web 一致：用 **受限模拟器** 替代真实执行，保留交互循环的结构，但把"执行"变成确定性的规则匹配。

#### 5.6.1 状态拆分

实现里严格区分两层状态：

- `world_state`（隐藏）
  - gold answer（ground truth 命令、查询、代码、或 flag）
  - 完整环境快照：Bash 的文件系统树 / SQL 的表数据 / Python 的测试用例 / CTF 的文件内容
  - 当前 step_index、执行历史、已提交标志、成功标志
- `terminal_state`（暴露给 agent）
  - 任务指令（自然语言）
  - 上一次执行的 stdout / stderr
  - 环境类型提示（如 "You are in a Bash terminal"）
  - 步数和剩余步数
  - 可选 metadata 提示：SQL 的表结构、Python 的函数签名要求、CTF 的文件名

这个拆分和 Mind2Web 的 `world_state` / `web_state` 完全对齐——agent 永远看不到 gold answer，只能通过交互获取反馈。

#### 5.6.2 当前支持的 sub-environment（4 种）

| Sub-Env | 数据来源 | 动作含义 | 模拟方式 |
|:---|:---|:---|:---|
| `bash` | NL2Bash 风格 | Shell 命令 | 模拟文件系统 + 受限命令解释器 |
| `sql` | Spider 风格 | SQL 查询 | sqlite3 in-memory（真实执行，确定性安全） |
| `python` | MBPP 风格 | Python 函数 | ast.parse 语法检查 + 受限测试执行 |
| `ctf` | picoCTF 风格 | Shell / Python | 模拟文件读取 + 已知库的确定性运算 |

统一动作空间为：

- `EXECUTE` — 提交代码/命令，返回执行结果
- `SUBMIT` — 提交最终答案，触发 gold-answer 比对
- `INSPECT` — 查看环境元信息（表结构 / 文件列表 / 函数签名等）

**和 ALFWorld / Mind2Web 的关键区别**：ALFWorld 有 14 个离散动作名，Mind2Web 有 5 个，InterCode 只有 3 个动作类型，但 `EXECUTE` 的 payload 是 **开放文本**（任意代码字符串），不是从固定集合中选择。这意味着 transition engine 必须能处理任意输入。

#### 5.6.3 Transition Engine（代码执行模拟）

与 Mind2Web 的 metadata-driven transition 思路一致，但落地方式因 sub-env 而异：

**SQL（最简单——可以真实执行）**：
- 把 `task_config.db_schema` + `task_config.db_rows` 加载进 sqlite3 in-memory database
- agent 提交的 SQL 直接在 sqlite3 上执行
- 返回真实结果集或 sqlite3 的错误信息
- SUBMIT 时用 IoU 比对结果集

这是 InterCode sandbox 中唯一可以做到"真实执行"的 sub-env，因为 sqlite3 是隔离的、确定性的、无副作用的。

**Bash（模拟文件系统）**：
- `task_config.filesystem` 是一个 `{path: content}` 字典，模拟目录树
- transition engine 实现一组受限命令：`ls`, `cat`, `find`, `wc`, `grep`, `head`, `tail`, `echo`, `mv`, `cp`, `rm`, `mkdir`
- 未识别的命令返回结构化的 `command not found` + 可用命令列表
- SUBMIT 时比对文件系统最终状态 + 命令输出

**Python（模拟测试）**：
- agent 提交 Python 代码 → `ast.parse` 做语法检查
- 如果语法正确，提取函数定义，逐个执行 `task_config.test_cases` 中的断言
- 返回 pass/fail 结果列表
- SUBMIT 时按 `passed_tests / total_tests` 计分

**CTF（模拟文件 + 确定性运算）**：
- `task_config.files` 定义可读文件的内容
- agent 用 `cat` 读文件后看到加密代码
- transition engine 可以模拟已知库（base64 decode、Fernet decrypt 等）的确定性运算
- SUBMIT 时做 flag 字符串精确匹配

#### 5.6.4 SandboxSpec 结构示例

**SQL 任务**：

```json
{
  "sandbox_type": "intercode",
  "sandbox_id": "intercode-sql-00042",
  "task_spec": {
    "name": "Query average GNP for US territories",
    "description": "What is the average GNP and total population in nations whose government is US territory?",
    "max_steps": 10,
    "metadata": {
      "sub_env": "sql",
      "difficulty": "medium",
      "source_dataset": "spider",
      "db_id": "world_1"
    }
  },
  "initial_state": {
    "task_config": {
      "sub_env": "sql",
      "instruction": "What is the average GNP and total population in nations whose government is US territory?",
      "gold_answer": "SELECT AVG(GNP), SUM(Population) FROM Country WHERE GovernmentForm LIKE '%US territory%'",
      "gold_result": [[510.333, 329000]],
      "db_schema": {
        "Country": ["Code", "Name", "Population", "GNP", "GovernmentForm"],
        "City": ["ID", "Name", "CountryCode", "Population"]
      },
      "db_rows": {"Country": ["...rows..."], "City": ["...rows..."]}
    },
    "terminal_output": "",
    "execution_history": [],
    "submitted": false
  },
  "action_space": {"actions": ["EXECUTE", "SUBMIT", "INSPECT"]},
  "observation_space": {"fields": ["instruction", "terminal_output", "sub_env", "step_index", "feedback"]}
}
```

**Bash 任务**：

```json
{
  "sandbox_type": "intercode",
  "sandbox_id": "intercode-bash-00017",
  "task_spec": {
    "name": "Count files in testbed",
    "description": "Count the number of files and directories in the testbed directory",
    "max_steps": 8,
    "metadata": {"sub_env": "bash", "difficulty": "easy"}
  },
  "initial_state": {
    "task_config": {
      "sub_env": "bash",
      "instruction": "Count the number of files and directories in the testbed directory",
      "gold_command": "find /testbed | wc -l",
      "gold_output": "42",
      "filesystem": {
        "/testbed/file1.txt": "hello world",
        "/testbed/dir1/": null,
        "/testbed/dir1/script.py": "print('hi')"
      }
    },
    "terminal_output": "$ ",
    "execution_history": [],
    "submitted": false
  }
}
```

#### 5.6.5 数据生成和 Agent-as-a-Sandbox

`task_generator.py` 负责程序化生成代码任务；`llm_task_agent.py` 可以进一步用
OpenRouter 对指令和环境做 LLM 增强。为了让"环境生成"本身也能 agent 化，
`agent_tools.py` 暴露以下工具：

- `seed_template(sub_env, difficulty)` — 从模板库抽取基础任务骨架
- `generate_db_schema(domain_hint)` — 生成 SQL 表结构和示例数据
- `generate_filesystem(task_hint)` — 生成 Bash 文件系统快照
- `generate_test_cases(function_spec)` — 为 Python 任务生成测试用例
- `generate_ctf_challenge(category)` — 生成带加密/编码逻辑的 CTF 题目
- `validate_with_oracle(spec)` — 用 gold answer 验证任务可解
- `estimate_difficulty(spec)` — 估算任务难度
- `finalize_spec(spec)` — 导出 `SandboxSpec`

因此这里不是只"加载 NL2Bash / Spider 数据"，而是可以持续出题、验题、导出 `SandboxSpec`。

#### 5.6.6 Reward 设计

`datasets/intercode/reward.py` 采用 **per-sub-env 的混合 reward**：

- 规则奖励（episode 级，占 ~90%）
  - **Bash**：`0.33 × fs_diff_score + 0.33 × output_similarity + 0.33 × md5_match`
  - **SQL**：`IoU(result_set, gold_set)` + 排序准确度（对 ORDER BY 查询）
  - **Python**：`passed_tests / total_tests`
  - **CTF**：二元奖励（flag 正确 = 1.0，否则 = 0.0）
- 过程奖励（step 级，占 ~10%）
  - `progress`：中间执行结果逐步接近 gold（如 SQL 部分列匹配、Python 部分测试通过）
  - `efficiency`：步数惩罚 `-step_index / max_steps`
  - `action_quality`：语法错误 / 运行时错误惩罚
  - `error_recovery`：在收到错误反馈后成功修正（正奖励）
  - `format_compliance`：输出可解析为合法 JSON action
  - `premature_submit`：无任何 EXECUTE 直接 SUBMIT 的惩罚

这个 reward 设计的核心洞察：InterCode 的训练价值不在于"一次写对代码"，而在于学会
**"写 → 执行 → 看错误 → 修正 → 再执行"的交互式调试循环**。因此 `error_recovery`
是唯一的正向过程奖励，而 `premature_submit` 是最重的过程惩罚。

也就是说，当前默认版本是 **programmatic dense reward**，不依赖 LLM judge。

#### 5.6.7 Validation

`datasets/intercode/validator.py` 当前会检查：

- schema 完整性（`task_config` 包含所有必要字段）
- oracle 可解性（gold answer 在模拟环境中执行后得到满分 reward）
- 非平凡性（随机 / 空 action 策略不能获得正 reward）
- 难度校准（oracle 解法所需步数在 `[2, max_steps-2]` 范围内）
- batch 内去重（基于 `(sub_env, instruction, gold_answer)` 指纹）

这保证了生成的任务至少是"可解的、可训练的、不会整批塌缩成重复样本"。

#### 5.6.8 建议新增模块

1. **Curriculum Scheduler（课程调度）**
   - 由 Scheduler Agent 通过 prompt 做课程调度
   - 核心目标：根据 per-sub-env 的 success_rate 和 syntax_error_rate 动态调整 sub-env 混合比例和难度

   **输入上下文**：
   - 近 N 轮训练摘要：`success_rate_per_sub_env`、`avg_steps_to_solve`、`syntax_error_rate`、`format_error_rate`
   - 当前课程配置：sub-env 混合比例、难度范围、max_steps

   **输出**：
   ```json
   {
     "difficulty_delta": 1,
     "sub_env_mix": {"bash": 0.3, "sql": 0.3, "python": 0.3, "ctf": 0.1},
     "max_steps": 8,
     "focus_on_error_recovery": true
   }
   ```

2. **Anti-Hacking Detector（奖励作弊检测）**
   - 重点检测模式：
     - **盲猜提交**：不执行任何代码，直接 SUBMIT 碰运气
     - **模板刷分**：对所有任务提交同一固定命令
     - **反馈忽视**：收到错误后重复提交相同代码
     - **空转消耗**：执行大量 INSPECT / 无关命令，不推进任务

   **建议输出**：
   ```json
   {
     "hack_flags": ["blind_submit", "ignoring_feedback"],
     "risk_score": 0.75,
     "penalty": -1.0,
     "invalidate": false,
     "evidence": ["Agent submitted 3 times without any EXECUTE step"]
   }
   ```

#### 5.6.9 推荐工作流（和 ASP 功能描述对齐）

```text
[1] Sandbox Agent 生成 task_config (instruction + gold + environment setup)
    -> 做结构校验、oracle 可解性验证、难度标注、batch 去重

[2] 对每个 task_config 构建 SandboxSpec
    -> 包含 sub_env 类型、初始环境状态、模拟器配置

[3] 训练模型在 InterCodeSandbox 中进行多轮代码交互
    -> agent 提交 EXECUTE / INSPECT / SUBMIT
    -> sandbox 返回 terminal_output + step reward
    -> 记录 turn-level code/output/reward

[4] Reward 模块打分
    -> per-sub-env 的 gold-answer 比对 + dense shaping

[5] 课程调度器根据训练信号调节下一批样本
    -> sub-env 混合比例 / 难度范围 / max_steps
```

#### 5.6.10 实现时的关键约束（建议直接写成规则）

- **安全性**：sandbox 不执行任意用户代码（SQL 例外：用 sqlite3 in-memory，天然隔离）
- **Sub-env 独立性**：每个 sub-env 的 simulator 完全独立，新增 sub-env 只需实现 `Simulator` 接口
- **Gold answer 必须可验证**：每个 `SandboxSpec` 的 gold answer 必须在 validator 中通过 oracle 执行
- **评估可复现**：所有数据生成必须记录 `seed`
- **奖励可解释**：reward 必须输出结构化分项，不做黑盒聚合
- **错误反馈信息丰富**：stderr 应包含可操作的修正提示（如 "Column 'GNP' not found, available columns: Code, Name, Population"）

#### 5.6.11 运行方式

查看样例：

```bash
python -m datasets.intercode.demo --mode episode --sub_env sql
python -m datasets.intercode.demo --mode episode --sub_env bash
python -m datasets.intercode.demo --mode batch --sub_env python
```

训练入口：

```bash
python -m datasets.intercode.train --model_name Qwen/Qwen3-8B
```

如果只想让任务生成阶段使用 OpenRouter：

```bash
python -m datasets.intercode.train \
  --use_llm_task_generation \
  --openrouter_model openai/gpt-5-mini
```

---
