# 案例库 / Case Library

## 概述

本目录存放与 Formal-ProgramManage 知识体系对应的**项目案例**，用于实践映射、检索练习与交错学习。每个案例按统一结构编写，并映射到 CML/VL/AL 文档与标准。

**用途**：配合 [LEARNING_PATHS.md](../LEARNING_PATHS.md) 轨道 A/B、[03-retrieval-practice-questions.md](../12-learning-support/03-retrieval-practice-questions.md) 与 [05-interleaved-learning-paths.md](../12-learning-support/05-interleaved-learning-paths.md)。

---

## 案例库结构规范

### 目录组织

```
case-library/
├── README.md                 # 本说明与索引
├── template.md              # 案例写作模板（复制后填写）
├── software/                 # 软件开发类案例
│   └── (待补充)
├── engineering/              # 工程管理类案例
│   └── (待补充)
├── ai-ml/                    # AI/ML 项目案例
│   └── (待补充)
└── cross-domain/             # 跨领域/综合案例
    └── (待补充)
```

### 单案例结构（每个案例需包含）

| 区块 | 内容 |
|------|------|
| **1. 背景 (Context)** | 组织类型与规模、项目目标、初始约束、关键干系人 |
| **2. 过程 (Process)** | 时间线、关键决策、遇到的挑战与应对 |
| **3. 理论映射 (Theory Mapping)** | 使用的 CML/VL 模型（如 2.1 生命周期、2.3 风险、3.1 验证）与标准（PMBOK/ISO） |
| **4. 关键学习点 (Key Learnings)** | 成功因素、失败教训、可复用模式 |
| **5. 练习 (Exercises)** | 分析题或仿真题，可链到 [03-retrieval-practice-questions.md](../12-learning-support/03-retrieval-practice-questions.md) |
| **6. 资源 (Resources)** | 相关文档链接、延伸阅读 |

### 案例分类维度

- **按行业**：软件、工程、AI/ML、医疗、教育、综合等
- **按规模**：小型（&lt;10 人/&lt;6 月）、中型（10–50 人/6–18 月）、大型（&gt;50 人/&gt;18 月）
- **按结果**：成功、部分成功、失败（教训）

### 决策情境 / 伦理两难 / 老手经验 类案例模板

用于补充「实践智慧」（Phronesis）：情境判断、隐性经验、伦理取舍。可与 [07-practical-guidance.md](../07-practical-guidance.md) 及 CML 文档中的「实践解释」小节配合使用。

| 区块 | 内容 |
|------|------|
| **1. 情境 (Situation)** | 背景、时间压力、信息不完整或冲突的干系人诉求 |
| **2. 决策选项 (Options)** | 2–4 个可行方案及各自利弊（可含「不行动」） |
| **3. 考量因素 (Considerations)** | 伦理、合规、风险、组织政治、长期 vs 短期 |
| **4. 实际选择与结果 (Outcome)** | 采取了哪一选项、短期/长期结果、若有「老手经验谈」可附简短点评 |
| **5. 理论映射 (Theory)** | 对应 CML 风险/质量/生命周期或 VL 验证中的哪些概念；可链到 [THREE_LAYER_EXPLANATIONS.md](../THREE_LAYER_EXPLANATIONS.md) |
| **6. 反思问题 (Reflection)** | 1–2 条供读者自问的反思题（见 [12-learning-support/README.md](../12-learning-support/README.md) §反思性问题） |

**示例场景**：范围蔓延是否接受、质量与进度的权衡、上报风险 vs 先内部化解、跨文化沟通中的决策、合规与交付压力的冲突。

---

## 案例索引（待扩充）

| 案例名称 | 行业 | 规模 | 理论映射 | 文件 |
|----------|------|------|----------|------|
| （样板）敏捷产品迭代交付 | 软件 | 中型 | 2.1 生命周期、4.1 敏捷、2.3 风险 | [template-sample.md](./template-sample.md) |

---

## 实践模板与检查清单（链接）

- **风险登记册**、**WBS 模板**等实践制品将逐步放在 `templates_and_standards/` 或本目录下 `templates/`。
- 标准与季度审查见 [STANDARDS_ALIGNMENT.md](../STANDARDS_ALIGNMENT.md) 与 [SUSTAINABLE_EXECUTION_PLAN.md](../SUSTAINABLE_EXECUTION_PLAN.md)。

---

**Last Updated**: 2026-02-04
**Status**: 结构已建立；样板案例见 template-sample.md
