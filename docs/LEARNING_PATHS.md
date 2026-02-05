# 学习路径 / Learning Paths

## 概述

本文档为 Formal-ProgramManage 知识体系提供三条并行的学习轨道，便于不同背景与目标的学习者选择入口并配合 [12-learning-support](./12-learning-support/) 的间隔重复与检索练习。先备知识与难度分级见 [01-learning-prerequisites.md](./12-learning-support/01-learning-prerequisites.md) 与 [04-concept-difficulty-ranking.md](./12-learning-support/04-concept-difficulty-ranking.md)。

---

## 轨道 A：入门轨道（Foundation Track）

**目标**：建立基本概念框架，理解「为什么」；面向新手到高级初学者。
**时长**：约 2–4 周（按每周 3–5 小时计）。

| 阶段 | 内容 | 文档与资源 |
|------|------|------------|
| Week 1 | 项目管理核心概念（非形式化） | [02-project-management/README.md](./02-project-management/README.md)、[NAVIGATION_GUIDE.md](./NAVIGATION_GUIDE.md)；以图表与案例为主 |
| Week 2 | 基础数学模型（简化版） | [01-foundations/mathematical-models.md](./01-foundations/mathematical-models.md) 中集合、图、概率的直观部分；最少公式、多图示 |
| Week 3 | 简单应用实例 | [04-industry-applications/software-development/agile-models.md](./04-industry-applications/software-development/agile-models.md)、[lifecycle-models.md](./02-project-management/lifecycle-models.md) 的「直观解释」小节 |
| Week 4 | 综合案例与自测 | [03-retrieval-practice-questions.md](./12-learning-support/03-retrieval-practice-questions.md) 入门题；[02-spaced-repetition-schedule.md](./12-learning-support/02-spaced-repetition-schedule.md) 安排复习 |

**建议**：先读 [项目完成总览](./PROJECT_COMPLETION_OVERVIEW.md) 与 [NAVIGATION_GUIDE.md](./NAVIGATION_GUIDE.md)，再按上表顺序；遇到形式化定义可先跳过，重点看「直观解释」与「应用解释」。

---

## 轨道 B：进阶轨道（Advanced Track）

**目标**：深入理论，掌握形式化方法与核心模型；面向胜任者到精通者。
**时长**：约 6–8 周。

| 阶段 | 内容 | 文档与资源 |
|------|------|------------|
| 1–2 周 | 形式化基础 | [01-foundations/README.md](./01-foundations/README.md)、[mathematical-models.md](./01-foundations/mathematical-models.md)、[semantic-models.md](./01-foundations/semantic-models.md) |
| 2–3 周 | 核心模型层（CML） | [lifecycle-models.md](./02-project-management/lifecycle-models.md)、[resource-models.md](./02-project-management/resource-models.md)、[risk-models.md](./02-project-management/risk-models.md)、[quality-models.md](./02-project-management/quality-models.md) |
| 2–3 周 | 形式化验证（VL） | [verification-theory.md](./03-formal-verification/verification-theory.md)、[model-checking.md](./03-formal-verification/model-checking.md)、[theorem-proving.md](./03-formal-verification/theorem-proving.md) |
| 1–2 周 | 应用与交错练习 | [04-industry-applications/README.md](./04-industry-applications/README.md) 选读；[05-interleaved-learning-paths.md](./12-learning-support/05-interleaved-learning-paths.md) |

**建议**：按 01 → 02 → 03 → 04 顺序；结合 [02-spaced-repetition-schedule.md](./12-learning-support/02-spaced-repetition-schedule.md) 安排复习，用 [03-retrieval-practice-questions.md](./12-learning-support/03-retrieval-practice-questions.md) 做检索练习。

---

## 轨道 C：专家轨道（Expert Track）

**目标**：理论创新、跨领域整合、形式化验证实现；面向精通者到专家。
**时长**：持续学习。

| 模块 | 内容 | 文档与资源 |
|------|------|------------|
| 基础与验证 | 形式化基础 + 验证理论 + 定理证明 | [01-foundations/](./01-foundations/)、[03-formal-verification/](./03-formal-verification/) 全文；[05-implementations/](./05-implementations/)（Rust/Haskell/Lean） |
| 前沿理论 | 量子、生物启发、全息、星际等 | [quantum-project-theory.md](./01-foundations/quantum-project-theory.md)、[bio-inspired-project-theory.md](./01-foundations/bio-inspired-project-theory.md)、[holographic-project-theory.md](./01-foundations/holographic-project-theory.md)、[interstellar-project-theory.md](./01-foundations/interstellar-project-theory.md) |
| 复杂性与系统 | Cynefin、系统动力学、CAS | [13-complexity-systems/](./13-complexity-systems/) |
| 实现与 CI 验证 | 自动化验证系统、一致性检查 | [06-ci-verification/](./06-ci-verification/) |

**建议**：可直接从 [01-foundations/](./01-foundations/) 与 [03-formal-verification/](./03-formal-verification/) 切入；参与标准与课程对标见 [README.md](./README.md) 中大学课程对标表与 [STANDARDS_ALIGNMENT.md](./STANDARDS_ALIGNMENT.md)。

---

## 按角色选择路径

| 角色 | 推荐轨道 | 首站文档 |
|------|----------|----------|
| 项目管理新手 | 轨道 A | [NAVIGATION_GUIDE.md](./NAVIGATION_GUIDE.md) → [02-project-management/README.md](./02-project-management/README.md) |
| 有 PM 经验、想学形式化 | 轨道 B | [01-learning-prerequisites.md](./12-learning-support/01-learning-prerequisites.md) → [01-foundations/README.md](./01-foundations/README.md) |
| 研究者 / 形式化方法专家 | 轨道 C | [01-foundations/](./01-foundations/) → [03-formal-verification/](./03-formal-verification/) |
| 软件开发 / 工程管理从业者 | 轨道 A 或 B | [04-industry-applications/software-development/](./04-industry-applications/software-development/) 或 [engineering-management/](./04-industry-applications/engineering-management/) |

---

## 权威来源对齐

本知识体系与以下权威来源对齐，学习时可结合查阅以确认覆盖与最新表述：

- **标准**：[STANDARDS_ALIGNMENT.md](./STANDARDS_ALIGNMENT.md)（ISO 21500:2021、ISO 21502:2020、PMBOK 7th/8th、ISO 21520 等）、[PMBOK_8_ALIGNMENT_PLAN.md](./PMBOK_8_ALIGNMENT_PLAN.md)（8th 原则、绩效域、流程映射）。
- **大学课程**：[README.md](./README.md) 中「大学课程对标表」（形式化方法：Stanford、CMU、Oxford、Cambridge 等；项目管理：MIT ESD.36、CMU 17-632）；年度审查见 README 与 [SUSTAINABLE_EXECUTION_PLAN.md](./SUSTAINABLE_EXECUTION_PLAN.md)。

按科学认知规律梳理的内容（三层解释、间隔重复、检索练习、难度–间隔）与上述权威覆盖一致，便于系统学习与考证准备。

---

## 与学习支持模块的衔接

- **先备知识**：[01-learning-prerequisites.md](./12-learning-support/01-learning-prerequisites.md) — 各层所需前置知识。
- **难度分级**：[04-concept-difficulty-ranking.md](./12-learning-support/04-concept-difficulty-ranking.md) — 概念难度 1–5，用于安排学习顺序与复习强度。
- **间隔重复**：[02-spaced-repetition-schedule.md](./12-learning-support/02-spaced-repetition-schedule.md) — 复习间隔与难度–间隔映射。
- **检索练习**：[03-retrieval-practice-questions.md](./12-learning-support/03-retrieval-practice-questions.md) — 自测与巩固。
- **交错学习**：[05-interleaved-learning-paths.md](./12-learning-support/05-interleaved-learning-paths.md) — 相似概念混合练习（如 LTL vs CTL、生命周期模型对比）。

---

**Last Updated**: 2026-02-04
**Status**: 初版；与 12-learning-support 及 NAVIGATION_GUIDE 同步维护
