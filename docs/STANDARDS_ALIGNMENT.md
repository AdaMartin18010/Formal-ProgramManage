# Standards Alignment / 标准对齐说明

## 概述

本文档列出 Formal-ProgramManage 项目所采用与参考的国际项目管理及形式化方法标准版本，便于引用一致性与后续更新审查。

## 项目管理标准

### ISO 项目、项目群与项目组合管理标准族

| 标准 | 版本 | 状态 | 项目采用 | 说明 |
|------|------|------|----------|------|
| **ISO 21500** | 2021 | 现行 | **采用** | Project, programme and portfolio management — Context and concepts. 已取代 ISO 21500:2012（2012 已撤销）。提供组织与外部环境背景、战略实施及综合治理概念。 |
| **ISO 21502** | 2020 | 现行 | **采用** | Project management — Guidance on project management. 项目管理的详细指导，替代原 ISO 21500:2012 中的操作性内容；支持预测、增量、迭代、适应、混合与敏捷等交付方式。 |
| **ISO 21503** | 2022 | 现行 | 参考 | Programme management — Guidance on programme management. |
| **ISO 21504** | 2022 | 现行 | 参考 | Project, programme and portfolio management — Guidance on portfolio management. |
| **ISO 21506** | 2024 | 现行 | 参考 | Vocabulary. |
| **ISO 21512** | 2024 | 现行 | 参考 | Earned value management implementation. |

**参考文献书写示例：**

- ISO 21500:2021. *Project, programme and portfolio management — Context and concepts*. International Organization for Standardization.
- ISO 21502:2020. *Project management — Guidance on project management*. International Organization for Standardization.

### PMI PMBOK Guide

| 版本 | 发布 | 状态 | 项目采用 | 说明 |
|------|------|------|----------|------|
| **PMBOK Guide 7th** | 2021 | 现行 | **采用（主参考）** | 原则与绩效域导向，12 条原则、8 个绩效域，支持预测、敏捷与混合。 |
| **PMBOK Guide 8th** | 2025年11月 | 最新 | **待同步** | 在 7th 基础上简化与澄清，更强调可操作性；六项核心原则、七个绩效域；扩展 AI、PMO、采购等。正式发布后建议更新本表及正文引用。 |

**参考文献书写示例：**

- Project Management Institute. (2021). *A guide to the project management body of knowledge (PMBOK guide)* (7th ed.). Project Management Institute.
- Project Management Institute. (2025). *A guide to the project management body of knowledge (PMBOK guide)* (8th ed.). Project Management Institute. （8th 发布后使用）

### ISO 21520（AI 在 P3M 中）（制定中）

| 标准 | 版本/阶段 | 状态 | 项目采用 | 说明 |
|------|------------|------|----------|------|
| **ISO 21520** | CD（Committee Draft） | 制定中 | **跟踪** | Project, programme and portfolio management — Artificial intelligence — Concepts, applications, and implications. 涵盖 AI 概念、应用、收益、风险、治理、伦理、数据治理等。2025 年 12 月进入 CD 咨询阶段（CD consultation）。正式发布后将在 [04-industry-applications/ai-management/ai-management.md](./04-industry-applications/ai-management/ai-management.md) 及本表中更新。 |

### 其他项目管理与风险标准

- **ISO 31000:2018** — 风险管理 — 指南。项目采用。
- **PRINCE2 2017** — 项目管理方法。参考。
- **CMMI-DEV** — 能力成熟度模型集成（开发）。参考。
- **ISO/IEC 25010** — 系统与软件质量模型。参考。

## 形式化方法与软件工程标准

- **IEEE Std 830** — 软件需求规格说明实践。参考。
- **ISO/IEC 15504** — 过程评估。参考。

## 文档中引用约定

1. **项目管理语境**：正文与参考文献优先使用 **ISO 21500:2021** 与 **ISO 21502:2020**；若需说明历史，可注明“ISO 21500:2012 已由 ISO 21500:2021 取代，详细指导见 ISO 21502:2020”。
2. **PMBOK**：当前以 **PMBOK 7th Edition** 为主要引用；PMBOK 8th 发布后，在新撰或修订内容中逐步采用 8th，并在本表中更新“项目采用”列。
3. **格式规范**：具体引用格式见 [REFERENCE_FORMAT_GUIDE.md](./REFERENCE_FORMAT_GUIDE.md)。

## 审查与更新

- **季度**：检查 ISO、PMI 官网是否有新版本或修订；跟踪 **ISO 21520**（AI in P3M）的 CD/DIS/FDIS 发布进度；核对 **ISO 21503**（项目群）、**ISO 21506**（词汇）、**ISO 21512**（EVM）的现行状态。
- **触发更新**：PMBOK 8th 正式发布、ISO 21500/21502 修订发布、**ISO 21520 正式发布**时，更新本文档及全库中相关引用。

---

**Last Updated**: 2026-02-04
**Status**: 现行
