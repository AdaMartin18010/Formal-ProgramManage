# PMBOK 8th Edition 对齐计划

## 概述

PMBOK Guide 第 8 版于 2025 年 11 月发布。本文档定义项目与 PMBOK 8 的对齐方案，并指导核心模型层（CML）文档的逐步更新。项目当前以 PMBOK 7th 为主参考；本计划用于向 8th 迁移时的原则、绩效域与流程映射。

**参考**：[STANDARDS_ALIGNMENT.md](./STANDARDS_ALIGNMENT.md) — 标准版本与引用约定。

---

## 1. PMBOK 8th 核心变化摘要

### 1.1 六项核心原则（由 7th 的 12 条简化）

**说明**：六原则表述以 **PMI 官方 PMBOK Guide 8th Edition 正文为准**。不同来源曾出现差异（如“Embed Quality”与“Focus on Value”的细分、“Integrate Sustainability”等）；正式引用时请核对 8th 正式版。详见 [STANDARDS_ALIGNMENT.md](./STANDARDS_ALIGNMENT.md)。

| # | PMBOK 8 核心原则（英文，以官方 8th 为准） | 中文 | 与 CML 的映射 |
|---|------------------------------------------|------|----------------|
| 1 | Adopt a Holistic View | 采用整体观 | 生命周期、范围、干系人、系统思维 |
| 2 | Focus on Value | 聚焦价值 | 质量管理、范围、成果导向 |
| 3 | Embed Quality into Processes and Deliverables | 将质量嵌入过程与可交付成果 | 质量规划/保证/控制、Governance/Scope/Resources |
| 4 | Be an Accountable Leader | 做负责任的领导者 | 治理、资源、风险、阶段责任 |
| 5 | Integrate Sustainability | 整合可持续性 | 生命周期、资源、治理、价值 |
| 6 | Build an Empowered Culture | 建立赋能文化 | 资源（团队）、干系人、治理 |

**与 7th 及部分解读的对应**：原 7th 的“Embrace Adaptability”“Demonstrate Accountability”“Manage Uncertainty”等思想在 8th 中并入上述六原则及七绩效域（如 Risk、Governance）；CML 映射时仍可引用这些主题，但原则列表以本表为准。

### 1.2 七个绩效域（Performance Domains）

PMBOK 8 将绩效域从 7th 的 8 个调整为 7 个；质量、沟通、采购不再作为独立绩效域，其内容并入其他域。

| 绩效域 | 英文 | 本项目 CML 对应文档 | 说明 |
|--------|------|---------------------|------|
| 1. Governance | 治理 | 生命周期、资源、质量 | 决策、角色、合规 |
| 2. Scope | 范围 | [lifecycle-models.md](./02-project-management/lifecycle-models.md) | 可交付成果、WBS、变更 |
| 3. Schedule | 进度 | [lifecycle-models.md](./02-project-management/lifecycle-models.md)、[resource-models.md](./02-project-management/resource-models.md) | 活动、顺序、工期、关键路径 |
| 4. Finance | 财务 | [resource-models.md](./02-project-management/resource-models.md) | 预算、成本、收益 |
| 5. Stakeholders | 干系人 | 生命周期、范围、治理 | 识别、参与、沟通（并入） |
| 6. Resources | 资源 | [resource-models.md](./02-project-management/resource-models.md) | 人力、物、设备、团队 |
| 7. Risk | 风险 | [risk-models.md](./02-project-management/risk-models.md) | 识别、分析、应对、监控 |

**质量**：并入 Governance、Scope、Resources（交付质量、过程质量）。对应 [quality-models.md](./02-project-management/quality-models.md)。
**沟通**：并入 Stakeholders、Governance。
**采购**：PMBOK 8 扩展采购主题；可放在 Resources 或单独小节，待 8th 正文确认后细化。

### 1.3 流程结构回归（约 40 个流程）

PMBOK 8 重新引入流程（Processes），分布在传统生命周期阶段中。CML 文档更新时需在相应章节补充或引用这些流程，而不是仅写原则与绩效域。

**建议映射方式**：

- **启动**：对应 [lifecycle-models.md](./02-project-management/lifecycle-models.md) 的「启动」阶段；补充 8th 启动流程列表与简要描述。
- **规划**：对应生命周期「规划」、[resource-models.md](./02-project-management/resource-models.md)（资源/成本规划）、[risk-models.md](./02-project-management/risk-models.md)（风险规划）、[quality-models.md](./02-project-management/quality-models.md)（质量规划）；补充 8th 规划流程列表。
- **执行**：对应生命周期「执行」与资源/质量执行活动；补充 8th 执行流程列表。
- **监控**：对应生命周期「监控」、风险监控、质量控制；补充 8th 监控流程列表。
- **收尾**：对应生命周期「收尾」；补充 8th 收尾流程列表。

具体流程名称与编号以 PMBOK 8 正式版为准；本文档仅规定「在 CML 各文档中预留流程级描述位置并标注 PMBOK 8」。

#### 1.3.1 PMBOK 8 流程列表（占位）

以下表格为占位，待 PMBOK 8 正式版到手后填齐流程名称与编号。

| 阶段 | 流程数（约） | 与本项目 CML 对应 | 备注 |
|------|--------------|-------------------|------|
| 启动 Initiating | 待填 | [lifecycle-models.md](./02-project-management/lifecycle-models.md) 启动阶段 | 正式版发布后补流程名称 |
| 规划 Planning | 待填 | lifecycle、[resource-models.md](./02-project-management/resource-models.md)、[risk-models.md](./02-project-management/risk-models.md)、[quality-models.md](./02-project-management/quality-models.md) | 同上 |
| 执行 Executing | 待填 | lifecycle、resource、quality | 同上 |
| 监控 Monitoring and Controlling | 待填 | lifecycle、resource、risk、quality | 同上 |
| 收尾 Closing | 待填 | lifecycle 收尾阶段 | 同上 |

**合计**：约 40 个流程（以 PMBOK 8 正式版为准）。CML 四文档中均已引用本表或本节，正式版发布后同步更新各文档中的流程列表。

### 1.4 现代主题扩展（PMBOK 8）

- **AI 在项目管理中的应用**：在 [04-industry-applications/ai-management/ai-management.md](./04-industry-applications/ai-management/ai-management.md) 中扩展，并与 [STANDARDS_ALIGNMENT.md](./STANDARDS_ALIGNMENT.md) 及 ISO 21520 对齐。
- **PMO（项目管理办公室）**：在生命周期或治理相关小节中增加 PMO 角色与职能简述，并标注 PMBOK 8。
- **采购管理现代化**：在资源或单独小节中补充采购主题，标注 PMBOK 8。

---

## 2. CML 文档更新清单

以下四个文档需逐步加入 PMBOK 8 对齐内容（保留 PMBOK 7 的既有内容，增加 8th 对照）。

| 文档 | 更新内容 | 优先级 |
|------|----------|--------|
| [02-project-management/lifecycle-models.md](./02-project-management/lifecycle-models.md) | 增加「PMBOK 8th 对标」小节：7 绩效域中的 Scope、Schedule、Governance、Stakeholders 与生命周期阶段；六原则中的 Holistic View、Adaptability；流程回归说明（启动/规划/执行/监控/收尾） | P0 |
| [02-project-management/resource-models.md](./02-project-management/resource-models.md) | 增加「PMBOK 8th 对标」：Resources、Finance、Governance 绩效域；原则 Empowered Culture、Accountability；资源与成本相关流程 | P0 |
| [02-project-management/risk-models.md](./02-project-management/risk-models.md) | 增加「PMBOK 8th 对标」：Risk 绩效域；原则 Manage Uncertainty；风险相关流程 | P0 |
| [02-project-management/quality-models.md](./02-project-management/quality-models.md) | 增加「PMBOK 8th 对标」：质量并入 Governance/Scope/Resources 的说明；原则 Focus on Value；质量相关流程（若有） | P0 |

**统一约定**：

- 在每个文档的「国际标准对标」或「标准对标」部分增加子节「PMBOK 8th Edition（2025）」。
- 引用格式：PMBOK Guide 8th Edition (2025)，并链到 [STANDARDS_ALIGNMENT.md](./STANDARDS_ALIGNMENT.md)。
- 若 8th 正式版与本文摘要有差异，以正式版为准并更新本计划。

---

## 3. 原则与绩效域映射表（供 CML 引用）

### 3.1 六原则 → CML 文档

| 原则 | lifecycle-models | resource-models | risk-models | quality-models |
|------|------------------|-----------------|-------------|----------------|
| Adopt a Holistic View | ✓ 阶段与系统观 | ✓ 资源整体观 | ✓ 风险全景 | ✓ 质量与价值 |
| Focus on Value | ✓ 可交付成果 | ✓ 成本效益 | — | ✓ 核心 |
| Embed Quality into Processes and Deliverables | ✓ 阶段质量 | ✓ 过程质量 | — | ✓ 核心 |
| Be an Accountable Leader | ✓ 阶段责任 | ✓ 资源责任 | ✓ 风险责任 | ✓ 质量责任 |
| Integrate Sustainability | ✓ 可持续生命周期 | ✓ 资源与可持续 | ✓ 风险与可持续 | ✓ 质量与价值 |
| Build an Empowered Culture | ✓ 团队与阶段 | ✓ 核心 | — | ✓ 持续改进 |

### 3.2 七绩效域 → CML 文档

| 绩效域 | lifecycle-models | resource-models | risk-models | quality-models |
|--------|------------------|-----------------|-------------|----------------|
| Governance | ✓ 阶段与治理 | ✓ 资源治理 | ✓ 风险治理 | ✓ 质量治理 |
| Scope | ✓ 核心 | — | — | ✓ 范围与质量 |
| Schedule | ✓ 核心 | ✓ 资源与进度 | — | — |
| Finance | — | ✓ 核心 | — | — |
| Stakeholders | ✓ 阶段与干系人 | ✓ 团队 | — | — |
| Resources | ✓ 阶段资源 | ✓ 核心 | — | ✓ 质量资源 |
| Risk | ✓ 阶段风险 | — | ✓ 核心 | ✓ 质量风险 |

---

## 4. 后续步骤与审查

1. **立即**：在 CML 四文档中增加「PMBOK 8th Edition（2025）对标」小节，内容包含：对应绩效域、对应原则、流程回归说明（列表或引用本计划）。
2. **PMBOK 8 正式版到手后**：核对六原则与七绩效域表述、约 40 个流程的名称与编号，更新本计划 1.1–1.3 及 CML 正文。
3. **季度审查**：在 [STANDARDS_ALIGNMENT.md](./STANDARDS_ALIGNMENT.md) 的审查条款中保留「PMBOK 8th 修订与项目引用更新」。

---

**Last Updated**: 2026-02-04
**Status**: 对齐计划已建立；CML 文档更新进行中
**维护**：与 STANDARDS_ALIGNMENT.md 同步维护
