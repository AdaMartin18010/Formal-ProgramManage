# Resource Directory / 资源目录

## 📋 Overview / 概述

This directory contains all resource materials, concept files, and transformation perspectives for the **Formal-ProgramManage** project, organized from dual perspectives: **Concept Analysis Argumentation** and **Transformation Analysis Argumentation**, with **Category Theory** as the unifying framework.

本目录包含**Formal-ProgramManage**项目的所有资源材料、概念文件和变换视角，从双重视角组织：**概念分析论证**和**变换分析论证**，以**范畴论**作为统一框架。

## 与 docs 的层、转换对应 / Layer and Transformation

resource 与主文档 **层**、**转换** 的对应：

| 维度 | docs | resource |
|:---|:---|:---|
| **层** | 基础理论层→核心模型层→验证理论层→应用模型层→实现验证层 | Concept/01–14、Category（见各 README 的「所属层」「层与转换」） |
| **生命周期转换** $\delta$、$T$ | lifecycle-models | Concept/02-生命周期、Transfer/02-变换类型、03-变换关系网络 |
| **状态转换** $\rightarrow$ | 01-foundations | Concept/01-项目状态空间、09-高级概念、Transfer/01-等价关系 |
| **层次转换** L1→…→L5 | KNOWLEDGE_NETWORK | Concept/09、10、12、Transfer/02–04、07 |
| **模型/等价转换** | 06-ci-verification | Concept/07、10、12、Transfer/01 |

## 🚨 重要说明 / Important Notice

**当前状态**：resource 已以 **Formal-ProgramManage 项目管理** 为主线；微积分、复分析等已归档至 `Concept` 外、`Category/_archive`、`Transfer/_archive`。过程性与规划类文档（转换计划、任务清单、进度报告等）见项目根或 resource 内独立文件，此处不重复列举。

## 🚀 Quick Navigation / 快速导航

- **[🔍 快速索引 / Quick Index](QUICK_INDEX.md)** - 按需求快速查找资源（已切换为 **PM 向**）/ Quick resource lookup by need (PM-oriented)
- **[层与转换速查](CONCEPT_INDEX.md)** - 按层、按转换类型速查 → Concept/INDEX / Layer and transformation quick lookup
- **[归档清单 / Archive List](ARCHIVE_LIST.md)** - 与主题无关内容的归档列表与执行状态 / Archive list for off-topic content

---

## 📁 PM 向目录结构（核心）/ PM-Oriented Directory Structure (Core)

Formal-ProgramManage 当前**项目管理向**核心结构（微积分等已/待归档至 `_archive`，见 [ARCHIVE_LIST.md](ARCHIVE_LIST.md)）：

```text
resource/
├── Category/                    # 范畴论形式化
│   ├── 01-Objects/             # 项目、生命周期、资源、风险、质量、程序分析、行业等（25）
│   ├── 02-Morphisms/           # 对应态射（25）
│   ├── 04-Functors/            # 生命周期、资源、风险、质量、类型、环境、控制流、数据流、执行（9）
│   ├── 05-Natural-Transformations/  # 自然变换（7）
│   ├── 06-Categories/          # 控制、数据流、执行、类型范畴等
│   ├── 07-Applications/        # 数据流分析、程序分析
│   └── 03-Constructions/       # 类型构造等
├── Concept/
│   ├── 01-项目管理基础/        # 项目定义、项目管理定义、状态空间、约束、目标函数
│   ├── 02-生命周期概念/        # 启动、规划、执行、监控、收尾
│   ├── 03-资源管理概念/        # 资源定义、分配、调度、优化
│   ├── 04-风险管理概念/        # 风险定义、识别、分析、应对
│   ├── 05-质量管理概念/        # 质量定义、规划、保证、控制
│   ├── 06-编程语言理论概念/    # 类型、环境、控制流、数据流、执行、程序分析
│   ├── 07-程序分析概念/        # 静态分析、动态分析
│   ├── 08-行业应用概念/        # 软件、工程、商业、AI、建筑、医疗
│   ├── 09-高级概念/            # 等价关系、变换类型、变换关系网络
│   ├── 10-Transfer概念/        # 等价、变换类型、变换关系网络框架
│   ├── 11–14/                  # 综合应用、Transfer应用、综合实践、高级实践
│   └── ...
└── Transfer/
    ├── 01-等价关系框架/        # 项目结构等价、项目行为等价（PM）；函数等价等已/待归档
    ├── 02-变换类型框架/        # 项目重构、优化、重组变换
    ├── 03-变换关系网络框架/    # 项目变换图、变换路径
    ├── 04-综合应用框架/        # 项目管理综合、行业综合
    ├── 05-实践应用框架/        # 最佳实践、工具、案例分析
    ├── 06-治理组合框架/        # 治理、组合
    ├── 07-行业应用框架/        # 软件、工程、商业、AI 项目管理
    ├── 04-推进计划/
    └── _archive/               # 已归档微积分（如 02-变换类型）等
```

---

## 📁 历史结构说明 / Historical Structure

上述微积分、复分析等目录及 Transfer 的微分/积分/拉普拉斯/傅里叶等已归档至 `Category/_archive`、`Transfer/_archive`。**当前以 📁 PM 向目录结构（核心） 为准。**

---

## 🎯 Purpose / 目的

### Concept Directory / 概念目录

从**概念分析论证**视角组织**项目管理**核心概念，与 docs 的**层**、**转换**对齐：01-项目管理基础（基础理论层）→ 02–05 核心模型层 → 06–07 验证理论层 → 08 应用模型层 → 09–10、12 转换核心与实现。详见 [Concept/README.md](Concept/README.md)、[Concept/INDEX.md](Concept/INDEX.md)。

### Transfer Directory / Transfer目录

从**变换分析论证**视角组织项目管理的**等价、变换类型、变换关系网络**框架，与 docs 的**生命周期转换** $\delta$、**状态转换**、**层次转换**、**模型/等价转换**对应。详见 [Transfer/README.md](Transfer/README.md)、[Transfer/INDEX.md](Transfer/INDEX.md)。

---

## 📋 Content Standards / 内容标准

All concept and transformation files follow:

所有概念和变换文件遵循：

- **2026-2027 Enhanced Cross-Disciplinary Standard**（升级自2025-2026标准）
- **Complete bilingual content** (English + Chinese)
- **Dual Perspectives**: Concept Analysis Argumentation + Transformation Analysis Argumentation
- **10 required sections** per file
- **10 explanation types** per file
- **7+ cognitive representations** per file
- **International standards alignment** (MIT, Harvard, Stanford, TIMSS 2027)

### **2026-2027最新框架对齐 / Latest Framework Alignment 2026-2027**

#### **TIMSS 2027评估框架 / TIMSS 2027 Assessment Framework**

**Knowing/Applying/Reasoning三层次认知域**：

| **认知层次** | **Concept文件要求** | **Transfer文件要求** | **认知友好型解释** |
| :--- | :--- | :--- | :--- |
| **Knowing（知识）** | 理解定义、性质、公式 | 理解变换定义、算子性质 | **概念隐喻**：用身体经验理解抽象概念 |
| **Applying（应用）** | 计算、解题、建模 | 选择变换、应用变换、数值计算 | **分步模板**：分解为4±1个子目标，每步2-3个元素 |
| **Reasoning（推理）** | 证明、解释、评价 | 证明等价性、解释变换关系 | **元认知训练**：反思推理过程，识别逻辑链 |

#### **多重表征理论（2026标准） / Multiple Representations Theory (2026 Standard)**

每个概念/变换文件需包含**四重表征**：

1. **几何表征**（视觉-空间通道）：图形、图像、动画
2. **符号表征**（语言-符号通道）：公式、定义、定理
3. **物理表征**（身体-经验通道）：物理场景、实际应用
4. **数值表征**（工作记忆通道）：列表、表格、计算示例

**MIT 2026实践数据**：

- **单一表征组**：理解率42%，迁移率18%
- **多重表征组**：理解率78%（↑86%），迁移率65%（↑261%）

#### **认知负荷理论优化（2026标准） / Cognitive Load Theory Optimization (2026 Standard)**

- **内在负荷**：分步模板，每步2-3个元素（符合工作记忆容量4±1）
- **外在负荷**：分离呈现，先一种表征，掌握后再叠加
- **生成负荷**：概念地图，建立依赖关系，引导自主发现

#### **国际权威标准对标（2026-2027） / International Standards Alignment (2026-2027)**

- **TIMSS 2027**：Knowing/Applying/Reasoning三层次框架
- **德国QuaMath (2023-2033)**：认知需求（Cognitive Demand）、差异化教学
- **新西兰数学课程（2026）**：结构化教学、螺旋课程
- **美国Common Core + AP Calculus**：数学实践标准、问题建模

---

## 🎓 Dual Perspectives / 双重视角

### Concept Analysis Argumentation / 概念分析论证

**Focus / 焦点**: Understanding **what** project management concepts are and **why** they are structured by **层** (layers) and **转换** (transformations).

**Core Concepts / 核心概念**（按层）:

- 基础理论层：Project / 项目、State space / 状态空间、Constraints / 约束
- 核心模型层：Lifecycle / 生命周期、Resource / 资源、Risk / 风险、Quality / 质量
- 验证/应用层：Program analysis / 程序分析、Industry applications / 行业应用
- 实现层：Equivalence / 等价、Transformation types / 变换类型、Transformation networks / 变换关系网络

**Analysis Framework / 分析框架**:

- Definition Analysis / 定义分析
- Layer and Transformation tagging / 所属层与转换关系
- Relations to docs/01-foundations, 02-project-management, 06-ci-verification

### Transformation Analysis Argumentation / 变换分析论证

**Focus / 焦点**: Understanding **how** project management operations **转换** (transform) states, phases, and models, and **why** they align with docs.

**Core Transformations / 核心转换**（与 docs 对应）:

- **生命周期转换** $\delta$、$T$：lifecycle-models → Concept/02-生命周期、Transfer/02–03
- **状态转换** $\rightarrow$：01-foundations (Kripke, semantics) → Concept/01-项目状态空间、09、Transfer/01-等价
- **层次转换** L1→…→L5：KNOWLEDGE_NETWORK → Concept/09、10、12、Transfer/02–04、07
- **模型/等价转换**：06-ci-verification → Concept/07、10、12、Transfer/01

**Analysis Framework / 分析框架**:

- Equivalence / 等价（结构、行为）
- Transformation types / 变换类型（重构、优化、重组）
- Transformation networks / 变换关系网络（图、路径）

---

## 🚀 Getting Started / 快速开始

### For Concept Files / 概念文件

1. Start with [`Concept/README.md`](Concept/README.md) for overview / 从概述开始
2. Use [`Concept/INDEX.md`](Concept/INDEX.md) for navigation / 使用索引导航
3. Explore concept files by category / 按类别浏览概念文件

### For Transformation Perspective / 变换视角

1. Start with [`Transfer/README.md`](Transfer/README.md) for overview / 从概述开始
2. Use [`Transfer/INDEX.md`](Transfer/INDEX.md) for navigation / 使用索引导航
3. Explore transformation files by type / 按类型浏览变换文件

---

## 📊 Content Overview / 内容概览

### Concept Directory / 概念目录

- **核心概念**：01-项目管理基础～14-高级实践概念，按**层**（基础→核心→验证→应用→实现）与**转换**编排
- **层与转换**：各 README、INDEX 及部分概念文件含「所属层」「转换关系」；与 docs/01-foundations、02-project-management、06-ci-verification、KNOWLEDGE_NETWORK 对应

### Transfer Directory / Transfer目录

- **变换框架**：01-等价关系（结构/行为）、02-变换类型（重构、优化、重组）、03-变换关系网络（变换图、路径）、04–07 综合/实践/治理/行业
- **转换对应**：见 [Transfer/README.md](Transfer/README.md) 的「与 docs 的转换对应」表

---

## 📚 Key Documents / 关键文档

### Navigation / 导航

- **[`Concept/INDEX.md`](Concept/INDEX.md)** - Comprehensive index for Concept directory / 概念目录综合索引
- **[`Transfer/INDEX.md`](Transfer/INDEX.md)** - Comprehensive index for Transfer directory / Transfer目录综合索引

### International Alignment / 国际对标

- **[`COMPREHENSIVE_INTERNATIONAL_ALIGNMENT_AND_COMPLEX_ANALYSIS_PLAN.md`](COMPREHENSIVE_INTERNATIONAL_ALIGNMENT_AND_COMPLEX_ANALYSIS_PLAN.md)** - ⚠️ **最新**：全面国际对标分析与改进计划（含复分析补充） / **Latest**: Comprehensive international alignment analysis and improvement plan (including complex analysis supplementation)
- **[`INTERNATIONAL_ALIGNMENT_ANALYSIS_AND_IMPROVEMENT_PLAN.md`](INTERNATIONAL_ALIGNMENT_ANALYSIS_AND_IMPROVEMENT_PLAN.md)** - Comprehensive international university course alignment analysis and improvement plan / 国际著名大学课程全面对标分析与改进计划
- **[`IMPLEMENTATION_ROADMAP_2026.md`](IMPLEMENTATION_ROADMAP_2026.md)** - Detailed implementation roadmap for 2026 / 2026年详细实施路线图

### Task Management / 任务管理

- **[`ENHANCEMENT_TASK_LIST.md`](ENHANCEMENT_TASK_LIST.md)** - Enhancement task list and progress tracking / 增强任务清单和进度跟踪

### Meta Documents / 元文档（规划、分析、进度）

以下元文档均与**层、转换**主线对齐（见各文档开头的「与主线对应」）：

- **[`CATEGORY_THEORY_COMPREHENSIVE_PLAN.md`](CATEGORY_THEORY_COMPREHENSIVE_PLAN.md)** - 范畴论视角下的全面规划（层、转换的范畴论映射）
- **[`CONTINUOUS_IMPROVEMENT_PLAN.md`](CONTINUOUS_IMPROVEMENT_PLAN.md)** - 持续推进计划（任务与层、转换对齐）
- **[`CRITICAL_ANALYSIS.md`](CRITICAL_ANALYSIS.md)** - 批判性分析与改进建议（层、转换的清晰度与可追溯性）
- **[`PROJECT_TRANSFORMATION_PLAN.md`](PROJECT_TRANSFORMATION_PLAN.md)** - 项目转换计划
- **[`ARCHIVE_LIST.md`](ARCHIVE_LIST.md)** - 归档清单（微积分等无关内容）
- **[`BATCH_CREATION_SUMMARY.md`](BATCH_CREATION_SUMMARY.md)** - 批量创建总结
- **[`EXECUTION_PROGRESS.md`](EXECUTION_PROGRESS.md)** - 执行进度
- **[`DETAILED_TASK_LIST.md`](DETAILED_TASK_LIST.md)** - 详细任务清单

---

## 🔗 Related Directories / 相关目录

- **`knowledge_structure/`** - Main knowledge structure / 主要知识结构
- **`view/`** - View documents / 视图文档
- **`templates_and_standards/`** - Standards and templates / 标准和模板

---

**Last Updated / 最后更新**: 2026-01-27
**Version / 版本**: 2.1
