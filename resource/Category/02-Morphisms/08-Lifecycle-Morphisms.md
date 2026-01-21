# Lifecycle Morphisms / 生命周期态射

## 📋 Table of Contents / 目录

- [Lifecycle Morphisms / 生命周期态射](#lifecycle-morphisms--生命周期态射)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Phase Transition Morphisms / 阶段转换态射](#21-phase-transition-morphisms--阶段转换态射)
    - [2.2 Morphism Composition / 态射复合](#22-morphism-composition--态射复合)
    - [2.3 Identity Morphism / 恒等态射](#23-identity-morphism--恒等态射)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义](#31-pmbok-7th-edition-definition--pmbok-第7版定义)
    - [3.2 ISO 21500 Standard Definition / ISO 21500 标准定义](#32-iso-21500-standard-definition--iso-21500-标准定义)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Basic Properties / 基本性质](#41-basic-properties--基本性质)
    - [4.2 Category-Theoretic Properties / 范畴论性质](#42-category-theoretic-properties--范畴论性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Morphisms / 与其他态射的关系](#51-relations-to-other-morphisms--与其他态射的关系)
    - [5.2 Natural Transformations / 自然变换](#52-natural-transformations--自然变换)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Standard Phase Transition / 标准阶段转换](#61-standard-phase-transition--标准阶段转换)
    - [6.2 Agile Sprint Transition / 敏捷冲刺转换](#62-agile-sprint-transition--敏捷冲刺转换)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Project Management Applications / 项目管理应用](#71-project-management-applications--项目管理应用)
    - [7.2 Category-Theoretic Applications / 范畴论应用](#72-category-theoretic-applications--范畴论应用)
    - [7.3 直观解释与一例 / Intuitive Explanation with One Example](#73-直观解释与一例--intuitive-explanation-with-one-example)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Standards / 标准](#81-standards--标准)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**核心模型层**（对应 docs/02-project-management/lifecycle-models；生命周期转换 $\delta$）
- **转换关系**：**Lifecycle Morphisms** = **生命周期转换** $\delta$（阶段转换、里程碑达成、交付物完成）；与 02-生命周期概念、Category/01-Objects/08-Lifecycle-Objects、Category/04-Functors/01-Lifecycle-Functor 对应。

**与 docs/02-project-management/lifecycle-models 的公式对应**：

- **生命周期转换** $\mathrm{transition}: P \times E \to P$（定义 2.1.3）→ 态射 $\tau: Ph_i \to Ph_j$；$Ph_i,Ph_j \in P$，$E \supseteq \{\mathrm{phase\_complete},\mathrm{gate\_approved},\mathrm{change\_requested},\mathrm{risk\_triggered}\}$。
- 阶段状态 $S=\{\mathrm{Initiated},\mathrm{Planning},\mathrm{Executing},\mathrm{Monitoring},\mathrm{Closing}\}$（定义 2.1.2）→ $\mathbf{Phase}$ 的对象；$\tau$ 实现 $S \times E \to S$ 的转换。
- 复合 $\tau_2 \circ \tau_1$ 对应顺序的 $\mathrm{transition}$ 调用。

---

## 1. Overview / 概述

**English / 英文**:

Lifecycle morphisms represent phase transitions, milestone achievements, and deliverable completions in the category $\mathbf{Phase}$. They capture how projects progress from one phase to another, ensuring proper sequencing and completion criteria. This document provides a category-theoretic perspective on lifecycle morphisms, aligning with PMBOK 7th Edition and ISO 21500 standards.

**中文**:

生命周期态射表示项目范畴 $\mathbf{Phase}$ 中的阶段转换、里程碑达成和交付物完成。它们捕捉项目如何从一个阶段进展到另一个阶段，确保正确的顺序和完成标准。本文档从范畴论视角提供生命周期态射的定义，对齐 PMBOK 第7版和 ISO 21500 标准。

**Key Insights / 关键洞察**:

- **Phase Transitions / 阶段转换**: Morphisms $\tau: Ph_i \to Ph_j$ represent phase transitions / 态射 $\tau: Ph_i \to Ph_j$ 表示阶段转换
- **Milestone Achievement / 里程碑达成**: Conditions for phase transitions / 阶段转换的条件
- **Deliverable Completion / 交付物完成**: Outputs required for transitions / 转换所需的输出
- **Composition / 复合**: Sequential phase transitions compose / 顺序阶段转换可以复合

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Phase Transition Morphisms / 阶段转换态射

**Definition 2.1** (Phase Transition Morphism)

A phase transition morphism $\tau: Ph_i \to Ph_j$ is a morphism in $\mathbf{Phase}$ that represents the transition from phase $Ph_i$ to phase $Ph_j$, defined as:

$$\tau(Ph_i) = Ph_j$$

subject to:

- **Preconditions / 前置条件**: All milestones in $Ph_i$ are achieved
- **Postconditions / 后置条件**: All deliverables in $Ph_i$ are completed
- **Validity / 有效性**: Transition is valid according to project constraints

### 2.2 Morphism Composition / 态射复合

**Definition 2.2** (Composition of Phase Transitions)

For phase transitions $\tau_1: Ph_1 \to Ph_2$ and $\tau_2: Ph_2 \to Ph_3$, their composition is:

$$(\tau_2 \circ \tau_1): Ph_1 \to Ph_3$$

defined by:
$$(\tau_2 \circ \tau_1)(Ph_1) = \tau_2(\tau_1(Ph_1)) = Ph_3$$

### 2.3 Identity Morphism / 恒等态射

**Definition 2.3** (Identity Phase Transition)

The identity morphism $\text{id}_{Ph}: Ph \to Ph$ represents staying in the same phase:

$$\text{id}_{Ph}(Ph) = Ph$$

---

## 3. Formal Definition / 形式化定义

### 3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义

**Definition 3.1** (Phase Transition - PMBOK 7th Edition)

A phase transition occurs when all phase deliverables are completed and phase gates are passed. In our formalization:

$$\tau: Ph_i \to Ph_j \text{ if } \forall d \in Deliverables(Ph_i): completed(d) \land \forall g \in Gates(Ph_i): passed(g)$$

**Standard Transitions / 标准转换**:

- **Initiation → Planning**: $\tau_{init}: Ph_{init} \to Ph_{plan}$
- **Planning → Execution**: $\tau_{plan}: Ph_{plan} \to Ph_{exec}$
- **Execution → Monitoring**: $\tau_{exec}: Ph_{exec} \to Ph_{mon}$
- **Monitoring → Closure**: $\tau_{mon}: Ph_{mon} \to Ph_{close}$

### 3.2 ISO 21500 Standard Definition / ISO 21500 标准定义

**Definition 3.2** (Phase Transition - ISO 21500:2012)

Phase transitions are controlled by phase gates. In our category-theoretic framework:

$$\tau: Ph_i \to Ph_j \text{ enabled if } gate(Ph_i, Ph_j) = True$$

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (Transition Determinism)

Phase transitions are deterministic:
$$\forall Ph_i, \exists! Ph_j: \tau(Ph_i) = Ph_j$$

**Property 4.2** (Transition Validity)

Phase transitions are valid only if preconditions are met:
$$\tau: Ph_i \to Ph_j \text{ valid } \iff \forall m \in Milestones(Ph_i): achieved(m)$$

**Property 4.3** (Transition Completeness)

Phase transitions ensure phase completion:
$$\tau: Ph_i \to Ph_j \Rightarrow \forall d \in Deliverables(Ph_i): completed(d)$$

### 4.2 Category-Theoretic Properties / 范畴论性质

**Property 4.4** (Associativity)

Phase transition composition is associative:
$$(\tau_3 \circ \tau_2) \circ \tau_1 = \tau_3 \circ (\tau_2 \circ \tau_1)$$

**Property 4.5** (Identity)

Identity morphisms act as units:
$$\text{id}_{Ph_j} \circ \tau = \tau = \tau \circ \text{id}_{Ph_i}$$

for $\tau: Ph_i \to Ph_j$.

---

## 5. Relations / 关系

### 5.1 Relations to Other Morphisms / 与其他态射的关系

**Relation 5.1** (Lifecycle → Resource Allocation)

Phase transitions trigger resource allocation:
$$alloc \circ \tau: Ph_i \to Allocation(Ph_j)$$

**Relation 5.2** (Lifecycle → Risk Assessment)

Phase transitions require risk assessment:
$$assess \circ \tau: Ph_i \to RiskSet(Ph_j)$$

**Relation 5.3** (Lifecycle → Quality Control)

Phase transitions involve quality control:
$$control \circ \tau: Ph_i \to QualityCheck(Ph_j)$$

### 5.2 Natural Transformations / 自然变换

**Natural Transformation 5.1** (Lifecycle-Resource)

There exists a natural transformation $\alpha: L \Rightarrow R$:
$$\alpha_\tau: L(\tau) \to R(\tau)$$

connecting lifecycle transitions to resource allocations.

---

## 6. Examples / 例子

### 6.1 Standard Phase Transition / 标准阶段转换

**Example 6.1** (Initiation to Planning Transition)

Consider the transition from initiation to planning:

$$\tau_{init}: Ph_{init} \to Ph_{plan}$$

where:

- **Preconditions**: Project charter approved, stakeholders identified
- **Postconditions**: Project plan created, resources allocated
- **Deliverables**: Project charter, stakeholder register

### 6.2 Agile Sprint Transition / 敏捷冲刺转换

**Example 6.2** (Sprint Transition)

In agile methodology, sprint transitions:

$$\tau_{sprint}: Sprint_i \to Sprint_{i+1}$$

where:

- **Preconditions**: Sprint goal achieved, sprint review completed
- **Postconditions**: Next sprint planned, backlog updated
- **Deliverables**: Sprint deliverables, retrospective report

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Phase Gate Control**: Controlling phase transitions using gates
- **Milestone Tracking**: Tracking milestone achievement for transitions
- **Deliverable Management**: Managing deliverables for phase completion
- **Transition Planning**: Planning phase transitions

### 7.2 Category-Theoretic Applications / 范畴论应用

- **Transition Composition**: Composing phase transitions
- **Transition Optimization**: Optimizing transition sequences
- **Transition Verification**: Verifying transition validity

### 7.3 直观解释与一例 / Intuitive Explanation with One Example

**Example 7.3** (Phase Transition as a Valid Step / 阶段转换即合规一步)

阶段转换态射 $\tau: Ph_i\to Ph_j$ 可看作**在阶段图上走一步**：只有满足前置（里程碑、交付物、门控）才能执行。复合 $\tau_2\circ\tau_1$ 即连续两步。例：$\tau_{init}: Init\to Plan$ 在 PMBOK 下要求 charter 获批、stakeholder 登记完成；$\tau_{sprint}: Sprint_i\to Sprint_{i+1}$ 要求评审与回顾完成。函子 $L$ 把项目映成阶段链，态射 $\tau$ 就是链上相邻节点间的边。见 [01-Lifecycle-Functor](../../04-Functors/01-Lifecycle-Functor.md)。

---

## 8. References / 参考文献

### 8.1 Standards / 标准

1. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
2. ISO 21500:2012. Guidance on project management. International Organization for Standardization.

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.
2. Awodey, S. (2010). *Category Theory* (2nd ed.). Oxford University Press.

### 8.3 Related Files / 相关文件

- [Lifecycle Objects](../../01-Objects/08-Lifecycle-Objects.md)
- [Project Objects](../../01-Objects/01-Project-Objects.md)
- [Lifecycle Functor](../../04-Functors/01-Lifecycle-Functor.md)
- [Lifecycle-Resource Natural Transformation](../../05-Natural-Transformations/01-Lifecycle-Resource-Natural-Transformation.md)
- **docs**：`docs/02-project-management/lifecycle-models`（$\mathcal{L}$、transition、$T$；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
