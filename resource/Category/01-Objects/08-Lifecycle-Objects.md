# Lifecycle Objects / 生命周期对象

## 📋 Table of Contents / 目录

- [Lifecycle Objects / 生命周期对象](#lifecycle-objects--生命周期对象)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Category of Phases / 阶段范畴](#21-category-of-phases--阶段范畴)
    - [2.2 Lifecycle Object Properties / 生命周期对象性质](#22-lifecycle-object-properties--生命周期对象性质)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义](#31-pmbok-7th-edition-definition--pmbok-第7版定义)
    - [3.2 ISO 21500 Standard Definition / ISO 21500 标准定义](#32-iso-21500-standard-definition--iso-21500-标准定义)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Basic Properties / 基本性质](#41-basic-properties--基本性质)
    - [4.2 Category-Theoretic Properties / 范畴论性质](#42-category-theoretic-properties--范畴论性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Objects / 与其他对象的关系](#51-relations-to-other-objects--与其他对象的关系)
    - [5.2 Natural Transformations / 自然变换](#52-natural-transformations--自然变换)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Standard Project Lifecycle / 标准项目生命周期](#61-standard-project-lifecycle--标准项目生命周期)
    - [6.2 Agile Project Lifecycle / 敏捷项目生命周期](#62-agile-project-lifecycle--敏捷项目生命周期)
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

- **所属层**：**核心模型层**（对应 docs/02-project-management/lifecycle-models；生命周期模型）
- **转换关系**：**Lifecycle Objects** 作为**生命周期转换** $\delta$ 的实体（项目阶段、里程碑、交付物）；与 02-生命周期概念、Category/02-Morphisms/08-Lifecycle-Morphisms、Category/04-Functors/01-Lifecycle-Functor 对应。
- **与 docs 的公式对应**：docs/02-project-management/lifecycle-models 的 $\mathcal{L}=(P,T,G,C)$、阶段状态 $S$、$\mathrm{transition}: P \times E \to P$、转换点 $T$ 与本文件的 $\mathbf{Phase}$、阶段、里程碑、交付物 对应。

---

## 1. Overview / 概述

**English / 英文**:

Lifecycle objects represent project phases, milestones, and deliverables in the category $\mathbf{Phase}$. They capture the temporal evolution of projects through distinct stages from initiation to closure. This document provides a category-theoretic perspective on lifecycle objects, aligning with PMBOK 7th Edition and ISO 21500 standards.

**中文**:

生命周期对象表示项目范畴 $\mathbf{Phase}$ 中的项目阶段、里程碑和交付物。它们捕捉项目从启动到收尾的各个阶段的时序演进。本文档从范畴论视角提供生命周期对象的定义，对齐 PMBOK 第7版和 ISO 21500 标准。

**Key Insights / 关键洞察**:

- **Phases / 阶段**: Projects progress through distinct phases / 项目通过不同阶段演进
- **Milestones / 里程碑**: Key decision points and achievements / 关键决策点和成就
- **Deliverables / 交付物**: Tangible outputs at each phase / 每个阶段的有形输出
- **Transitions / 转换**: Morphisms between phases / 阶段之间的态射

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Category of Phases / 阶段范畴

**Definition 2.1** (Category $\mathbf{Phase}$)

The category $\mathbf{Phase}$ is defined as follows:

- **Objects / 对象**: Project phases $Ph = (Name, State, Milestones, Deliverables)$ where:
  - $Name$ is the phase name (e.g., Initiation, Planning, Execution, Monitoring, Closure)
  - $State$ is the phase state space $S_{ph} \subseteq S$ where $S$ is the project state space
  - $Milestones$ is the set of milestones $M = \{m_1, m_2, \ldots, m_n\}$
  - $Deliverables$ is the set of deliverables $D = \{d_1, d_2, \ldots, d_k\}$

- **Morphisms / 态射**: Phase transitions $\tau: Ph_i \to Ph_j$ representing progression from phase $i$ to phase $j$

- **Composition / 复合**: Composition of phase transitions $(\tau_2 \circ \tau_1): Ph_1 \to Ph_3$

- **Identity / 恒等**: Identity transition $\text{id}_{Ph}: Ph \to Ph$ representing staying in the same phase

### 2.2 Lifecycle Object Properties / 生命周期对象性质

**Axiom 2.1** (Phase Sequence)

For any project lifecycle, phases occur in a sequence:
$$Ph_1 \xrightarrow{\tau_1} Ph_2 \xrightarrow{\tau_2} Ph_3 \xrightarrow{\tau_3} \cdots \xrightarrow{\tau_{n-1}} Ph_n$$

**Axiom 2.2** (Milestone Achievement)

Each milestone $m \in Milestones$ must be achieved before phase transition:
$$\forall \tau: Ph_i \to Ph_j, \forall m \in Milestones(Ph_i): achieved(m) \Rightarrow \tau \text{ is enabled}$$

---

## 3. Formal Definition / 形式化定义

### 3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义

**Definition 3.1** (Project Lifecycle - PMBOK 7th Edition)

A project lifecycle is the series of phases that a project passes through from its start to its completion. In our formalization:

$$Lifecycle(P) = (Ph_1, Ph_2, \ldots, Ph_n)$$

where each phase $Ph_i$ is an object in $\mathbf{Phase}$.

**Standard Phases / 标准阶段**:

- **Initiation / 启动**: $Ph_{init}$ - Project authorization and initial planning
- **Planning / 规划**: $Ph_{plan}$ - Detailed planning and preparation
- **Execution / 执行**: $Ph_{exec}$ - Work performance and deliverables creation
- **Monitoring / 监控**: $Ph_{mon}$ - Tracking, reviewing, and regulating progress
- **Closure / 收尾**: $Ph_{close}$ - Finalizing all activities and project closure

### 3.2 ISO 21500 Standard Definition / ISO 21500 标准定义

**Definition 3.2** (Project Lifecycle - ISO 21500:2012)

A project lifecycle consists of project phases. In our category-theoretic framework:

$$Lifecycle: \mathbf{Project} \to \mathbf{Phase}^*$$

where $\mathbf{Phase}^*$ denotes sequences of phases.

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (Phase Non-emptiness)

For any lifecycle, there exists at least one phase:
$$|Lifecycle(P)| \geq 1$$

**Property 4.2** (Phase Uniqueness)

Each phase in a lifecycle is unique:
$$\forall Ph_i, Ph_j \in Lifecycle(P): i \neq j \Rightarrow Ph_i \neq Ph_j$$

**Property 4.3** (Transition Determinism)

Phase transitions are deterministic:
$$\forall Ph_i, \exists! Ph_j: \tau(Ph_i) = Ph_j$$

### 4.2 Category-Theoretic Properties / 范畴论性质

**Property 4.4** (Lifecycle Functor)

The lifecycle mapping is a functor:
$$L: \mathbf{Project} \to \mathbf{Phase}$$

**Property 4.5** (Phase Transition Composition)

Phase transitions compose associatively:
$$(\tau_3 \circ \tau_2) \circ \tau_1 = \tau_3 \circ (\tau_2 \circ \tau_1)$$

---

## 5. Relations / 关系

### 5.1 Relations to Other Objects / 与其他对象的关系

**Relation 5.1** (Lifecycle → Project)

Every lifecycle belongs to a project:
$$L^{-1}: \mathbf{Phase} \to \mathbf{Project}$$

**Relation 5.2** (Lifecycle → Resources)

Each phase requires resources:
$$R \circ L: \mathbf{Project} \to \mathbf{Resource}$$

**Relation 5.3** (Lifecycle → Risks)

Each phase has associated risks:
$$Risk \circ L: \mathbf{Project} \to \mathbf{Risk}$$

**Relation 5.4** (Lifecycle → Quality)

Each phase has quality requirements:
$$Q \circ L: \mathbf{Project} \to \mathbf{Quality}$$

### 5.2 Natural Transformations / 自然变换

**Natural Transformation 5.1** (Lifecycle-Resource)

There exists a natural transformation $\alpha: L \Rightarrow R$:
$$\alpha_P: L(P) \to R(P)$$

connecting lifecycle phases to resource requirements.

---

## 6. Examples / 例子

### 6.1 Standard Project Lifecycle / 标准项目生命周期

**Example 6.1** (Software Development Lifecycle)

Consider a software development project with lifecycle:

$$Lifecycle(P_{sw}) = (Ph_{init}, Ph_{plan}, Ph_{exec}, Ph_{mon}, Ph_{close})$$

where:

- $Ph_{init}$: Project charter, stakeholder identification
- $Ph_{plan}$: Requirements, design, schedule
- $Ph_{exec}$: Coding, unit testing
- $Ph_{mon}$: Integration testing, quality control
- $Ph_{close}$: Deployment, documentation, handover

### 6.2 Agile Project Lifecycle / 敏捷项目生命周期

**Example 6.2** (Agile Sprint Lifecycle)

An agile project has iterative lifecycle:

$$Lifecycle(P_{agile}) = (Sprint_1, Sprint_2, \ldots, Sprint_n)$$

where each sprint follows: Planning → Development → Review → Retrospective.

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Phase Planning**: Using lifecycle objects to plan project phases
- **Milestone Tracking**: Monitoring milestone achievement
- **Deliverable Management**: Managing deliverables at each phase
- **Transition Control**: Controlling phase transitions

### 7.2 Category-Theoretic Applications / 范畴论应用

- **Lifecycle Composition**: Composing lifecycles using morphisms
- **Phase Transformation**: Transforming phases using functors
- **Lifecycle Optimization**: Optimizing lifecycles using natural transformations

### 7.3 直观解释与一例 / Intuitive Explanation with One Example

**Example 7.3** (Lifecycle as a Pipeline / 生命周期即流水线)

生命周期对象 $Lc=(Ph,Gate,Deliver)$ 可看作一条**阶段流水线**：$Ph$ 是阶段序列，$Gate$ 是阶段门，$Deliver$ 是交付物。函子 $L:\mathbf{Project}\to\mathbf{Phase}$ 把项目「压成」一条 phase 链。例如 $L(P_{sw})=Ph_1\to Ph_2\to\cdots\to Ph_5$（启动→规划→执行→监控→收尾）；自然变换 $\alpha:L\Rightarrow R$ 表示「每到一个阶段，对应一批资源需求」，即 $\alpha_P: L(P)\to R(P)$ 把阶段信息映射为资源调配。

---

## 8. References / 参考文献

### 8.1 Standards / 标准

1. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
2. ISO 21500:2012. Guidance on project management. International Organization for Standardization.

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.
2. Awodey, S. (2010). *Category Theory* (2nd ed.). Oxford University Press.

### 8.3 Related Files / 相关文件

- [Project Objects](01-Project-Objects.md)
- [Resource Objects](09-Resource-Objects.md)
- [Lifecycle Morphisms](../../02-Morphisms/08-Lifecycle-Morphisms.md)
- [Lifecycle Functor](../../04-Functors/01-Lifecycle-Functor.md)
- **docs**：`docs/02-project-management/lifecycle-models`（$\mathcal{L}=(P,T,G,C)$、$\mathrm{transition}$、转换点 $T$；与 0. 公式对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
