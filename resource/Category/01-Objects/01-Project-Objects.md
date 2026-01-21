# Project Objects / 项目对象

## 📋 Table of Contents / 目录

- [Project Objects / 项目对象](#project-objects--项目对象)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Category of Projects / 项目范畴](#21-category-of-projects--项目范畴)
    - [2.2 Project Object Properties / 项目对象性质](#22-project-object-properties--项目对象性质)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 ISO 21500 Standard Definition / ISO 21500 标准定义](#31-iso-21500-standard-definition--iso-21500-标准定义)
    - [3.2 PMBOK 7th Edition Definition / PMBOK 第7版定义](#32-pmbok-7th-edition-definition--pmbok-第7版定义)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Basic Properties / 基本性质](#41-basic-properties--基本性质)
    - [4.2 Category-Theoretic Properties / 范畴论性质](#42-category-theoretic-properties--范畴论性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Objects / 与其他对象的关系](#51-relations-to-other-objects--与其他对象的关系)
    - [5.2 Natural Transformations / 自然变换](#52-natural-transformations--自然变换)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Simple Project Example / 简单项目例子](#61-simple-project-example--简单项目例子)
    - [6.2 Complex Project Example / 复杂项目例子](#62-complex-project-example--复杂项目例子)
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

- **所属层**：**基础理论层**（对应 docs/01-foundations；项目定义、状态空间、约束）
- **转换关系**：**Project Objects** 作为**状态转换**的基础（项目状态空间、状态转换 $\rightarrow$）；与 01-项目定义、03-项目状态空间、Category/02-Morphisms、Category/04-Functors/01-Lifecycle-Functor 对应。

**与 docs 的对应**：状态空间 $S$、Kripke 型转换 $\delta: S \times \Sigma \to S$、约束 $C$ 见 docs/01-foundations；生命周期中的 $P,T,G,C$ 见 docs/02-project-management/lifecycle-models。

---

## 1. Overview / 概述

**English / 英文**:

Project objects are the fundamental entities in the category of projects $\mathbf{Project}$. They represent projects as structured entities with states, resources, time constraints, and constraints. This document provides a category-theoretic perspective on project objects, aligning with authoritative resources and providing formal definitions.

**中文**:

项目对象是项目范畴 $\mathbf{Project}$ 中的基本实体。它们将项目表示为具有状态、资源、时间约束和约束条件的结构化实体。本文档从范畴论视角提供项目对象的定义，对齐权威资源并提供形式化定义。

**Key Insights / 关键洞察**:

- **Basic Structure / 基本结构**: Projects are objects in the category $\mathbf{Project}$ / 项目是范畴 $\mathbf{Project}$ 中的对象
- **State Space / 状态空间**: Each project has a state space $S \subseteq \mathbb{R}^n$ / 每个项目都有一个状态空间 $S \subseteq \mathbb{R}^n$
- **Resources / 资源**: Projects have resource sets $R$ / 项目有资源集合 $R$
- **Constraints / 约束**: Projects have constraint functions $C: S \times R \times T \to \{True, False\}$ / 项目有约束函数 $C: S \times R \times T \to \{True, False\}$

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Category of Projects / 项目范畴

**Definition 2.1** (Category $\mathbf{Project}$)

The category $\mathbf{Project}$ is defined as follows:

- **Objects / 对象**: Projects $P = (S, R, T, C)$ where:
  - $S$ is the state space, $S \subseteq \mathbb{R}^n$
  - $R$ is the resource set, $R = \{r_i \mid r_i \in \mathbb{R}^+, i \in \mathbb{N}\}$
  - $T$ is the time constraints, $T \subseteq \mathbb{R}^+ \times \mathbb{R}^+$
  - $C$ is the constraint function, $C: S \times R \times T \to \{True, False\}$

- **Morphisms / 态射**: Project transformations $f: P_1 \to P_2$ that preserve project structure

- **Composition / 复合**: Composition of project transformations $(g \circ f): P_1 \to P_3$ where $f: P_1 \to P_2$ and $g: P_2 \to P_3$

- **Identity / 恒等**: Identity transformation $\text{id}_P: P \to P$ that maps each project to itself

### 2.2 Project Object Properties / 项目对象性质

**Axiom 2.1** (Project Object Existence)

For any valid state space $S$, resource set $R$, time constraints $T$, and constraint function $C$, there exists a project object $P = (S, R, T, C)$ in $\mathbf{Project}$.

**Axiom 2.2** (Project Object Uniqueness)

If two project objects $P_1 = (S_1, R_1, T_1, C_1)$ and $P_2 = (S_2, R_2, T_2, C_2)$ have identical components, then $P_1 = P_2$.

---

## 3. Formal Definition / 形式化定义

### 3.1 ISO 21500 Standard Definition / ISO 21500 标准定义

**Definition 3.1** (Project - ISO 21500:2012)

A project is a unique set of processes consisting of coordinated and controlled activities with start and finish dates, undertaken to achieve an objective. In our formalization:

$$P = (S, R, T, C)$$

where:

- $S$: State space representing project states
- $R$: Resource set representing available resources
- $T$: Time constraints representing project timeline
- $C$: Constraint function ensuring feasibility

### 3.2 PMBOK 7th Edition Definition / PMBOK 第7版定义

**Definition 3.2** (Project - PMBOK 7th Edition)

A project is a temporary endeavor undertaken to create a unique product, service, or result. In our category-theoretic framework:

$$P \in \text{Ob}(\mathbf{Project})$$

with morphisms representing project transitions and transformations.

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (State Space Non-emptiness)

For any project $P = (S, R, T, C)$, the state space $S$ is non-empty:
$$S \neq \emptyset$$

**Property 4.2** (Resource Set Non-negativity)

For any project $P = (S, R, T, C)$, all resources are non-negative:
$$\forall r \in R: r \geq 0$$

**Property 4.3** (Time Constraint Validity)

For any project $P = (S, R, T, C)$, time constraints are valid:
$$\forall (t_1, t_2) \in T: t_1 < t_2$$

### 4.2 Category-Theoretic Properties / 范畴论性质

**Property 4.4** (Project Object Composition)

Project objects compose under morphisms:
$$(h \circ g) \circ f = h \circ (g \circ f)$$

for composable morphisms $f: P_1 \to P_2$, $g: P_2 \to P_3$, $h: P_3 \to P_4$.

**Property 4.5** (Project Object Identity)

Every project object has an identity morphism:
$$\text{id}_P \circ f = f = f \circ \text{id}_P$$

for any morphism $f: P \to P'$.

---

## 5. Relations / 关系

### 5.1 Relations to Other Objects / 与其他对象的关系

**Relation 5.1** (Project → Lifecycle)

Every project $P$ has an associated lifecycle $L(P)$:
$$L: \mathbf{Project} \to \mathbf{Phase}$$

**Relation 5.2** (Project → Resources)

Every project $P$ has resource requirements $R(P)$:
$$R: \mathbf{Project} \to \mathbf{Resource}$$

**Relation 5.3** (Project → Risks)

Every project $P$ has associated risks $Risk(P)$:
$$Risk: \mathbf{Project} \to \mathbf{Risk}$$

**Relation 5.4** (Project → Quality)

Every project $P$ has quality attributes $Q(P)$:
$$Q: \mathbf{Project} \to \mathbf{Quality}$$

### 5.2 Natural Transformations / 自然变换

**Natural Transformation 5.1** (Lifecycle-Resource)

There exists a natural transformation $\alpha: L \Rightarrow R$ connecting lifecycle and resource functors:
$$\alpha_P: L(P) \to R(P)$$

---

## 6. Examples / 例子

### 6.1 Simple Project Example / 简单项目例子

**Example 6.1** (Software Development Project)

Consider a software development project $P_{sw}$:

$$P_{sw} = (S_{sw}, R_{sw}, T_{sw}, C_{sw})$$

where:

- $S_{sw} = \{\text{Init}, \text{Plan}, \text{Dev}, \text{Test}, \text{Deploy}\}$ - project states
- $R_{sw} = \{5 \text{ developers}, 2 \text{ testers}, \$100k \text{ budget}\}$ - resources
- $T_{sw} = \{(0, 180)\}$ - 180 days timeline
- $C_{sw}$ - constraints ensuring feasibility

### 6.2 Complex Project Example / 复杂项目例子

**Example 6.2** (Construction Project)

Consider a construction project $P_{constr}$:

$$P_{constr} = (S_{constr}, R_{constr}, T_{constr}, C_{constr})$$

with multiple phases, diverse resources, and complex constraints.

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Project Planning**: Using project objects to model project structure
- **Resource Allocation**: Mapping projects to resource requirements
- **Risk Assessment**: Identifying risks associated with projects
- **Quality Management**: Ensuring quality attributes in projects

### 7.2 Category-Theoretic Applications / 范畴论应用

- **Project Composition**: Composing projects using morphisms
- **Project Transformation**: Transforming projects using functors
- **Project Relationships**: Understanding relationships using natural transformations

### 7.3 直观解释与一例 / Intuitive Explanation with One Example

**Example 7.3** (Project as a Structured Box / 项目即结构化盒子)

把项目 $P=(S,R,T,C)$ 想成一个**带状态的盒子**：$S$ 是盒内状态（阶段、进展），$R$ 是可用资源，$T$ 是时间窗，$C$ 检查是否可行。范畴中的态射 $f:P_1\to P_2$ 就是把一个盒子里的状态与约束，合规地迁到另一个盒子；函子 $L$、$R$、$Risk$、$Q$ 则从 $P$ 抽出一维（生命周期、资源、风险、质量）做分析。例如软件开发 $P_{sw}$：$L(P_{sw})$ 给出 Init→Plan→Dev→Test→Deploy 的路径，$Risk(P_{sw})$ 给出技术债、需求变更等风险集。

---

## 8. References / 参考文献

### 8.1 Standards / 标准

1. ISO 21500:2012. Guidance on project management. International Organization for Standardization.
2. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.
2. Awodey, S. (2010). *Category Theory* (2nd ed.). Oxford University Press.
3. Riehl, E. (2017). *Category Theory in Context*. Dover Publications.

### 8.3 Related Files / 相关文件

- [Lifecycle Objects](08-Lifecycle-Objects.md)
- [Resource Objects](09-Resource-Objects.md)
- [Risk Objects](10-Risk-Objects.md)
- [Quality Objects](11-Quality-Objects.md)
- [Lifecycle Functor](../../04-Functors/01-Lifecycle-Functor.md)
- **docs**：`docs/01-foundations`（状态空间 $S$、Kripke、$\rightarrow$）；`docs/02-project-management/lifecycle-models`（$P,T,G,C$；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
