# Lifecycle Functor / 生命周期函子

## 📋 Table of Contents / 目录

- [Lifecycle Functor / 生命周期函子](#lifecycle-functor--生命周期函子)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Lifecycle Functor Definition / 生命周期函子定义](#21-lifecycle-functor-definition--生命周期函子定义)
    - [2.2 Functor Properties / 函子性质](#22-functor-properties--函子性质)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义](#31-pmbok-7th-edition-definition--pmbok-第7版定义)
    - [3.2 ISO 21500 Standard Definition / ISO 21500 标准定义](#32-iso-21500-standard-definition--iso-21500-标准定义)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Basic Properties / 基本性质](#41-basic-properties--基本性质)
    - [4.2 Category-Theoretic Properties / 范畴论性质](#42-category-theoretic-properties--范畴论性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Functors / 与其他函子的关系](#51-relations-to-other-functors--与其他函子的关系)
    - [5.2 Natural Transformations / 自然变换](#52-natural-transformations--自然变换)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Standard Lifecycle Example / 标准生命周期例子](#61-standard-lifecycle-example--标准生命周期例子)
    - [6.2 Agile Lifecycle Example / 敏捷生命周期例子](#62-agile-lifecycle-example--敏捷生命周期例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Project Management Applications / 项目管理应用](#71-project-management-applications--项目管理应用)
    - [7.2 Category-Theoretic Applications / 范畴论应用](#72-category-theoretic-applications--范畴论应用)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Standards / 标准](#81-standards--标准)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**核心模型层**（对应 docs/02-project-management；层次转换 L1→…→L5）
- **转换关系**：**Lifecycle Functor** = **层次转换**（项目 → 生命周期阶段的层间映射）；与 02-生命周期概念、Category/01-Objects/08-Lifecycle-Objects、Category/02-Morphisms/08-Lifecycle-Morphisms、Category/05-Natural-Transformations/01-Lifecycle-Resource-Natural-Transformation 对应。

**与 docs/02-project-management/lifecycle-models 的公式对应**：

- 生命周期 $\mathcal{L}=(P,T,G,C)$（定义 2.1.1）→ 函子对象映射 $L(P)=(Ph_1,\ldots,Ph_n)$ 中 $Ph_i \in P$。
- **生命周期转换** $\mathrm{transition}: P \times E \to P$（定义 2.1.3）→ 函子态射映射 $L(f): L(P_1)\to L(P_2)$ 保持阶段转换；事件 $E \supseteq \{\mathrm{phase\_complete},\mathrm{gate\_approved},\mathrm{change\_requested},\mathrm{risk\_triggered}\}$。
- PMBOK 五过程组 $\mathcal{L}_{PMBOK}=(\mathrm{Initiating},\mathrm{Planning},\mathrm{Executing},\mathrm{Monitoring\,\&\,Controlling},\mathrm{Closing})$（定义 2.1.4）→ $L(P)$ 的序列与之一致。

---

## 1. Overview / 概述

**English / 英文**:

The lifecycle functor $L: \mathbf{Project} \to \mathbf{Phase}$ maps projects to their lifecycle phases. It preserves project structure while extracting the temporal evolution aspect. This document provides a category-theoretic perspective on the lifecycle functor, aligning with PMBOK 7th Edition and ISO 21500 standards.

**中文**:

生命周期函子 $L: \mathbf{Project} \to \mathbf{Phase}$ 将项目映射到其生命周期阶段。它在提取时序演进方面的同时保持项目结构。本文档从范畴论视角提供生命周期函子的定义，对齐 PMBOK 第7版和 ISO 21500 标准。

**Key Insights / 关键洞察**:

- **Functor Mapping / 函子映射**: Projects map to phase sequences / 项目映射到阶段序列
- **Structure Preservation / 结构保持**: Functor preserves project structure / 函子保持项目结构
- **Composition / 复合**: Functor preserves composition / 函子保持复合
- **Natural Transformations / 自然变换**: Connects to other functors / 连接到其他函子

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Lifecycle Functor Definition / 生命周期函子定义

**Definition 2.1** (Lifecycle Functor)

The lifecycle functor $L: \mathbf{Project} \to \mathbf{Phase}$ is defined as:

- **Object Mapping / 对象映射**:
  $$L(P) = Lifecycle(P) = (Ph_1, Ph_2, \ldots, Ph_n)$$
  where $P \in \mathbf{Project}$ and $Ph_i \in \mathbf{Phase}$.

- **Morphism Mapping / 态射映射**:
  For a project morphism $f: P_1 \to P_2$, the functor maps it to:
  $$L(f): L(P_1) \to L(P_2)$$
  preserving phase transitions.

### 2.2 Functor Properties / 函子性质

**Axiom 2.1** (Functor Identity Preservation)

The lifecycle functor preserves identity:
$$L(\text{id}_P) = \text{id}_{L(P)}$$

**Axiom 2.2** (Functor Composition Preservation)

The lifecycle functor preserves composition:
$$L(g \circ f) = L(g) \circ L(f)$$

for composable morphisms $f: P_1 \to P_2$ and $g: P_2 \to P_3$.

---

## 3. Formal Definition / 形式化定义

### 3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义

**Definition 3.1** (Project Lifecycle - PMBOK 7th Edition)

A project lifecycle is the series of phases that a project passes through. In our functor framework:

$$L: \mathbf{Project} \to \mathbf{Phase}$$

where $L(P)$ extracts the lifecycle from project $P$.

**Standard Lifecycle / 标准生命周期**:
$$L(P) = (Ph_{init}, Ph_{plan}, Ph_{exec}, Ph_{mon}, Ph_{close})$$

### 3.2 ISO 21500 Standard Definition / ISO 21500 标准定义

**Definition 3.2** (Project Lifecycle - ISO 21500:2012)

A project lifecycle consists of project phases. In our category-theoretic framework:

$$L: \mathbf{Project} \to \mathbf{Phase}^*$$

where $\mathbf{Phase}^*$ denotes sequences of phases.

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (Lifecycle Existence)

Every project has a lifecycle:
$$\forall P \in \mathbf{Project}: L(P) \neq \emptyset$$

**Property 4.2** (Lifecycle Uniqueness)

Each project has a unique lifecycle:
$$\forall P_1, P_2 \in \mathbf{Project}: P_1 = P_2 \Rightarrow L(P_1) = L(P_2)$$

**Property 4.3** (Phase Sequence)

Lifecycle phases form a sequence:
$$L(P) = (Ph_1, Ph_2, \ldots, Ph_n) \text{ where } Ph_i \to Ph_{i+1}$$

### 4.2 Category-Theoretic Properties / 范畴论性质

**Property 4.4** (Functor Covariance)

The lifecycle functor is covariant:
$$L: \mathbf{Project} \to \mathbf{Phase}$$

**Property 4.5** (Functor Faithfulness)

The lifecycle functor is faithful:
$$L(f_1) = L(f_2) \Rightarrow f_1 = f_2$$

for morphisms $f_1, f_2: P_1 \to P_2$.

---

## 5. Relations / 关系

### 5.1 Relations to Other Functors / 与其他函子的关系

**Relation 5.1** (Lifecycle → Resource)

Lifecycle phases require resources:
$$R \circ L: \mathbf{Project} \to \mathbf{Resource}$$

**Relation 5.2** (Lifecycle → Risk)

Lifecycle phases have associated risks:
$$Risk \circ L: \mathbf{Project} \to \mathbf{Risk}$$

**Relation 5.3** (Lifecycle → Quality)

Lifecycle phases have quality requirements:
$$Q \circ L: \mathbf{Project} \to \mathbf{Quality}$$

### 5.2 Natural Transformations / 自然变换

**Natural Transformation 5.1** (Lifecycle-Resource)

There exists a natural transformation $\alpha: L \Rightarrow R$:
$$\alpha_P: L(P) \to R(P)$$

connecting lifecycle phases to resource requirements.

**Natural Transformation 5.2** (Lifecycle-Quality)

There exists a natural transformation $\delta: L \Rightarrow Q$:
$$\delta_P: L(P) \to Q(P)$$

connecting lifecycle phases to quality requirements.

---

## 6. Examples / 例子

### 6.1 Standard Lifecycle Example / 标准生命周期例子

**Example 6.1** (Software Development Lifecycle)

Consider a software project $P_{sw}$:

$$L(P_{sw}) = (Ph_{init}, Ph_{plan}, Ph_{exec}, Ph_{mon}, Ph_{close})$$

where each phase is mapped from the project.

### 6.2 Agile Lifecycle Example / 敏捷生命周期例子

**Example 6.2** (Agile Sprint Lifecycle)

Consider an agile project $P_{agile}$:

$$L(P_{agile}) = (Sprint_1, Sprint_2, \ldots, Sprint_n)$$

where sprints form the lifecycle.

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Lifecycle Extraction**: Extracting lifecycle from projects
- **Phase Planning**: Planning phases using functor
- **Lifecycle Analysis**: Analyzing lifecycle structure
- **Lifecycle Optimization**: Optimizing lifecycle sequences

### 7.2 Category-Theoretic Applications / 范畴论应用

- **Functor Composition**: Composing lifecycle functor with other functors
- **Natural Transformations**: Understanding relationships via natural transformations
- **Category Mapping**: Mapping between project and phase categories

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
- [Lifecycle Morphisms](../../02-Morphisms/08-Lifecycle-Morphisms.md)
- [Lifecycle-Resource Natural Transformation](../../05-Natural-Transformations/01-Lifecycle-Resource-Natural-Transformation.md)
- **docs**：`docs/02-project-management/lifecycle-models`（$\mathcal{L}$、层次转换；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
