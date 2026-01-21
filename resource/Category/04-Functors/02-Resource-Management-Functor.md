# Resource Management Functor / 资源管理函子

## 📋 Table of Contents / 目录

- [Resource Management Functor / 资源管理函子](#resource-management-functor--资源管理函子)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Resource Management Functor Definition / 资源管理函子定义](#21-resource-management-functor-definition--资源管理函子定义)
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
    - [6.1 Software Project Resources / 软件项目资源](#61-software-project-resources--软件项目资源)
    - [6.2 Construction Project Resources / 建筑项目资源](#62-construction-project-resources--建筑项目资源)
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
- **转换关系**：**Resource Management Functor** = **层次转换**（项目 → 资源需求的层间映射）；与 03-资源管理概念、Category/01-Objects/09-Resource-Objects、Category/02-Morphisms/09-Resource-Morphisms、Category/05-Natural-Transformations/01-Lifecycle-Resource-Natural-Transformation 对应。

**与 docs 的公式对应**：$R(P)=Resources(P)$ 与 docs 的 $\mathcal{R}=(H,M,T,F)$ 及 $\mathrm{allocate}:\mathcal{T}\times\mathcal{R}\to\mathbb{R}^+$ 对应；资源约束、优化模型见 `docs/02-project-management/resource-models`。

---

## 1. Overview / 概述

**English / 英文**:

The resource management functor $R: \mathbf{Project} \to \mathbf{Resource}$ maps projects to their resource requirements. It extracts resource needs from projects while preserving project structure. This document provides a category-theoretic perspective on the resource management functor, aligning with PMBOK 7th Edition and ISO 21500 standards.

**中文**:

资源管理函子 $R: \mathbf{Project} \to \mathbf{Resource}$ 将项目映射到其资源需求。它在提取资源需求的同时保持项目结构。本文档从范畴论视角提供资源管理函子的定义，对齐 PMBOK 第7版和 ISO 21500 标准。

**Key Insights / 关键洞察**:

- **Resource Extraction / 资源提取**: Projects map to resource sets / 项目映射到资源集合
- **Structure Preservation / 结构保持**: Functor preserves project structure / 函子保持项目结构
- **Resource Allocation / 资源分配**: Functor enables resource allocation / 函子支持资源分配
- **Natural Transformations / 自然变换**: Connects to lifecycle and risk functors / 连接到生命周期和风险函子

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Resource Management Functor Definition / 资源管理函子定义

**Definition 2.1** (Resource Management Functor)

The resource management functor $R: \mathbf{Project} \to \mathbf{Resource}$ is defined as:

- **Object Mapping / 对象映射**:
  $$R(P) = Resources(P) = \{Res_1, Res_2, \ldots, Res_n\}$$
  where $P \in \mathbf{Project}$ and $Res_i \in \mathbf{Resource}$.

- **Morphism Mapping / 态射映射**:
  For a project morphism $f: P_1 \to P_2$, the functor maps it to:
  $$R(f): R(P_1) \to R(P_2)$$
  preserving resource transformations.

### 2.2 Functor Properties / 函子性质

**Axiom 2.1** (Functor Identity Preservation)

The resource functor preserves identity:
$$R(\text{id}_P) = \text{id}_{R(P)}$$

**Axiom 2.2** (Functor Composition Preservation)

The resource functor preserves composition:
$$R(g \circ f) = R(g) \circ R(f)$$

for composable morphisms $f: P_1 \to P_2$ and $g: P_2 \to P_3$.

---

## 3. Formal Definition / 形式化定义

### 3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义

**Definition 3.1** (Resource Management - PMBOK 7th Edition)

Resource management includes processes to identify, acquire, and manage resources. In our functor framework:

$$R: \mathbf{Project} \to \mathbf{Resource}$$

where $R(P)$ extracts resources from project $P$.

**Resource Types / 资源类型**:

- **Human Resources / 人力资源**: $R_{human}(P)$
- **Material Resources / 物质资源**: $R_{material}(P)$
- **Technical Resources / 技术资源**: $R_{technical}(P)$
- **Financial Resources / 财务资源**: $R_{financial}(P)$

### 3.2 ISO 21500 Standard Definition / ISO 21500 标准定义

**Definition 3.2** (Resource Management - ISO 21500:2012)

Resource management includes processes to manage project resources. In our category-theoretic framework:

$$R: \mathbf{Project} \to \mathbf{Resource}$$

where $R$ maps projects to resource requirements.

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (Resource Existence)

Every project requires resources:
$$\forall P \in \mathbf{Project}: R(P) \neq \emptyset$$

**Property 4.2** (Resource Feasibility)

Resource requirements are feasible:
$$\forall P \in \mathbf{Project}, \forall Res \in R(P): Capacity(Res) \geq Requirement(Res, P)$$

**Property 4.3** (Resource Allocation)

Resources can be allocated:
$$alloc: R(P) \times T \to \mathbb{R}^+$$

### 4.2 Category-Theoretic Properties / 范畴论性质

**Property 4.4** (Functor Covariance)

The resource functor is covariant:
$$R: \mathbf{Project} \to \mathbf{Resource}$$

**Property 4.5** (Functor Composition)

The resource functor composes with lifecycle functor:
$$R \circ L: \mathbf{Project} \to \mathbf{Resource} \times \mathbf{Phase}$$

---

## 5. Relations / 关系

### 5.1 Relations to Other Functors / 与其他函子的关系

**Relation 5.1** (Resource → Lifecycle)

Resources are allocated to lifecycle phases:
$$R \circ L: \mathbf{Project} \to \mathbf{Resource} \times \mathbf{Phase}$$

**Relation 5.2** (Resource → Risk)

Resource constraints create risks:
$$Risk \circ R: \mathbf{Project} \to \mathbf{Risk}$$

**Relation 5.3** (Resource → Quality)

Resource quality affects project quality:
$$Q \circ R: \mathbf{Project} \to \mathbf{Quality}$$

### 5.2 Natural Transformations / 自然变换

**Natural Transformation 5.1** (Lifecycle-Resource)

There exists a natural transformation $\alpha: L \Rightarrow R$:
$$\alpha_P: L(P) \to R(P)$$

connecting lifecycle phases to resource requirements.

**Natural Transformation 5.2** (Resource-Risk)

There exists a natural transformation $\beta: R \Rightarrow Risk$:
$$\beta_P: R(P) \to Risk(P)$$

connecting resource constraints to risks.

---

## 6. Examples / 例子

### 6.1 Software Project Resources / 软件项目资源

**Example 6.1** (Development Team Resources)

Consider a software project $P_{sw}$:

$$R(P_{sw}) = \{Res_{dev}, Res_{tester}, Res_{budget}\}$$

where:

- $Res_{dev}$ - developer resources
- $Res_{tester}$ - tester resources
- $Res_{budget}$ - budget resources

### 6.2 Construction Project Resources / 建筑项目资源

**Example 6.2** (Construction Resources)

Consider a construction project $P_{constr}$:

$$R(P_{constr}) = \{Res_{labor}, Res_{material}, Res_{equipment}, Res_{budget}\}$$

with diverse resource types.

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Resource Planning**: Planning resources using functor
- **Resource Allocation**: Allocating resources to projects
- **Resource Optimization**: Optimizing resource utilization
- **Resource Monitoring**: Monitoring resource usage

### 7.2 Category-Theoretic Applications / 范畴论应用

- **Functor Composition**: Composing resource functor with other functors
- **Natural Transformations**: Understanding relationships via natural transformations
- **Category Mapping**: Mapping between project and resource categories

---

## 8. References / 参考文献

### 8.1 Standards / 标准

1. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
2. ISO 21500:2012. Guidance on project management. International Organization for Standardization.

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.
2. Awodey, S. (2010). *Category Theory* (2nd ed.). Oxford University Press.

### 8.3 Related Files / 相关文件

- [Resource Objects](../../01-Objects/09-Resource-Objects.md)
- [Resource Morphisms](../../02-Morphisms/09-Resource-Morphisms.md)
- [Lifecycle-Resource Natural Transformation](../../05-Natural-Transformations/01-Lifecycle-Resource-Natural-Transformation.md)
- **docs**：`docs/02-project-management/resource-models`（$\mathcal{R}$、allocate；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
