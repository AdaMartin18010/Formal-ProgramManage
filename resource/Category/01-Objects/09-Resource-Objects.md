# Resource Objects / 资源对象

## 📋 Table of Contents / 目录

- [Resource Objects / 资源对象](#resource-objects--资源对象)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Category of Resources / 资源范畴](#21-category-of-resources--资源范畴)
    - [2.2 Resource Object Properties / 资源对象性质](#22-resource-object-properties--资源对象性质)
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
    - [6.1 Human Resource Example / 人力资源例子](#61-human-resource-example--人力资源例子)
    - [6.2 Financial Resource Example / 财务资源例子](#62-financial-resource-example--财务资源例子)
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

- **所属层**：**核心模型层**（对应 docs/02-project-management；资源管理模型）
- **转换关系**：**Resource Objects** 作为**状态转换**的实体（资源分配、调度、优化作为状态转换）；与 03-资源管理概念、Category/02-Morphisms/09-Resource-Morphisms、Category/04-Functors/02-Resource-Management-Functor 对应。

**与 docs 的公式对应**：资源四元组 $\mathcal{R}=(H,M,T,F)$、分配函数 $\mathrm{allocate}:\mathcal{T}\times\mathcal{R}\to\mathbb{R}^+$、约束 $C=(R,L,U)$、优化 $V(i,r)=\max_{0\le x\le r}\{v_i(x)+V(i-1,r-x)\}$ 见 `docs/02-project-management/resource-models`。

---

## 1. Overview / 概述

**English / 英文**:

Resource objects represent project resources (human, material, technical, financial) in the category $\mathbf{Resource}$. They capture resource allocation, scheduling, and optimization in project management. This document provides a category-theoretic perspective on resource objects, aligning with PMBOK 7th Edition and ISO 21500 standards.

**中文**:

资源对象表示项目范畴 $\mathbf{Resource}$ 中的项目资源（人力资源、物质资源、技术资源、财务资源）。它们捕捉项目管理中的资源分配、调度和优化。本文档从范畴论视角提供资源对象的定义，对齐 PMBOK 第7版和 ISO 21500 标准。

**Key Insights / 关键洞察**:

- **Resource Types / 资源类型**: Human, Material, Technical, Financial / 人力资源、物质资源、技术资源、财务资源
- **Allocation / 分配**: Resources are allocated to projects and tasks / 资源被分配给项目和任务
- **Scheduling / 调度**: Resources are scheduled over time / 资源在时间上被调度
- **Optimization / 优化**: Resource allocation can be optimized / 资源分配可以被优化

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Category of Resources / 资源范畴

**Definition 2.1** (Category $\mathbf{Resource}$)

The category $\mathbf{Resource}$ is defined as follows:

- **Objects / 对象**: Resources $Res = (Type, Capacity, Availability, Cost)$ where:
  - $Type \in \{Human, Material, Technical, Financial\}$ - resource type
  - $Capacity \in \mathbb{R}^+$ - resource capacity
  - $Availability: T \to [0,1]$ - availability function over time
  - $Cost: \mathbb{R}^+ \to \mathbb{R}^+$ - cost function

- **Morphisms / 态射**: Resource transformations $f: Res_1 \to Res_2$ representing resource modifications

- **Composition / 复合**: Composition of resource transformations $(g \circ f): Res_1 \to Res_3$

- **Identity / 恒等**: Identity transformation $\text{id}_{Res}: Res \to Res$

### 2.2 Resource Object Properties / 资源对象性质

**Axiom 2.1** (Resource Capacity Non-negativity)

For any resource $Res = (Type, Capacity, Availability, Cost)$:
$$Capacity \geq 0$$

**Axiom 2.2** (Resource Availability Boundedness)

For any resource $Res$ and time $t \in T$:
$$0 \leq Availability(t) \leq 1$$

**Axiom 2.3** (Resource Cost Non-negativity)

For any resource $Res$ and quantity $q \geq 0$:
$$Cost(q) \geq 0$$

---

## 3. Formal Definition / 形式化定义

### 3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义

**Definition 3.1** (Project Resources - PMBOK 7th Edition)

Project resources include team members, facilities, equipment, materials, supplies, and other resources needed to complete project work. In our formalization:

$$Resources(P) = \{Res_1, Res_2, \ldots, Res_n\}$$

where each $Res_i$ is an object in $\mathbf{Resource}$.

**Resource Categories / 资源类别**:

- **Human Resources / 人力资源**: $Res_{human} = (Human, Capacity_{human}, Availability_{human}, Cost_{human})$
- **Material Resources / 物质资源**: $Res_{material} = (Material, Capacity_{material}, Availability_{material}, Cost_{material})$
- **Technical Resources / 技术资源**: $Res_{technical} = (Technical, Capacity_{technical}, Availability_{technical}, Cost_{technical})$
- **Financial Resources / 财务资源**: $Res_{financial} = (Financial, Capacity_{financial}, Availability_{financial}, Cost_{financial})$

### 3.2 ISO 21500 Standard Definition / ISO 21500 标准定义

**Definition 3.2** (Resource Management - ISO 21500:2012)

Resource management includes processes to identify, acquire, and manage resources needed for the project. In our category-theoretic framework:

$$R: \mathbf{Project} \to \mathbf{Resource}$$

where $R$ is the resource management functor.

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (Resource Allocation Function)

Resource allocation is a function:
$$alloc: \mathbf{Project} \times \mathbf{Resource} \times T \to \mathbb{R}^+$$

satisfying:
$$\forall P, Res, t: alloc(P, Res, t) \leq Capacity(Res) \cdot Availability(Res, t)$$

**Property 4.2** (Resource Conservation)

Total resource allocation does not exceed available resources:
$$\sum_{P \in \mathbf{Project}} alloc(P, Res, t) \leq Capacity(Res) \cdot Availability(Res, t)$$

**Property 4.3** (Resource Scheduling)

Resources can be scheduled over time:
$$schedule: \mathbf{Resource} \times T \to \mathbf{Task}$$

### 4.2 Category-Theoretic Properties / 范畴论性质

**Property 4.4** (Resource Management Functor)

Resource management is a functor:
$$R: \mathbf{Project} \to \mathbf{Resource}$$

**Property 4.5** (Resource Composition)

Resources compose under allocation:
$$alloc(P, Res_1 \otimes Res_2, t) = alloc(P, Res_1, t) + alloc(P, Res_2, t)$$

where $\otimes$ denotes resource composition.

---

## 5. Relations / 关系

### 5.1 Relations to Other Objects / 与其他对象的关系

**Relation 5.1** (Resource → Project)

Every resource is allocated to projects:
$$R^{-1}: \mathbf{Resource} \to 2^{\mathbf{Project}}$$

**Relation 5.2** (Resource → Lifecycle)

Resources are allocated to lifecycle phases:
$$R \circ L: \mathbf{Project} \to \mathbf{Resource} \times \mathbf{Phase}$$

**Relation 5.3** (Resource → Risk)

Resource constraints create risks:
$$Risk \circ R: \mathbf{Project} \to \mathbf{Risk}$$

**Relation 5.4** (Resource → Quality)

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

### 6.1 Human Resource Example / 人力资源例子

**Example 6.1** (Development Team)

Consider a development team resource:

$$Res_{dev} = (Human, 10, Availability_{dev}, Cost_{dev})$$

where:

- $Capacity = 10$ developers
- $Availability(t)$ - availability schedule
- $Cost(q)$ - cost for $q$ developers

### 6.2 Financial Resource Example / 财务资源例子

**Example 6.2** (Project Budget)

Consider a project budget resource:

$$Res_{budget} = (Financial, \$500k, Availability_{budget}, Cost_{budget})$$

where:

- $Capacity = \$500k$ total budget
- $Availability(t)$ - budget release schedule
- $Cost(q)$ - cost function

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Resource Allocation**: Allocating resources to projects and tasks
- **Resource Scheduling**: Scheduling resources over time
- **Resource Optimization**: Optimizing resource allocation
- **Resource Monitoring**: Monitoring resource utilization

### 7.2 Category-Theoretic Applications / 范畴论应用

- **Resource Composition**: Composing resources using morphisms
- **Resource Transformation**: Transforming resources using functors
- **Resource Relationships**: Understanding relationships using natural transformations

### 7.3 直观解释与一例 / Intuitive Explanation with One Example

**Example 7.3** (Resource as a Pool Mapped from Projects / 资源即由项目映射的池)

资源对象 $Res=(Pool,Cap,Alloc)$ 是**资源池**：$Pool$ 为可分配单元集合，$Cap$ 为容量，$Alloc$ 为分配关系。函子 $R:\mathbf{Project}\to\mathbf{Resource}$ 从项目 $P$ 抽出 $R(P)$：人力、预算、设备等。例：$R(P_{sw})=\{5\,\text{dev}, 2\,\text{tester}, 100k\,\$\}$；态射 $Alloc: R(P)\to R(P')$ 表示在两项目间调配资源。与生命周期通过 $\alpha: L\Rightarrow R$ 关联：每一阶段对应一资源剖面。

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
- [Lifecycle Objects](08-Lifecycle-Objects.md)
- [Resource Morphisms](../../02-Morphisms/09-Resource-Morphisms.md)
- [Resource Management Functor](../../04-Functors/02-Resource-Management-Functor.md)
- **docs**：`docs/02-project-management/resource-models`（$\mathcal{R}=(H,M,T,F)$、allocate、约束 $C=(R,L,U)$、优化；与 0. 公式对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
