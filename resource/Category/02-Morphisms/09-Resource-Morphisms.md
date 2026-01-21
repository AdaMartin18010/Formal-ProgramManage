# Resource Management Morphisms / 资源管理态射

## 📋 Table of Contents / 目录

- [Resource Management Morphisms / 资源管理态射](#resource-management-morphisms--资源管理态射)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Resource Allocation Morphism / 资源分配态射](#21-resource-allocation-morphism--资源分配态射)
    - [2.2 Resource Scheduling Morphism / 资源调度态射](#22-resource-scheduling-morphism--资源调度态射)
    - [2.3 Resource Optimization Morphism / 资源优化态射](#23-resource-optimization-morphism--资源优化态射)
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
    - [6.1 Resource Allocation Example / 资源分配例子](#61-resource-allocation-example--资源分配例子)
    - [6.2 Resource Optimization Example / 资源优化例子](#62-resource-optimization-example--资源优化例子)
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
- **转换关系**：**Resource Morphisms** = **状态转换**（资源分配、调度、优化作为状态转换 $\rightarrow$）；与 03-资源管理概念、Category/01-Objects/09-Resource-Objects、Category/04-Functors/02-Resource-Management-Functor 对应。

**与 docs 的公式对应**：$alloc(P,Res,t)=q$ 与 docs 的 $\mathrm{allocate}:\mathcal{T}\times\mathcal{R}\to\mathbb{R}^+$ 对应；$q\le Capacity(Res)\cdot Availability(Res,t)$ 与资源约束 $C=(R,L,U)$ 对应。见 `docs/02-project-management/resource-models`。

---

## 1. Overview / 概述

**English / 英文**:

Resource management morphisms represent resource allocation, scheduling, and optimization operations in the category $\mathbf{Resource}$. They capture how resources are allocated to projects, scheduled over time, and optimized for efficiency. This document provides a category-theoretic perspective on resource management morphisms, aligning with PMBOK 7th Edition and ISO 21500 standards.

**中文**:

资源管理态射表示资源范畴 $\mathbf{Resource}$ 中的资源分配、调度和优化操作。它们捕捉资源如何被分配给项目、在时间上被调度以及为效率而优化。本文档从范畴论视角提供资源管理态射的定义，对齐 PMBOK 第7版和 ISO 21500 标准。

**Key Insights / 关键洞察**:

- **Resource Allocation / 资源分配**: Morphisms $alloc: \mathbf{Project} \times \mathbf{Resource} \to \mathbf{Allocation}$ / 态射 $alloc: \mathbf{Project} \times \mathbf{Resource} \to \mathbf{Allocation}$
- **Resource Scheduling / 资源调度**: Morphisms $schedule: \mathbf{Resource} \times T \to \mathbf{Task}$ / 态射 $schedule: \mathbf{Resource} \times T \to \mathbf{Task}$
- **Resource Optimization / 资源优化**: Morphisms $optimize: \mathbf{Allocation} \to \mathbf{Allocation}$ / 态射 $optimize: \mathbf{Allocation} \to \mathbf{Allocation}$
- **Resource Monitoring / 资源监控**: Morphisms $monitor: \mathbf{Resource} \to \mathbf{ResourceState}$ / 态射 $monitor: \mathbf{Resource} \to \mathbf{ResourceState}$

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Resource Allocation Morphism / 资源分配态射

**Definition 2.1** (Resource Allocation Morphism)

A resource allocation morphism $alloc: \mathbf{Project} \times \mathbf{Resource} \to \mathbf{Allocation}$ is defined as:

$$alloc(P, Res, t) = q$$

where:

- $P \in \mathbf{Project}$ - project
- $Res \in \mathbf{Resource}$ - resource
- $t \in T$ - time point
- $q \in \mathbb{R}^+$ - allocated quantity

subject to the constraint:
$$q \leq Capacity(Res) \cdot Availability(Res, t)$$

### 2.2 Resource Scheduling Morphism / 资源调度态射

**Definition 2.2** (Resource Scheduling Morphism)

A resource scheduling morphism $schedule: \mathbf{Resource} \times T \to \mathbf{Task}$ assigns resources to tasks over time:

$$schedule(Res, t) = Task$$

where $Task$ is the task assigned to resource $Res$ at time $t$.

### 2.3 Resource Optimization Morphism / 资源优化态射

**Definition 2.3** (Resource Optimization Morphism)

A resource optimization morphism $optimize: \mathbf{Allocation} \to \mathbf{Allocation}$ optimizes resource allocation:

$$optimize(Alloc) = Alloc^*$$

where $Alloc^*$ is the optimized allocation maximizing some objective function.

---

## 3. Formal Definition / 形式化定义

### 3.1 PMBOK 7th Edition Definition / PMBOK 第7版定义

**Definition 3.1** (Resource Allocation - PMBOK 7th Edition)

Resource allocation assigns resources to project activities. In our formalization:

$$alloc: \mathbf{Project} \times \mathbf{Resource} \times T \to \mathbb{R}^+$$

satisfying resource constraints and project requirements.

**Allocation Types / 分配类型**:

- **Level Allocation / 平衡分配**: Smooth resource allocation over time
- **Peak Allocation / 峰值分配**: Maximum resource allocation at peaks
- **Optimized Allocation / 优化分配**: Optimal allocation based on objectives

### 3.2 ISO 21500 Standard Definition / ISO 21500 标准定义

**Definition 3.2** (Resource Management - ISO 21500:2012)

Resource management includes processes to identify, acquire, and manage resources. In our category-theoretic framework:

$$R: \mathbf{Project} \to \mathbf{Resource}$$

where $R$ is the resource management functor, and morphisms represent resource operations.

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (Allocation Feasibility)

Resource allocation is feasible:
$$\forall P, Res, t: alloc(P, Res, t) \leq Capacity(Res) \cdot Availability(Res, t)$$

**Property 4.2** (Allocation Conservation)

Total allocation does not exceed capacity:
$$\sum_{P \in \mathbf{Project}} alloc(P, Res, t) \leq Capacity(Res)$$

**Property 4.3** (Scheduling Consistency)

Resource scheduling is consistent:
$$\forall Res, t_1, t_2: schedule(Res, t_1) \cap schedule(Res, t_2) = \emptyset \text{ if } t_1 \neq t_2$$

### 4.2 Category-Theoretic Properties / 范畴论性质

**Property 4.4** (Allocation Composition)

Resource allocations compose:
$$alloc(P, Res_1 \otimes Res_2, t) = alloc(P, Res_1, t) + alloc(P, Res_2, t)$$

**Property 4.5** (Optimization Idempotence)

Resource optimization is idempotent:
$$optimize \circ optimize = optimize$$

---

## 5. Relations / 关系

### 5.1 Relations to Other Morphisms / 与其他态射的关系

**Relation 5.1** (Resource → Lifecycle)

Resource allocation follows lifecycle phases:
$$alloc \circ L: \mathbf{Project} \to \mathbf{Allocation} \times \mathbf{Phase}$$

**Relation 5.2** (Resource → Risk)

Resource constraints create risks:
$$Risk \circ alloc: \mathbf{Project} \times \mathbf{Resource} \to \mathbf{Risk}$$

**Relation 5.3** (Resource → Quality)

Resource quality affects project quality:
$$Q \circ alloc: \mathbf{Project} \times \mathbf{Resource} \to \mathbf{Quality}$$

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

### 6.1 Resource Allocation Example / 资源分配例子

**Example 6.1** (Developer Allocation)

Consider allocating developers to a project:

$$alloc(P_{sw}, Res_{dev}, t) = 5$$

where:

- $P_{sw}$ - software project
- $Res_{dev}$ - developer resource (capacity 10)
- $t$ - time point
- $5$ - allocated developers

### 6.2 Resource Optimization Example / 资源优化例子

**Example 6.2** (Budget Optimization)

Consider optimizing budget allocation:

$$optimize(Alloc_{budget}) = Alloc_{budget}^*$$

where $Alloc_{budget}^*$ minimizes cost while meeting project requirements.

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Resource Allocation**: Allocating resources to projects and tasks
- **Resource Scheduling**: Scheduling resources over time
- **Resource Optimization**: Optimizing resource utilization
- **Resource Monitoring**: Monitoring resource usage

### 7.2 Category-Theoretic Applications / 范畴论应用

- **Allocation Composition**: Composing resource allocations
- **Optimization Transformation**: Transforming allocations using optimization
- **Resource Relationships**: Understanding relationships using natural transformations

### 7.3 直观解释与一例 / Intuitive Explanation with One Example

**Example 7.3** (Allocation as a Morphism between Resource Slices / 分配即资源切片间的态射)

资源态射 $Alloc: R(P)\to R(P')$、$Schedule: R(P)\times T\to R(P)$、$optimize: Alloc\to Alloc^*$ 分别表示**项目间调配**、**按时段调度**、**在约束下优化**。例：$Alloc_{budget}(P_{sw})$ 把 100k 分到设计/开发/测试；$optimize(Alloc_{budget})$ 在满足交付前提下最小化成本。与 [09-Resource-Objects](../../01-Objects/09-Resource-Objects.md)、[02-Resource-Management-Functor](../../04-Functors/02-Resource-Management-Functor.md) 一致：函子 $R$ 抽出 $R(P)$，态射在其上运算。

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
- [Project Objects](../../01-Objects/01-Project-Objects.md)
- [Resource Management Functor](../../04-Functors/02-Resource-Management-Functor.md)
- [Lifecycle-Resource Natural Transformation](../../05-Natural-Transformations/01-Lifecycle-Resource-Natural-Transformation.md)
- **docs**：`docs/02-project-management/resource-models`（$\mathcal{R}$、allocate；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
