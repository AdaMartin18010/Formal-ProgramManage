# Mathematical Objects / 数学对象

## 📋 Table of Contents / 目录

- [Mathematical Objects / 数学对象](#mathematical-objects--数学对象)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Set Category / 集合范畴](#21-set-category--集合范畴)
    - [2.2 Graph Category / 图范畴](#22-graph-category--图范畴)
    - [2.3 Probability Category / 概率范畴](#23-probability-category--概率范畴)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 Set Theory / 集合论](#31-set-theory--集合论)
    - [3.2 Graph Theory / 图论](#32-graph-theory--图论)
    - [3.3 Probability Theory / 概率论](#33-probability-theory--概率论)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Set Properties / 集合性质](#41-set-properties--集合性质)
    - [4.2 Graph Properties / 图性质](#42-graph-properties--图性质)
    - [4.3 Probability Properties / 概率性质](#43-probability-properties--概率性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Project Management / 与项目管理的关系](#51-relations-to-project-management--与项目管理的关系)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Set Example / 集合例子](#61-set-example--集合例子)
    - [6.2 Graph Example / 图例子](#62-graph-example--图例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Project Management Applications / 项目管理应用](#71-project-management-applications--项目管理应用)
    - [7.3 直观解释与一例 / Intuitive Explanation with One Example](#73-直观解释与一例--intuitive-explanation-with-one-example)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Mathematics / 数学](#81-mathematics--数学)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**基础理论层**（对应 docs/01-foundations；数学基础）
- **转换关系**：**Mathematical Objects** 作为**状态转换**的基础（集合、图、概率空间作为状态空间的基础结构）；与 01-项目管理基础、Category/01-Objects/01-Project-Objects 对应。

---

## 1. Overview / 概述

**English / 英文**:

Mathematical objects represent mathematical structures (sets, graphs, probability spaces) used in project management in the categories $\mathbf{Set}$, $\mathbf{Graph}$, $\mathbf{Prob}$. They provide the mathematical foundation for project management models. This document provides a category-theoretic perspective on mathematical objects, aligning with authoritative mathematical resources.

**中文**:

数学对象表示项目管理中使用的数学结构（集合、图、概率空间），属于范畴 $\mathbf{Set}$、$\mathbf{Graph}$、$\mathbf{Prob}$。它们为项目管理模型提供数学基础。本文档从范畴论视角提供数学对象的定义，对齐权威数学资源。

**Key Insights / 关键洞察**:

- **Set Category / 集合范畴**: $\mathbf{Set}$ - sets and functions / 集合和函数
- **Graph Category / 图范畴**: $\mathbf{Graph}$ - graphs and graph homomorphisms / 图和图同态
- **Probability Category / 概率范畴**: $\mathbf{Prob}$ - probability spaces and measurable functions / 概率空间和可测函数
- **Project Mapping / 项目映射**: Mathematical structures model project structures / 数学结构建模项目结构

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Set Category / 集合范畴

**Definition 2.1** (Category $\mathbf{Set}$)

The category $\mathbf{Set}$ consists of:

- **Objects / 对象**: Sets $A, B, C, \ldots$
- **Morphisms / 态射**: Functions $f: A \to B$
- **Composition / 复合**: Function composition
- **Identity / 恒等**: Identity functions

### 2.2 Graph Category / 图范畴

**Definition 2.2** (Category $\mathbf{Graph}$)

The category $\mathbf{Graph}$ consists of:

- **Objects / 对象**: Graphs $G = (V, E)$
- **Morphisms / 态射**: Graph homomorphisms $\phi: G_1 \to G_2$
- **Composition / 复合**: Composition of homomorphisms
- **Identity / 恒等**: Identity homomorphisms

### 2.3 Probability Category / 概率范畴

**Definition 2.3** (Category $\mathbf{Prob}$)

The category $\mathbf{Prob}$ consists of:

- **Objects / 对象**: Probability spaces $(\Omega, \mathcal{F}, P)$
- **Morphisms / 态射**: Measurable functions $f: (\Omega_1, \mathcal{F}_1, P_1) \to (\Omega_2, \mathcal{F}_2, P_2)$
- **Composition / 复合**: Composition of measurable functions
- **Identity / 恒等**: Identity measurable functions

---

## 3. Formal Definition / 形式化定义

### 3.1 Set Theory / 集合论

**Definition 3.1** (Set)

A set is a collection of distinct objects. In project management:

- **Project Set / 项目集合**: $\mathcal{P} = \{P_1, P_2, \ldots, P_n\}$
- **Resource Set / 资源集合**: $R = \{r_1, r_2, \ldots, r_k\}$
- **Task Set / 任务集合**: $T = \{t_1, t_2, \ldots, t_m\}$

### 3.2 Graph Theory / 图论

**Definition 3.2** (Graph)

A graph $G = (V, E)$ consists of vertices and edges. In project management:

- **Project Network Graph / 项目网络图**: $G_{project} = (Tasks, Dependencies)$
- **Resource Dependency Graph / 资源依赖图**: $G_{resource} = (Resources, Dependencies)$

### 3.3 Probability Theory / 概率论

**Definition 3.3** (Probability Space)

A probability space $(\Omega, \mathcal{F}, P)$ models uncertainty. In project management:

- **Risk Probability Space / 风险概率空间**: $(\Omega_{risk}, \mathcal{F}_{risk}, P_{risk})$
- **Schedule Probability Space / 进度概率空间**: $(\Omega_{schedule}, \mathcal{F}_{schedule}, P_{schedule})$

---

## 4. Properties / 性质

### 4.1 Set Properties / 集合性质

**Property 4.1** (Set Cardinality)

Sets have cardinality:
$$|A| \in \mathbb{N} \cup \{\infty\}$$

**Property 4.2** (Set Operations)

Sets support operations:

- **Union / 并集**: $A \cup B$
- **Intersection / 交集**: $A \cap B$
- **Difference / 差集**: $A \setminus B$

### 4.2 Graph Properties / 图性质

**Property 4.3** (Graph Connectivity)

Graphs can be connected or disconnected.

**Property 4.4** (Graph Acyclicity)

Graphs can be acyclic or cyclic.

### 4.3 Probability Properties / 概率性质

**Property 4.5** (Probability Measure)

Probability measures satisfy:
$$P(\Omega) = 1, P(\emptyset) = 0$$

---

## 5. Relations / 关系

### 5.1 Relations to Project Management / 与项目管理的关系

**Relation 5.1** (Set → Project)

Projects are sets:
$$P \in \mathbf{Set}$$

**Relation 5.2** (Graph → Project Network)

Project networks are graphs:
$$ProjectNetwork \in \mathbf{Graph}$$

**Relation 5.3** (Probability → Risk)

Risks use probability:
$$Risk \in \mathbf{Prob}$$

---

## 6. Examples / 例子

### 6.1 Set Example / 集合例子

**Example 6.1** (Project Set)

Consider project set:

$$\mathcal{P} = \{P_{sw}, P_{constr}, P_{research}\}$$

with multiple projects.

### 6.2 Graph Example / 图例子

**Example 6.2** (Task Dependency Graph)

Consider task dependency graph:

$$G_{tasks} = (\{t_1, t_2, t_3\}, \{(t_1, t_2), (t_2, t_3)\})$$

representing task dependencies.

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Set Operations**: Operations on project sets
- **Graph Analysis**: Analyzing project networks
- **Probability Modeling**: Modeling project uncertainties
- **Mathematical Modeling**: Mathematical project models

### 7.3 直观解释与一例 / Intuitive Explanation with One Example

**Example 7.3** (Mathematical Objects as Building Blocks / 数学对象即基础积木)

数学对象（集合、图、概率空间）可看作项目管理的**基础积木**：集合 $\mathcal{P} \in \mathbf{Set}$ 是项目的容器（如项目集合 $\{P_{sw}, P_{constr}\}$），图 $G \in \mathbf{Graph}$ 是任务依赖网络（如 $G_{tasks}=(\{t_1, t_2, t_3\}, \{(t_1, t_2), (t_2, t_3)\})$ 表示任务依赖关系），概率空间 $(\Omega, \mathcal{F}, P) \in \mathbf{Prob}$ 是风险与不确定性的模型。范畴 $\mathbf{Set}$、$\mathbf{Graph}$、$\mathbf{Prob}$ 中的态射表示集合映射、图同态、概率测度变换。例如项目网络 $G_{tasks}$：通过图分析可找出关键路径；函子 $Graph: \mathbf{Project} \to \mathbf{Graph}$ 将项目结构映射为依赖图，支持任务调度与资源优化。

---

## 8. References / 参考文献

### 8.1 Mathematics / 数学

1. Halmos, P. R. (1974). *Naive Set Theory*. Springer.
2. Diestel, R. (2017). *Graph Theory* (5th ed.). Springer.
3. Billingsley, P. (2012). *Probability and Measure* (Anniversary ed.). Wiley.

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.
2. Awodey, S. (2010). *Category Theory* (2nd ed.). Oxford University Press.

### 8.3 Related Files / 相关文件

- [Project Objects](01-Project-Objects.md)
- [Mathematical Morphisms](../../02-Morphisms/02-Mathematical-Morphisms.md)
- **docs**：`docs/01-foundations`（集合、图、概率等数学基础；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
