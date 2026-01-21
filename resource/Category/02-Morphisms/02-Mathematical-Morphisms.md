# Mathematical Morphisms / 数学态射

## 📋 Table of Contents / 目录

- [Mathematical Morphisms / 数学态射](#mathematical-morphisms--数学态射)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Set Functions / 集合函数](#21-set-functions--集合函数)
    - [2.2 Graph Homomorphisms / 图同态](#22-graph-homomorphisms--图同态)
    - [2.3 Probability Measures / 概率测度](#23-probability-measures--概率测度)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 Set Theory / 集合论](#31-set-theory--集合论)
    - [3.2 Graph Theory / 图论](#32-graph-theory--图论)
    - [3.3 Probability Theory / 概率论](#33-probability-theory--概率论)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Function Properties / 函数性质](#41-function-properties--函数性质)
    - [4.2 Graph Properties / 图性质](#42-graph-properties--图性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Project Management / 与项目管理的关系](#51-relations-to-project-management--与项目管理的关系)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Set Function Example / 集合函数例子](#61-set-function-example--集合函数例子)
    - [6.2 Graph Homomorphism Example / 图同态例子](#62-graph-homomorphism-example--图同态例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Project Management Applications / 项目管理应用](#71-project-management-applications--项目管理应用)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Mathematics / 数学](#81-mathematics--数学)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**基础理论层**（对应 docs/01-foundations；数学基础）
- **转换关系**：**Mathematical Morphisms** = **状态转换**（集合映射、图同态、概率测度作为状态转换的基础）；与 Category/01-Objects/02-Mathematical-Objects 对应。

---

## 1. Overview / 概述

**English / 英文**:

Mathematical morphisms represent set mappings, graph homomorphisms, and probability measures in the categories $\mathbf{Set}$, $\mathbf{Graph}$, and $\mathbf{Prob}$. They capture mathematical transformations used in project management. This document provides a category-theoretic perspective on mathematical morphisms.

**中文**:

数学态射表示集合映射、图同态和概率测度，属于范畴 $\mathbf{Set}$、$\mathbf{Graph}$ 和 $\mathbf{Prob}$。它们捕捉项目管理中使用的数学变换。本文档从范畴论视角提供数学态射的定义。

**Key Insights / 关键洞察**:

- **Set Functions / 集合函数**: $f: A \to B$ - functions between sets / 集合之间的函数
- **Graph Homomorphisms / 图同态**: $\phi: G_1 \to G_2$ - graph homomorphisms / 图同态
- **Probability Measures / 概率测度**: $P: \mathcal{F} \to [0,1]$ - probability measures / 概率测度
- **Project Mapping / 项目映射**: Mathematical morphisms model project transformations / 数学态射建模项目变换

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Set Functions / 集合函数

**Definition 2.1** (Set Function Morphism)

A set function $f: A \to B$ is a morphism in $\mathbf{Set}$ mapping elements:
$$f(a) = b \text{ where } a \in A, b \in B$$

### 2.2 Graph Homomorphisms / 图同态

**Definition 2.2** (Graph Homomorphism)

A graph homomorphism $\phi: G_1 \to G_2$ preserves graph structure:
$$\phi: V_1 \to V_2 \text{ such that } (u,v) \in E_1 \Rightarrow (\phi(u), \phi(v)) \in E_2$$

### 2.3 Probability Measures / 概率测度

**Definition 2.3** (Probability Measure)

A probability measure $P: \mathcal{F} \to [0,1]$ assigns probabilities:
$$P(\Omega) = 1, P(\emptyset) = 0, P(\bigcup_i A_i) = \sum_i P(A_i)$$

---

## 3. Formal Definition / 形式化定义

### 3.1 Set Theory / 集合论

**Definition 3.1** (Set Function)

A function $f: A \to B$ maps elements. In project management:

- **Project Mapping / 项目映射**: $f: \mathcal{P}_1 \to \mathcal{P}_2$ - project transformation
- **Resource Mapping / 资源映射**: $f: R_1 \to R_2$ - resource transformation

### 3.2 Graph Theory / 图论

**Definition 3.2** (Graph Homomorphism)

A graph homomorphism preserves structure. In project management:

- **Network Transformation / 网络变换**: Transforming project networks
- **Dependency Preservation / 依赖保持**: Preserving dependencies

### 3.3 Probability Theory / 概率论

**Definition 3.3** (Probability Measure)

A probability measure models uncertainty. In project management:

- **Risk Probability / 风险概率**: $P_{risk}$ - risk probabilities
- **Schedule Probability / 进度概率**: $P_{schedule}$ - schedule probabilities

---

## 4. Properties / 性质

### 4.1 Function Properties / 函数性质

**Property 4.1** (Function Composition)

Functions compose:
$$(g \circ f)(a) = g(f(a))$$

**Property 4.2** (Function Identity)

Identity functions exist:
$$\text{id}_A(a) = a$$

### 4.2 Graph Properties / 图性质

**Property 4.3** (Homomorphism Composition)

Graph homomorphisms compose.

**Property 4.4** (Homomorphism Identity)

Identity homomorphisms exist.

---

## 5. Relations / 关系

### 5.1 Relations to Project Management / 与项目管理的关系

**Relation 5.1** (Set Functions → Project Transformations)

Set functions model project transformations.

**Relation 5.2** (Graph Homomorphisms → Network Transformations)

Graph homomorphisms model network transformations.

**Relation 5.3** (Probability Measures → Risk Modeling)

Probability measures model risks.

---

## 6. Examples / 例子

### 6.1 Set Function Example / 集合函数例子

**Example 6.1** (Project Transformation)

Consider project transformation:

$$f: P_1 \to P_2$$

transforming project $P_1$ to $P_2$.

### 6.2 Graph Homomorphism Example / 图同态例子

**Example 6.2** (Network Transformation)

Consider network transformation:

$$\phi: G_{old} \to G_{new}$$

transforming project network.

---

## 7. Applications / 应用

### 7.1 Project Management Applications / 项目管理应用

- **Project Transformations**: Transforming projects using functions
- **Network Analysis**: Analyzing networks using homomorphisms
- **Risk Modeling**: Modeling risks using probability measures
- **Mathematical Modeling**: Mathematical project models

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

- [Mathematical Objects](../../01-Objects/02-Mathematical-Objects.md)
- [Project Objects](../../01-Objects/01-Project-Objects.md)
- **docs**：`docs/01-foundations`（集合、图、概率等；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
