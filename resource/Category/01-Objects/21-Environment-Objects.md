# Environment Objects / 环境对象

## 📋 Table of Contents / 目录

- [Environment Objects / 环境对象](#environment-objects--环境对象)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Category of Environments / 环境范畴](#21-category-of-environments--环境范畴)
    - [2.2 Environment Object Properties / 环境对象性质](#22-environment-object-properties--环境对象性质)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 Harper Definition / Harper 定义](#31-harper-definition--harper-定义)
    - [3.2 Project Management Mapping / 项目管理映射](#32-project-management-mapping--项目管理映射)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Basic Properties / 基本性质](#41-basic-properties--基本性质)
    - [4.2 Category-Theoretic Properties / 范畴论性质](#42-category-theoretic-properties--范畴论性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Objects / 与其他对象的关系](#51-relations-to-other-objects--与其他对象的关系)
    - [5.2 Natural Transformations / 自然变换](#52-natural-transformations--自然变换)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Programming Language Example / 编程语言例子](#61-programming-language-example--编程语言例子)
    - [6.2 Project Management Example / 项目管理例子](#62-project-management-example--项目管理例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Programming Language Applications / 编程语言应用](#71-programming-language-applications--编程语言应用)
    - [7.2 Project Management Applications / 项目管理应用](#72-project-management-applications--项目管理应用)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Programming Language Theory / 编程语言理论](#81-programming-language-theory--编程语言理论)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**验证理论层（支撑）**（对应 docs/03-formal-verification、06-ci-verification；环境对象支撑形式化验证）
- **转换关系**：**Environment Objects** 作为**模型转换**的实体（环境构建、环境扩展作为模型构建方法）；与 06-编程语言理论概念/04-变量与环境、Category/04-Functors/06-Environment-Functors、Category/05-Natural-Transformations/05-Type-Environment-Natural-Transformation 对应。

---

## 1. Overview / 概述

**English / 英文**:

Environment objects represent variable environments and contexts in the category $\mathbf{Env}$. They capture variable bindings, scopes, and context information. This document provides a category-theoretic perspective on environment objects, aligning with authoritative resources from Harper, Pierce, and other programming language theory experts.

**中文**:

环境对象表示变量环境和上下文，属于范畴 $\mathbf{Env}$。它们捕捉变量绑定、作用域和上下文信息。本文档从范畴论视角提供环境对象的定义，对齐 Harper、Pierce 等编程语言理论权威资源。

**Key Insights / 关键洞察**:

- **Variable Environment / 变量环境**: $\Gamma = \{x_1: A_1, \ldots, x_n: A_n\}$ / 变量环境
- **Environment Extension / 环境扩展**: $\Gamma, x: A$ - adding variable / 添加变量
- **Scope / 作用域**: Variable scope management / 变量作用域管理
- **Context / 上下文**: Project context, resource context / 项目上下文、资源上下文

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Category of Environments / 环境范畴

**Definition 2.1** (Category $\mathbf{Env}$)

The category $\mathbf{Env}$ is defined as follows:

- **Objects / 对象**: Environments $\Gamma = \{x_1: A_1, \ldots, x_n: A_n\}$ where $x_i$ are variables and $A_i$ are types
- **Morphisms / 态射**: Environment extensions $\Gamma \to \Gamma, x: A$
- **Composition / 复合**: Composition of environment extensions
- **Identity / 恒等**: Identity extension $\text{id}_\Gamma: \Gamma \to \Gamma$

### 2.2 Environment Object Properties / 环境对象性质

**Axiom 2.1** (Environment Non-emptiness)

Environments can be empty or non-empty:
$$\Gamma = \emptyset \text{ or } \Gamma \neq \emptyset$$

**Axiom 2.2** (Variable Uniqueness)

Variables in environment are unique:
$$\forall x_i, x_j \in \Gamma: i \neq j \Rightarrow x_i \neq x_j$$

---

## 3. Formal Definition / 形式化定义

### 3.1 Harper Definition / Harper 定义

**Definition 3.1** (Environment - Harper)

An environment is a finite mapping from variables to types. In our category-theoretic framework:

$$\Gamma \in \text{Ob}(\mathbf{Env})$$

**Environment Operations / 环境操作**:

- **Lookup / 查找**: $\Gamma(x)$ - type of variable $x$
- **Extension / 扩展**: $\Gamma, x: A$ - add variable $x$ with type $A$
- **Restriction / 限制**: $\Gamma \setminus x$ - remove variable $x$

### 3.2 Project Management Mapping / 项目管理映射

**Definition 3.2** (Project Environment)

In project management, environments represent contexts:

- **Project Context / 项目上下文**: $\Gamma_{project} = \{project: Project, phase: Phase, \ldots\}$
- **Resource Context / 资源上下文**: $\Gamma_{resource} = \{resources: ResourceSet, allocation: Allocation, \ldots\}$

---

## 4. Properties / 性质

### 4.1 Basic Properties / 基本性质

**Property 4.1** (Environment Extension)

Environments can be extended:
$$\Gamma, x: A \in \mathbf{Env} \text{ if } \Gamma \in \mathbf{Env} \text{ and } x \notin \text{dom}(\Gamma)$$

**Property 4.2** (Environment Lookup)

Variable lookup is defined:
$$\Gamma(x) = A \text{ if } x: A \in \Gamma$$

**Property 4.3** (Environment Composition)

Environments compose:
$$(\Gamma_1, \Gamma_2) = \Gamma_1 \cup \Gamma_2 \text{ if } \text{dom}(\Gamma_1) \cap \text{dom}(\Gamma_2) = \emptyset$$

### 4.2 Category-Theoretic Properties / 范畴论性质

**Property 4.4** (Environment Functor)

Environment management is a functor:
$$Env: \mathbf{Type} \to \mathbf{Env}$$

**Property 4.5** (Environment Morphism Composition)

Environment morphisms compose associatively.

---

## 5. Relations / 关系

### 5.1 Relations to Other Objects / 与其他对象的关系

**Relation 5.1** (Environment → Type)

Environments contain type information:
$$Type \circ Env: \mathbf{Env} \to \mathbf{Type}$$

**Relation 5.2** (Environment → Project Management)

Environments map to project contexts:
$$ProjectContext: \mathbf{Env} \to \mathbf{ProjectContext}$$

**Relation 5.3** (Environment → Scope)

Environments define scopes:
$$Scope: \mathbf{Env} \to \mathbf{Scope}$$

### 5.2 Natural Transformations / 自然变换

**Natural Transformation 5.1** (Type-Environment)

There exists a natural transformation $\eta: Type \Rightarrow Env$:
$$\eta_T: Type(T) \to Env(T)$$

connecting types to environments.

---

## 6. Examples / 例子

### 6.1 Programming Language Example / 编程语言例子

**Example 6.1** (Variable Environment)

Consider a variable environment:

$$\Gamma = \{x: Int, y: String, f: Int \to Bool\}$$

with typed variables.

### 6.2 Project Management Example / 项目管理例子

**Example 6.2** (Project Context)

Consider a project context:

$$\Gamma_{project} = \{project: P_{sw}, phase: Ph_{exec}, resources: R_{dev}\}$$

with project information.

---

## 7. Applications / 应用

### 7.1 Programming Language Applications / 编程语言应用

- **Variable Binding**: Binding variables in environments
- **Scope Management**: Managing variable scopes
- **Type Checking**: Type checking using environments
- **Context Management**: Managing program contexts

### 7.2 Project Management Applications / 项目管理应用

- **Project Context**: Managing project contexts
- **Resource Context**: Managing resource contexts
- **Context Extension**: Extending project contexts
- **Context Lookup**: Looking up context information

---

## 8. References / 参考文献

### 8.1 Programming Language Theory / 编程语言理论

1. Harper, R. (2016). *Practical Foundations for Programming Languages* (2nd ed.). Cambridge University Press.
2. Pierce, B. C. (2002). *Types and Programming Languages*. MIT Press.

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.
2. Awodey, S. (2010). *Category Theory* (2nd ed.). Oxford University Press.

### 8.3 Related Files / 相关文件

- [Type Objects](20-Type-Objects.md)
- [Scope Objects](22-Scope-Objects.md)
- [Environment Functors](../../04-Functors/06-Environment-Functors.md)
- **docs**：`docs/03-formal-verification`、`docs/06-ci-verification`（环境、变量绑定；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
