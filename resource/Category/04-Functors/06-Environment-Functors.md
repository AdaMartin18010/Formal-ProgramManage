# Environment Functors / 环境函子

## 📋 Table of Contents / 目录

- [Environment Functors / 环境函子](#environment-functors--环境函子)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Environment Functor Definition / 环境函子定义](#21-environment-functor-definition--环境函子定义)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 Harper Definition / Harper 定义](#31-harper-definition--harper-定义)
    - [3.2 Project Management Mapping / 项目管理映射](#32-project-management-mapping--项目管理映射)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Functor Properties / 函子性质](#41-functor-properties--函子性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Functors / 与其他函子的关系](#51-relations-to-other-functors--与其他函子的关系)
    - [5.2 Natural Transformations / 自然变换](#52-natural-transformations--自然变换)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Environment Construction Example / 环境构建例子](#61-environment-construction-example--环境构建例子)
    - [6.2 Project Context Example / 项目上下文例子](#62-project-context-example--项目上下文例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Programming Language Applications / 编程语言应用](#71-programming-language-applications--编程语言应用)
    - [7.2 Project Management Applications / 项目管理应用](#72-project-management-applications--项目管理应用)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Programming Language Theory / 编程语言理论](#81-programming-language-theory--编程语言理论)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**验证理论层（支撑）**（对应 docs/03-formal-verification、06-ci-verification；环境函子支撑形式化验证）
- **转换关系**：**Environment Functors** = **层次转换**（类型 → 环境的层间映射，支撑模型转换）；与 06-编程语言理论概念/04-变量与环境、Category/01-Objects/21-Environment-Objects、Category/05-Natural-Transformations/05-Type-Environment-Natural-Transformation 对应。
- **与 docs 的公式对应**：docs/03-formal-verification、06-ci-verification 的 $Env:\mathbf{Type}\to\mathbf{Env}$、$extend$、$lookup$、$\llbracket e\rrbracket:\mathbf{Env}\to\mathbf{Val}$（指称语义）与本文件的环境函子、环境扩展/查找 对应。

---

## 1. Overview / 概述

**English / 英文**:

Environment functors map types, programs, and contexts to variable environments in the category $\mathbf{Env}$. They capture how environments are constructed and managed. This document provides a category-theoretic perspective on environment functors, aligning with authoritative resources from Harper, Pierce, and other programming language theory experts.

**中文**:

环境函子将类型、程序和上下文映射到变量环境，属于范畴 $\mathbf{Env}$。它们捕捉环境如何构建和管理。本文档从范畴论视角提供环境函子的定义，对齐 Harper、Pierce 等编程语言理论权威资源。

**Key Insights / 关键洞察**:

- **Environment Construction / 环境构建**: $Env: \mathbf{Type} \to \mathbf{Env}$ / 环境构建函子
- **Environment Extension / 环境扩展**: Extending environments with variables / 用变量扩展环境
- **Environment Lookup / 环境查找**: Looking up variables in environments / 在环境中查找变量
- **Project Mapping / 项目映射**: Environments map to project contexts / 环境映射到项目上下文

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Environment Functor Definition / 环境函子定义

**Definition 2.1** (Environment Functor)

The environment functor $Env: \mathbf{Type} \to \mathbf{Env}$ maps:

- **Objects / 对象**: Types $A \in \mathbf{Type}$ to environments $Env(A) \in \mathbf{Env}$
- **Morphisms / 态射**: Type morphisms $f: A \to B$ to environment morphisms $Env(f): Env(A) \to Env(B)$

**Functor Properties / 函子性质**:

- **Identity Preservation / 恒等保持**: $Env(\text{id}_A) = \text{id}_{Env(A)}$
- **Composition Preservation / 复合保持**: $Env(g \circ f) = Env(g) \circ Env(f)$

---

## 3. Formal Definition / 形式化定义

### 3.1 Harper Definition / Harper 定义

**Definition 3.1** (Environment - Harper)

An environment is a finite mapping from variables to types. In our category-theoretic framework:

$$Env: \mathbf{Type} \to \mathbf{Env}$$

**Environment Operations / 环境操作**:

- **Environment Extension / 环境扩展**: $extend: Env \times (Var \times Type) \to Env$
- **Environment Lookup / 环境查找**: $lookup: Env \times Var \to Type$
- **Environment Restriction / 环境限制**: $restrict: Env \times VarSet \to Env$

### 3.2 Project Management Mapping / 项目管理映射

**Definition 3.2** (Project Environment)

In project management, environments represent contexts:

- **Project Context / 项目上下文**: $Env_{project}: \mathbf{Project} \to \mathbf{Env}$
- **Resource Context / 资源上下文**: $Env_{resource}: \mathbf{Resource} \to \mathbf{Env}$

---

## 4. Properties / 性质

### 4.1 Functor Properties / 函子性质

**Property 4.1** (Functor Identity)

Environment functor preserves identity:
$$Env(\text{id}_A) = \text{id}_{Env(A)}$$

**Property 4.2** (Functor Composition)

Environment functor preserves composition:
$$Env(g \circ f) = Env(g) \circ Env(f)$$

**Property 4.3** (Environment Extension)

Environments can be extended:
$$extend(Env, x: A) = Env \cup \{x: A\}$$

---

## 5. Relations / 关系

### 5.1 Relations to Other Functors / 与其他函子的关系

**Relation 5.1** (Environment → Type)

Environment functor relates to type functor:
$$Type: \mathbf{Env} \to \mathbf{Type}$$

**Relation 5.2** (Environment → Scope)

Environment functor relates to scope:
$$Scope: \mathbf{Env} \to \mathbf{Scope}$$

### 5.2 Natural Transformations / 自然变换

**Natural Transformation 5.1** (Type-Environment)

There exists a natural transformation $\eta: Type \Rightarrow Env$:
$$\eta_A: Type(A) \to Env(A)$$

connecting types to environments.

---

## 6. Examples / 例子

### 6.1 Environment Construction Example / 环境构建例子

**Example 6.1** (Variable Environment)

Consider variable environment:

$$Env = \{x: Int, y: String, f: Int \to Bool\}$$

with typed variables.

### 6.2 Project Context Example / 项目上下文例子

**Example 6.2** (Project Context)

Consider project context:

$$Env_{project} = \{project: P_{sw}, phase: Ph_{exec}, resources: R_{dev}\}$$

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

- [Environment Objects](../../01-Objects/21-Environment-Objects.md)
- [Type Functors](05-Type-Functors.md)
- [Type-Environment Natural Transformation](../../05-Natural-Transformations/05-Type-Environment-Natural-Transformation.md)
- **docs**：`docs/03-formal-verification`、`docs/06-ci-verification`（环境、变量绑定；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
