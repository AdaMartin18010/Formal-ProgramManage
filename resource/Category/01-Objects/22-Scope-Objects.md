# Scope Objects / 作用域对象

## 📋 Table of Contents / 目录

- [Scope Objects / 作用域对象](#scope-objects--作用域对象)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Category of Scopes / 作用域范畴](#21-category-of-scopes--作用域范畴)
    - [2.2 Scope Object Properties / 作用域对象性质](#22-scope-object-properties--作用域对象性质)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 Lexical Scope / 词法作用域](#31-lexical-scope--词法作用域)
    - [3.2 Project Management Mapping / 项目管理映射](#32-project-management-mapping--项目管理映射)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Scope Properties / 作用域性质](#41-scope-properties--作用域性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Objects / 与其他对象的关系](#51-relations-to-other-objects--与其他对象的关系)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Lexical Scope Example / 词法作用域例子](#61-lexical-scope-example--词法作用域例子)
    - [6.2 Project Scope Example / 项目范围例子](#62-project-scope-example--项目范围例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Programming Language Applications / 编程语言应用](#71-programming-language-applications--编程语言应用)
    - [7.2 Project Management Applications / 项目管理应用](#72-project-management-applications--项目管理应用)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Programming Language Theory / 编程语言理论](#81-programming-language-theory--编程语言理论)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**验证理论层（支撑）**（对应 docs/03-formal-verification、06-ci-verification；作用域对象支撑形式化验证）
- **转换关系**：**Scope Objects** 作为**状态转换**的实体（作用域管理作为状态转换）；与 06-编程语言理论概念、Category/01-Objects/21-Environment-Objects 对应。

---

## 1. Overview / 概述

**English / 英文**:

Scope objects represent variable scopes, lexical scopes, and scope hierarchies in the category $\mathbf{Scope}$. They capture how variables and bindings are scoped in programming languages and project management contexts. This document provides a category-theoretic perspective on scope objects, aligning with authoritative resources from Harper, Pierce, and other programming language theory experts.

**中文**:

作用域对象表示变量作用域、词法作用域和作用域层次结构，属于范畴 $\mathbf{Scope}$。它们捕捉变量和绑定如何在编程语言和项目管理上下文中作用域化。本文档从范畴论视角提供作用域对象的定义，对齐 Harper、Pierce 等编程语言理论权威资源。

**Key Insights / 关键洞察**:

- **Lexical Scope / 词法作用域**: Scope determined by program structure / 由程序结构决定的作用域
- **Dynamic Scope / 动态作用域**: Scope determined at runtime / 运行时决定的作用域
- **Scope Hierarchy / 作用域层次**: Nested scopes / 嵌套作用域
- **Project Mapping / 项目映射**: Scope maps to project scope / 作用域映射到项目范围

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Category of Scopes / 作用域范畴

**Definition 2.1** (Category $\mathbf{Scope}$)

The category $\mathbf{Scope}$ consists of:

- **Objects / 对象**: Scopes $S \in \mathbf{Scope}$ representing variable scopes
- **Morphisms / 态射**: Scope extensions $extend: S_1 \to S_2$ where $S_2$ extends $S_1$
- **Composition / 复合**: Composition of scope extensions
- **Identity / 恒等**: Identity scope extension $\text{id}_S: S \to S$

### 2.2 Scope Object Properties / 作用域对象性质

**Axiom 2.1** (Scope Nesting)

Scopes can be nested:
$$S_1 \subseteq S_2 \Rightarrow S_1 \text{ nested in } S_2$$

**Axiom 2.2** (Scope Uniqueness)

Variables in scope are unique:
$$\forall x \in S: \exists! \text{ binding for } x$$

---

## 3. Formal Definition / 形式化定义

### 3.1 Lexical Scope / 词法作用域

**Definition 3.1** (Lexical Scope - Harper)

Lexical scope is determined by program structure. In our framework:

$$Scope_{lexical}: \mathbf{Program} \to \mathbf{Scope}$$

**Scope Operations / 作用域操作**:

- **Scope Creation / 作用域创建**: $newScope()$ - create new scope
- **Scope Extension / 作用域扩展**: $extend(S, x: A)$ - extend scope with variable
- **Scope Lookup / 作用域查找**: $lookup(S, x)$ - lookup variable in scope

### 3.2 Project Management Mapping / 项目管理映射

**Definition 3.2** (Project Scope)

In project management, scope represents project scope:

- **Project Scope / 项目范围**: Scope of project deliverables
- **Resource Scope / 资源范围**: Scope of resources
- **Task Scope / 任务范围**: Scope of tasks

---

## 4. Properties / 性质

### 4.1 Scope Properties / 作用域性质

**Property 4.1** (Scope Nesting)

Scopes can be nested:
$$S_1 \subseteq S_2 \Rightarrow S_1 \text{ nested in } S_2$$

**Property 4.2** (Scope Shadowing)

Inner scopes shadow outer scopes:
$$x \in S_1 \land x \in S_2 \land S_1 \subseteq S_2 \Rightarrow S_2 \text{ shadows } S_1$$

**Property 4.3** (Scope Extension)

Scopes can be extended:
$$extend(S, x: A) = S \cup \{x: A\}$$

---

## 5. Relations / 关系

### 5.1 Relations to Other Objects / 与其他对象的关系

**Relation 5.1** (Scope → Environment)

Scopes define environments:
$$Environment: \mathbf{Scope} \to \mathbf{Env}$$

**Relation 5.2** (Scope → Project Management)

Scopes map to project scopes:
$$ProjectScope: \mathbf{Scope} \to \mathbf{ProjectScope}$$

---

## 6. Examples / 例子

### 6.1 Lexical Scope Example / 词法作用域例子

**Example 6.1** (Nested Scopes)

Consider nested scopes:

$$S_{outer} = \{x: Int\}, S_{inner} = S_{outer} \cup \{y: String\}$$

with nested scoping.

### 6.2 Project Scope Example / 项目范围例子

**Example 6.2** (Project Deliverable Scope)

Consider project deliverable scope:

$$Scope_{project} = \{deliverable_1, deliverable_2, deliverable_3\}$$

defining project scope.

---

## 7. Applications / 应用

### 7.1 Programming Language Applications / 编程语言应用

- **Variable Scoping**: Managing variable scopes
- **Scope Analysis**: Analyzing program scopes
- **Scope Resolution**: Resolving variable references
- **Scope Optimization**: Optimizing scope usage

### 7.2 Project Management Applications / 项目管理应用

- **Project Scope Management**: Managing project scope
- **Scope Definition**: Defining project scope
- **Scope Control**: Controlling scope changes
- **Scope Verification**: Verifying scope compliance

---

## 8. References / 参考文献

### 8.1 Programming Language Theory / 编程语言理论

1. Harper, R. (2016). *Practical Foundations for Programming Languages* (2nd ed.). Cambridge University Press.
2. Pierce, B. C. (2002). *Types and Programming Languages*. MIT Press.

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.
2. Awodey, S. (2010). *Category Theory* (2nd ed.). Oxford University Press.

### 8.3 Related Files / 相关文件

- [Environment Objects](21-Environment-Objects.md)
- [Type Objects](20-Type-Objects.md)
- [Scope Morphisms](../../02-Morphisms/18-Scope-Morphisms.md)
- **docs**：`docs/03-formal-verification`、`docs/06-ci-verification`（作用域、词法作用域；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
