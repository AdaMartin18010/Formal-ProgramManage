# Scope Morphisms / 作用域态射

## 📋 Table of Contents / 目录

- [Scope Morphisms / 作用域态射](#scope-morphisms--作用域态射)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Scope Extension Morphism / 作用域扩展态射](#21-scope-extension-morphism--作用域扩展态射)
    - [2.2 Scope Lookup Morphism / 作用域查找态射](#22-scope-lookup-morphism--作用域查找态射)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 Harper Definition / Harper 定义](#31-harper-definition--harper-定义)
    - [3.2 Project Management Mapping / 项目管理映射](#32-project-management-mapping--项目管理映射)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Scope Properties / 作用域性质](#41-scope-properties--作用域性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Morphisms / 与其他态射的关系](#51-relations-to-other-morphisms--与其他态射的关系)
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

- **所属层**：**验证理论层（支撑）**（对应 docs/03-formal-verification、06-ci-verification；作用域态射支撑形式化验证）
- **转换关系**：**Scope Morphisms** = **状态转换**（作用域扩展、作用域查找作为状态转换）；与 06-编程语言理论概念、Category/01-Objects/22-Scope-Objects 对应。

---

## 1. Overview / 概述

**English / 英文**:

Scope morphisms represent scope extension, scope restriction, and scope lookup operations in the category $\mathbf{Scope}$. They capture how scopes are managed and manipulated. This document provides a category-theoretic perspective on scope morphisms, aligning with authoritative resources from Harper, Pierce, and other programming language theory experts.

**中文**:

作用域态射表示作用域扩展、作用域限制和作用域查找操作，属于范畴 $\mathbf{Scope}$。它们捕捉作用域如何被管理和操作。本文档从范畴论视角提供作用域态射的定义，对齐 Harper、Pierce 等编程语言理论权威资源。

**Key Insights / 关键洞察**:

- **Scope Extension / 作用域扩展**: $extend: Scope \times (Var \times Type) \to Scope$ / 作用域扩展函数
- **Scope Restriction / 作用域限制**: $restrict: Scope \times VarSet \to Scope$ / 作用域限制函数
- **Scope Lookup / 作用域查找**: $lookup: Scope \times Var \to Type$ / 作用域查找函数
- **Scope Nesting / 作用域嵌套**: Nested scopes / 嵌套作用域

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Scope Extension Morphism / 作用域扩展态射

**Definition 2.1** (Scope Extension Morphism)

A scope extension morphism $extend: S_1 \to S_2$ extends scope:

$$extend(S_1, x: A) = S_2 \text{ where } S_2 = S_1 \cup \{x: A\}$$

### 2.2 Scope Lookup Morphism / 作用域查找态射

**Definition 2.2** (Scope Lookup Morphism)

A scope lookup morphism $lookup: Scope \times Var \to Type$ looks up variables:

$$lookup(S, x) = A \text{ if } x: A \in S$$

---

## 3. Formal Definition / 形式化定义

### 3.1 Harper Definition / Harper 定义

**Definition 3.1** (Scope Operations - Harper)

Scope operations manage scopes. In our framework:

$$ScopeOp: \mathbf{Scope} \to \mathbf{Scope}$$

**Scope Operations / 作用域操作**:

- **Scope Creation / 作用域创建**: $newScope()$ - create new scope
- **Scope Extension / 作用域扩展**: $extend(S, x: A)$ - extend scope
- **Scope Lookup / 作用域查找**: $lookup(S, x)$ - lookup variable
- **Scope Restriction / 作用域限制**: $restrict(S, V)$ - restrict scope

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

### 5.1 Relations to Other Morphisms / 与其他态射的关系

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

$$S_{outer} = \{x: Int\}, S_{inner} = extend(S_{outer}, y: String)$$

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

- [Scope Objects](../../01-Objects/22-Scope-Objects.md)
- [Environment Objects](../../01-Objects/21-Environment-Objects.md)
- **docs**：`docs/03-formal-verification`、`docs/06-ci-verification`（作用域；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
