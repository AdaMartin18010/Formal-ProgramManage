# Substitution Morphisms / 替换态射

## 📋 Table of Contents / 目录

- [Substitution Morphisms / 替换态射](#substitution-morphisms--替换态射)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Substitution Morphism / 替换态射](#21-substitution-morphism--替换态射)
    - [2.2 Substitution Composition / 替换复合](#22-substitution-composition--替换复合)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 Harper Definition / Harper 定义](#31-harper-definition--harper-定义)
    - [3.2 Project Management Mapping / 项目管理映射](#32-project-management-mapping--项目管理映射)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Substitution Properties / 替换性质](#41-substitution-properties--替换性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Morphisms / 与其他态射的关系](#51-relations-to-other-morphisms--与其他态射的关系)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Variable Substitution Example / 变量替换例子](#61-variable-substitution-example--变量替换例子)
    - [6.2 Project Resource Substitution Example / 项目资源替换例子](#62-project-resource-substitution-example--项目资源替换例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Programming Language Applications / 编程语言应用](#71-programming-language-applications--编程语言应用)
    - [7.2 Project Management Applications / 项目管理应用](#72-project-management-applications--项目管理应用)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Programming Language Theory / 编程语言理论](#81-programming-language-theory--编程语言理论)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**验证理论层（支撑）**（对应 docs/03-formal-verification、06-ci-verification；替换态射支撑形式化验证）
- **转换关系**：**Substitution Morphisms** = **模型转换**（变量替换、项替换、类型替换作为模型转换方法）；与 06-编程语言理论概念、Category/03-Constructions/01-Type-Constructions 对应。

**与 docs/03-formal-verification、06-ci-verification 的公式对应**：

- $subst(t,\sigma)=t[\sigma]$、$(\sigma_2\circ\sigma_1)(t)=\sigma_2(\sigma_1(t))$ → 定理证明、类型论中的 项重写、$\beta$-归约、等式推理；$[x \mapsto v]$、$[t/x]$、$[\sigma/\alpha]$ 与 13-Proof-Morphisms、20-Type-Objects 衔接。
- 验证系统 $VS=(M,\Phi,\mathcal{L},\models,\mathcal{V},\mathcal{R})$ 中的 模型变换、项/公式的代换 → 替换态射；与 06-ci-verification 的程序变换、等价转换 对应。

---

## 1. Overview / 概述

**English / 英文**:

Substitution morphisms represent variable substitution, term substitution, and type substitution operations. They capture how substitutions are performed in programming languages and project management contexts. This document provides a category-theoretic perspective on substitution morphisms, aligning with authoritative resources from Harper, Pierce, and other programming language theory experts.

**中文**:

替换态射表示变量替换、项替换和类型替换操作。它们捕捉替换如何在编程语言和项目管理上下文中执行。本文档从范畴论视角提供替换态射的定义，对齐 Harper、Pierce 等编程语言理论权威资源。

**Key Insights / 关键洞察**:

- **Variable Substitution / 变量替换**: $[x \mapsto v]$ - substituting variable / 替换变量
- **Term Substitution / 项替换**: $[t/x]$ - substituting term / 替换项
- **Type Substitution / 类型替换**: $[\sigma/\alpha]$ - substituting type / 替换类型
- **Substitution Composition / 替换复合**: Composing substitutions / 复合替换

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Substitution Morphism / 替换态射

**Definition 2.1** (Substitution Morphism)

A substitution morphism $subst: Term \times Substitution \to Term$:

$$subst(t, \sigma) = t[\sigma]$$

where $\sigma$ is a substitution.

### 2.2 Substitution Composition / 替换复合

**Definition 2.2** (Substitution Composition)

Substitutions compose:

$$(\sigma_2 \circ \sigma_1)(t) = \sigma_2(\sigma_1(t))$$

---

## 3. Formal Definition / 形式化定义

### 3.1 Harper Definition / Harper 定义

**Definition 3.1** (Substitution - Harper)

Substitution replaces variables with terms. In our framework:

$$subst: \mathbf{Term} \times \mathbf{Substitution} \to \mathbf{Term}$$

**Substitution Operations / 替换操作**:

- **Variable Substitution / 变量替换**: $[x \mapsto v]$ - replace variable
- **Term Substitution / 项替换**: $[t/x]$ - replace term
- **Type Substitution / 类型替换**: $[\sigma/\alpha]$ - replace type

### 3.2 Project Management Mapping / 项目管理映射

**Definition 3.2** (Project Substitution)

In project management, substitution represents replacement:

- **Resource Substitution / 资源替换**: Replacing resources
- **Task Substitution / 任务替换**: Replacing tasks
- **Plan Substitution / 计划替换**: Replacing plans

---

## 4. Properties / 性质

### 4.1 Substitution Properties / 替换性质

**Property 4.1** (Substitution Composition)

Substitutions compose:

$$(\sigma_3 \circ \sigma_2) \circ \sigma_1 = \sigma_3 \circ (\sigma_2 \circ \sigma_1)$$

**Property 4.2** (Substitution Identity)

Identity substitution exists:

$$\text{id}(t) = t$$

---

## 5. Relations / 关系

### 5.1 Relations to Other Morphisms / 与其他态射的关系

**Relation 5.1** (Substitution → Environment)

Substitution uses environments:

$$Environment: \mathbf{Substitution} \to \mathbf{Env}$$

**Relation 5.2** (Substitution → Type)

Substitution preserves types:

$$Type: \mathbf{Substitution} \to \mathbf{Type}$$

---

## 6. Examples / 例子

### 6.1 Variable Substitution Example / 变量替换例子

**Example 6.1** (Variable Replacement)

Consider variable substitution:

$$[x \mapsto 5](x + 1) = 5 + 1 = 6$$

replacing variable $x$ with value $5$.

### 6.2 Project Resource Substitution Example / 项目资源替换例子

**Example 6.2** (Resource Replacement)

Consider resource substitution:

$$[Dev_1 \mapsto Dev_2](Task) = Task'$$

replacing developer $Dev_1$ with $Dev_2$.

---

## 7. Applications / 应用

### 7.1 Programming Language Applications / 编程语言应用

- **Variable Substitution**: Substituting variables in programs
- **Term Substitution**: Substituting terms
- **Type Substitution**: Substituting types
- **Evaluation**: Evaluating expressions using substitution

### 7.2 Project Management Applications / 项目管理应用

- **Resource Substitution**: Substituting project resources
- **Task Substitution**: Substituting tasks
- **Plan Substitution**: Substituting plans

---

## 8. References / 参考文献

### 8.1 Programming Language Theory / 编程语言理论

1. Harper, R. (2016). *Practical Foundations for Programming Languages* (2nd ed.). Cambridge University Press.
2. Pierce, B. C. (2002). *Types and Programming Languages*. MIT Press.

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.

### 8.3 Related Files / 相关文件

- [Environment Objects](../../01-Objects/21-Environment-Objects.md)
- [Type Objects](../../01-Objects/20-Type-Objects.md)
- **docs**：`docs/03-formal-verification`（替换、环境；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
