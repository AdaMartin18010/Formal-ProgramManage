# Axiomatic Semantics Morphisms / 公理语义态射

## 📋 Table of Contents / 目录

- [Axiomatic Semantics Morphisms / 公理语义态射](#axiomatic-semantics-morphisms--公理语义态射)
  - [📋 Table of Contents / 目录](#-table-of-contents--目录)
  - [0. 所属层与转换关系 / Layer and Transformation](#0-所属层与转换关系--layer-and-transformation)
  - [1. Overview / 概述](#1-overview--概述)
  - [2. Category Theory Definition / 范畴论定义](#2-category-theory-definition--范畴论定义)
    - [2.1 Hoare Triple Morphism / Hoare三元组态射](#21-hoare-triple-morphism--hoare三元组态射)
    - [2.2 Verification Morphism / 验证态射](#22-verification-morphism--验证态射)
  - [3. Formal Definition / 形式化定义](#3-formal-definition--形式化定义)
    - [3.1 Hoare Definition / Hoare 定义](#31-hoare-definition--hoare-定义)
    - [3.2 Project Management Mapping / 项目管理映射](#32-project-management-mapping--项目管理映射)
  - [4. Properties / 性质](#4-properties--性质)
    - [4.1 Verification Properties / 验证性质](#41-verification-properties--验证性质)
  - [5. Relations / 关系](#5-relations--关系)
    - [5.1 Relations to Other Morphisms / 与其他态射的关系](#51-relations-to-other-morphisms--与其他态射的关系)
  - [6. Examples / 例子](#6-examples--例子)
    - [6.1 Assignment Example / 赋值例子](#61-assignment-example--赋值例子)
    - [6.2 Project Property Example / 项目性质例子](#62-project-property-example--项目性质例子)
  - [7. Applications / 应用](#7-applications--应用)
    - [7.1 Programming Language Applications / 编程语言应用](#71-programming-language-applications--编程语言应用)
    - [7.2 Project Management Applications / 项目管理应用](#72-project-management-applications--项目管理应用)
  - [8. References / 参考文献](#8-references--参考文献)
    - [8.1 Semantics Theory / 语义理论](#81-semantics-theory--语义理论)
    - [8.2 Category Theory / 范畴论](#82-category-theory--范畴论)
    - [8.3 Related Files / 相关文件](#83-related-files--相关文件)

---

## 0. 所属层与转换关系 / Layer and Transformation

- **所属层**：**验证理论层**（对应 docs/06-ci-verification、docs/03-formal-verification；公理语义）
- **转换关系**：**Axiomatic Semantics Morphisms** = **模型转换**（Hoare三元组、程序验证作为模型转换方法）；与 Category/01-Objects/12-Verification-Objects、Category/02-Morphisms/12-Verification-Morphisms 对应。

**与 docs/03-formal-verification、06-ci-verification 的公式对应**：

- Hoare 三元组 $\{P\}C\{Q\}$、规则 $\{Q[v/x]\}x:=v\{Q\}$、$\{P\}e_1\{R\},\{R\}e_2\{Q\} \Rightarrow \{P\}e_1;e_2\{Q\}$ → 验证系统 $V(m,\phi)$、$m \models \phi$、theorem-proving；与 13-Proof-Morphisms、12-Verification-Morphisms 一致。
- 程序验证、正确性证明 → 06-ci-verification 的 形式化验证、Hoare 逻辑、验证条件生成 衔接。

---

## 1. Overview / 概述

**English / 英文**:

Axiomatic semantics morphisms represent Hoare triples and program verification using preconditions and postconditions. They capture how program properties are specified and verified. This document provides a category-theoretic perspective on axiomatic semantics morphisms, aligning with authoritative resources from Hoare and other verification theory experts.

**中文**:

公理语义态射表示使用前置条件和后置条件的Hoare三元组和程序验证。它们捕捉程序性质如何被指定和验证。本文档从范畴论视角提供公理语义态射的定义，对齐 Hoare 等验证理论权威资源。

**Key Insights / 关键洞察**:

- **Hoare Triple / Hoare三元组**: $\{P\} e \{Q\}$ / Hoare三元组
- **Precondition / 前置条件**: $P$ - precondition / 前置条件
- **Postcondition / 后置条件**: $Q$ - postcondition / 后置条件
- **Program Verification / 程序验证**: Verifying program properties / 验证程序性质

---

## 2. Category Theory Definition / 范畴论定义

### 2.1 Hoare Triple Morphism / Hoare三元组态射

**Definition 2.1** (Hoare Triple Morphism)

A Hoare triple morphism $\{P\} e \{Q\}$:

$$\{P\} e \{Q\} \text{ if } P \Rightarrow \text{post}(e, Q)$$

where $P$ is precondition and $Q$ is postcondition.

### 2.2 Verification Morphism / 验证态射

**Definition 2.2** (Verification Morphism)

A verification morphism $verify: Program \times Property \to Proof$:

$$verify(P, Prop) = \pi \text{ where } \pi \text{ proves } Prop$$

---

## 3. Formal Definition / 形式化定义

### 3.1 Hoare Definition / Hoare 定义

**Definition 3.1** (Hoare Logic - Hoare)

Hoare logic uses triples $\{P\} e \{Q\}$:

- **Precondition / 前置条件**: $P$ - what holds before
- **Program / 程序**: $e$ - program statement
- **Postcondition / 后置条件**: $Q$ - what holds after

**Hoare Rules / Hoare规则**:

- **Assignment / 赋值**: $\{Q[v/x]\} x := v \{Q\}$
- **Sequence / 序列**: $\{P\} e_1 \{R\}, \{R\} e_2 \{Q\} \Rightarrow \{P\} e_1; e_2 \{Q\}$
- **Conditional / 条件**: $\{P \land c\} e_1 \{Q\}, \{P \land \neg c\} e_2 \{Q\} \Rightarrow \{P\} \text{if } c \text{ then } e_1 \text{ else } e_2 \{Q\}$

### 3.2 Project Management Mapping / 项目管理映射

**Definition 3.2** (Project Verification)

In project management, verification checks properties:

- **Project Properties / 项目性质**: Properties of projects
- **Process Properties / 过程性质**: Properties of processes
- **Outcome Properties / 成果性质**: Properties of outcomes

---

## 4. Properties / 性质

### 4.1 Verification Properties / 验证性质

**Property 4.1** (Verification Soundness)

Verification is sound:

$$\{P\} e \{Q\} \Rightarrow \text{if } P \text{ then } Q \text{ after } e$$

**Property 4.2** (Verification Completeness)

Verification is complete:

$$\text{if } P \text{ then } Q \text{ after } e \Rightarrow \{P\} e \{Q\}$$

---

## 5. Relations / 关系

### 5.1 Relations to Other Morphisms / 与其他态射的关系

**Relation 5.1** (Axiomatic → Operational)

Axiomatic semantics relates to operational semantics:

$$OperationalSemantics: \mathbf{AxSem} \to \mathbf{OpSem}$$

**Relation 5.2** (Axiomatic → Denotational)

Axiomatic semantics relates to denotational semantics:

$$DenotationalSemantics: \mathbf{AxSem} \to \mathbf{DenSem}$$

---

## 6. Examples / 例子

### 6.1 Assignment Example / 赋值例子

**Example 6.1** (Variable Assignment)

Consider assignment:

$$\{x = 0\} x := x + 1 \{x = 1\}$$

verifying assignment property.

### 6.2 Project Property Example / 项目性质例子

**Example 6.2** (Project Completion)

Consider project completion:

$$\{ProjectStarted\} ExecuteProject \{ProjectCompleted\}$$

verifying project completion property.

---

## 7. Applications / 应用

### 7.1 Programming Language Applications / 编程语言应用

- **Program Verification**: Verifying program properties
- **Property Specification**: Specifying program properties
- **Correctness Proofs**: Proving program correctness
- **Safety Verification**: Verifying program safety

### 7.2 Project Management Applications / 项目管理应用

- **Project Verification**: Verifying project properties
- **Process Verification**: Verifying process properties
- **Outcome Verification**: Verifying outcome properties

---

## 8. References / 参考文献

### 8.1 Semantics Theory / 语义理论

1. Hoare, C. A. R. (1969). An axiomatic basis for computer programming. *Communications of the ACM*, 12(10), 576-580.

### 8.2 Category Theory / 范畴论

1. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.

### 8.3 Related Files / 相关文件

- [Semantic Objects](../../01-Objects/03-Semantic-Objects.md)
- [Verification Objects](../../01-Objects/12-Verification-Objects.md)
- **docs**：`docs/03-formal-verification`（公理语义；与 0. 对应）

---

**Last Updated / 最后更新**: 2025-01-XX
**Status / 状态**: ✅ Complete / 完成
**Version / 版本**: 1.0
