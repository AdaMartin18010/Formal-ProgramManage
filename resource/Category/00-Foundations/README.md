# 00-Foundations 范畴论基础

## 概述

本目录为 **Category** 的范畴论基础，支撑全目录的**态射=转换**、**函子=层间映射**、**自然变换=函子间转换**，与 docs 的**层**、**转换**对应。

## 文件

| 文件 | 说明 | 与层/转换 |
|:---|:---|:---|
| 01-Category-Definition.md | 范畴、对象、态射、复合、恒等 | **态射**即**转换**（$f: A \to B$）；对象可为项目、状态、阶段等 |
| 03-Functors-Natural-Transformations.md | 函子、自然变换 | **函子**=**层间/范畴间映射**；**自然变换**=函子间的**转换关系** |
| 04-Yoneda-Lemma.md | Yoneda 引理 | 将对象与态射统一为可转换的视角，支撑泛性质与**转换**的形式化 |

## 归档

**02-Calculus-Categories.md** 已归档至 `_archive/00-Foundations-Calculus/`（微积分范畴，与 Formal-ProgramManage 项目管理主线无关）。

## 与 docs 的对应

- **态射** $\leftrightarrow$ 生命周期转换 $\delta$、状态转换 $\rightarrow$（docs/02-project-management/lifecycle-models、01-foundations）
- **函子** $\leftrightarrow$ 层次转换、模型间映射（docs/KNOWLEDGE_NETWORK、06-ci-verification）
