# Learning Prerequisites Map / 先备知识地图

## 1. Overview / 概述

### 1.1 Introduction / 简介

**English**: This document provides a comprehensive prerequisite knowledge map for the Formal-ProgramManage knowledge system. It defines the foundational knowledge required before studying each topic area, helping learners efficiently plan their learning path.

**中文**: 本文档为Formal-ProgramManage知识体系提供全面的先备知识地图。它定义了学习每个主题领域之前所需的基础知识，帮助学习者高效规划学习路径。

### 1.2 Theoretical Basis / 理论基础

Based on:

- **Bloom's Taxonomy** (Anderson & Krathwohl, 2001): Learning hierarchy
- **Zone of Proximal Development** (Vygotsky): Optimal learning challenge
- **Cognitive Load Theory** (Sweller): Working memory constraints
- **Prerequisite Learning** (Gagné): Learning sequences

---

## 2. Knowledge Layer Prerequisites / 知识层次先备知识

### 2.1 Foundation Layer (FL) Prerequisites / 基础理论层先备知识

```mermaid
graph TD
    subgraph Prerequisites[先备知识 / Prerequisites]
        M1[Basic Mathematics<br>基础数学]
        M2[Set Theory<br>集合论]
        M3[Logic<br>逻辑学]
        M4[Probability<br>概率论]
    end

    subgraph FL[Foundation Layer 基础理论层]
        FL1[Formal Foundations<br>形式化基础]
        FL2[Mathematical Models<br>数学模型]
        FL3[Semantic Models<br>语义模型]
    end

    M1 --> FL2
    M2 --> FL1
    M3 --> FL1
    M4 --> FL2
    M2 --> FL3
    M3 --> FL3
```

| FL Topic / FL主题 | Prerequisites / 先备知识 | Difficulty / 难度 |
|-------------------|-------------------------|-------------------|
| FL-1.1 Formal Foundations | Set theory, Propositional logic, Predicate logic | High |
| FL-1.2 Mathematical Models | Linear algebra basics, Calculus, Probability | High |
| FL-1.3 Semantic Models | Formal languages, Automata theory | High |
| FL-1.4 Quantum Theory | Quantum mechanics basics, Linear algebra | Very High |
| FL-1.5 Bio-inspired Theory | Biology basics, Optimization theory | Medium |

### 2.2 Core Model Layer (CML) Prerequisites / 核心模型层先备知识

```mermaid
graph TD
    subgraph Prerequisites[先备知识]
        P1[FL-1.1 Formal Foundations]
        P2[Basic PM Knowledge]
        P3[Process Thinking]
    end

    subgraph CML[Core Model Layer 核心模型层]
        C1[Lifecycle Models]
        C2[Resource Models]
        C3[Risk Models]
        C4[Quality Models]
    end

    P1 --> C1
    P1 --> C2
    P1 --> C3
    P1 --> C4
    P2 --> C1
    P2 --> C2
    P3 --> C1
```

| CML Topic / CML主题 | Prerequisites / 先备知识 | Difficulty / 难度 |
|---------------------|-------------------------|-------------------|
| CML-2.1 Lifecycle Models | FL-1.1, PMBOK basics, State machines | Medium |
| CML-2.2 Resource Models | FL-1.2, Optimization basics | Medium |
| CML-2.3 Risk Models | FL-1.2, Probability, Statistics | Medium |
| CML-2.4 Quality Models | FL-1.1, Statistical process control | Medium |

### 2.3 Verification Layer (VL) Prerequisites / 验证理论层先备知识

```mermaid
graph TD
    subgraph Prerequisites[先备知识]
        P1[FL-1.1 Formal Foundations]
        P2[FL-1.3 Semantic Models]
        P3[Automata Theory]
        P4[Logic Programming]
    end

    subgraph VL[Verification Layer 验证理论层]
        V1[Model Checking]
        V2[Theorem Proving]
        V3[Consistency Checking]
    end

    P1 --> V1
    P2 --> V1
    P3 --> V1
    P1 --> V2
    P4 --> V2
    P2 --> V3
```

| VL Topic / VL主题 | Prerequisites / 先备知识 | Difficulty / 难度 |
|-------------------|-------------------------|-------------------|
| VL-3.1 Model Checking | Temporal logic (LTL, CTL), Kripke structures | High |
| VL-3.2 Theorem Proving | Proof theory, Type theory | Very High |
| VL-3.3 Consistency | Model theory, First-order logic | High |

### 2.4 Application Layer (AL) Prerequisites / 应用模型层先备知识

```mermaid
graph TD
    subgraph Prerequisites[先备知识]
        P1[CML-2.1 to CML-2.4]
        P2[Domain Knowledge]
        P3[VL Basics]
    end

    subgraph AL[Application Layer 应用模型层]
        A1[Software Development]
        A2[Engineering Management]
        A3[Business Management]
        A4[Emerging Tech]
    end

    P1 --> A1
    P1 --> A2
    P1 --> A3
    P1 --> A4
    P2 --> A1
    P2 --> A2
    P2 --> A3
    P3 --> A4
```

| AL Topic / AL主题 | Prerequisites / 先备知识 | Difficulty / 难度 |
|-------------------|-------------------------|-------------------|
| AL-4.1 Software Development | CML, Agile/Scrum basics | Medium |
| AL-4.2 Engineering Management | CML, Systems engineering | Medium |
| AL-4.3 Business Management | CML, Business fundamentals | Medium |
| AL-4.4+ Emerging Tech | CML, VL, Domain expertise | High |

---

## 3. Concept-Level Prerequisites / 概念级先备知识

### 3.1 Core Concept Dependency Graph / 核心概念依赖图

```mermaid
graph LR
    subgraph Level0[Level 0: Fundamentals 基础]
        L0A[Set Theory<br>集合论]
        L0B[Logic<br>逻辑学]
        L0C[Probability<br>概率论]
    end

    subgraph Level1[Level 1: Formal Basics 形式化基础]
        L1A[State Machines<br>状态机]
        L1B[Temporal Logic<br>时序逻辑]
        L1C[Kripke Structures<br>Kripke结构]
    end

    subgraph Level2[Level 2: PM Formal 项目管理形式化]
        L2A[Project State Space<br>项目状态空间]
        L2B[Lifecycle Transitions<br>生命周期转换]
        L2C[Resource Allocation<br>资源分配]
    end

    subgraph Level3[Level 3: Verification 验证]
        L3A[Model Checking<br>模型检验]
        L3B[Property Verification<br>属性验证]
    end

    L0A --> L1A
    L0B --> L1B
    L0A --> L1C
    L0B --> L1C

    L1A --> L2A
    L1B --> L2B
    L1C --> L2A
    L0C --> L2C

    L2A --> L3A
    L2B --> L3A
    L1B --> L3B
    L3A --> L3B
```

### 3.2 Detailed Concept Prerequisites / 详细概念先备知识

| Concept / 概念 | Direct Prerequisites / 直接先备知识 | Indirect Prerequisites / 间接先备知识 |
|----------------|-----------------------------------|--------------------------------------|
| Kripke Structure | Set theory, Binary relations | Logic |
| LTL Formulas | Propositional logic, Path semantics | Set theory |
| CTL Formulas | Tree semantics, State branching | Logic, Set theory |
| MDP | Probability theory, State machines | Calculus |
| Resource Allocation | Optimization, Constraints | Linear algebra |
| Risk Matrix | Probability, Impact assessment | Statistics |
| Model Checking | Kripke structures, Temporal logic | All above |

---

## 4. Learning Path Recommendations / 学习路径推荐

### 4.1 Path A: For PM Practitioners / 路径A：项目管理从业者

```mermaid
flowchart LR
    subgraph Phase1[Phase 1: Foundation 基础阶段]
        A1[PMBOK Review] --> A2[Basic Logic]
        A2 --> A3[Set Theory Basics]
    end

    subgraph Phase2[Phase 2: Core Concepts 核心概念]
        B1[State Machines] --> B2[Project State Space]
        B2 --> B3[Lifecycle Models]
    end

    subgraph Phase3[Phase 3: Application 应用阶段]
        C1[Industry Models] --> C2[Tool Practice]
    end

    Phase1 --> Phase2 --> Phase3
```

**Duration / 时长**: 8-12 weeks

### 4.2 Path B: For Computer Scientists / 路径B：计算机科学背景

```mermaid
flowchart LR
    subgraph Phase1[Phase 1: PM Basics PM基础]
        A1[PMBOK Overview] --> A2[Lifecycle Concepts]
    end

    subgraph Phase2[Phase 2: Formal PM 形式化PM]
        B1[Apply Formal Methods] --> B2[Verification]
        B2 --> B3[Model Checking]
    end

    subgraph Phase3[Phase 3: Advanced 高级]
        C1[Theorem Proving] --> C2[Tool Integration]
    end

    Phase1 --> Phase2 --> Phase3
```

**Duration / 时长**: 6-10 weeks

### 4.3 Path C: Comprehensive / 路径C：综合学习

```mermaid
flowchart TD
    subgraph Week1_2[Week 1-2]
        W1[Logic + Set Theory]
    end

    subgraph Week3_4[Week 3-4]
        W2[PMBOK + State Machines]
    end

    subgraph Week5_6[Week 5-6]
        W3[Temporal Logic + Project State]
    end

    subgraph Week7_8[Week 7-8]
        W4[Core Models CML]
    end

    subgraph Week9_10[Week 9-10]
        W5[Verification VL]
    end

    subgraph Week11_12[Week 11-12]
        W6[Applications + Tools]
    end

    Week1_2 --> Week3_4 --> Week5_6 --> Week7_8 --> Week9_10 --> Week11_12
```

**Duration / 时长**: 12 weeks

---

## 5. Self-Assessment Checklist / 自我评估清单

### 5.1 Foundation Prerequisites / 基础先备知识

Check your readiness for Foundation Layer (FL):

- [ ] Can define and work with sets, subsets, unions, intersections
- [ ] Understand propositional logic (AND, OR, NOT, IMPLIES)
- [ ] Can write and evaluate predicate logic formulas
- [ ] Understand basic probability concepts (P(A), P(A|B), Bayes)
- [ ] Familiar with functions and relations
- [ ] Can read and understand basic mathematical notation

### 5.2 Core Model Prerequisites / 核心模型先备知识

Check your readiness for Core Model Layer (CML):

- [ ] Completed FL-1.1 (Formal Foundations)
- [ ] Understand project lifecycle phases (Initiation → Closing)
- [ ] Familiar with PMBOK or equivalent framework
- [ ] Can model systems as state machines
- [ ] Understand optimization concepts

### 5.3 Verification Prerequisites / 验证先备知识

Check your readiness for Verification Layer (VL):

- [ ] Understand Kripke structures
- [ ] Can write LTL/CTL formulas
- [ ] Familiar with proof techniques
- [ ] Have programming experience

---

## 6. Resources for Prerequisites / 先备知识资源

### 6.1 Mathematics / 数学

| Topic | Resource | Type |
|-------|----------|------|
| Set Theory | MIT OCW 18.510 | Course |
| Logic | Stanford Encyclopedia of Philosophy | Reference |
| Probability | Khan Academy Probability | Video |

### 6.2 Formal Methods / 形式化方法

| Topic | Resource | Type |
|-------|----------|------|
| Automata | Introduction to Automata (Hopcroft) | Textbook |
| Model Checking | Principles of Model Checking (Baier) | Textbook |
| TLA+ | TLA+ Video Course (Lamport) | Video |

### 6.3 Project Management / 项目管理

| Topic | Resource | Type |
|-------|----------|------|
| PMBOK | PMI PMBOK Guide 7th Edition | Standard |
| Agile | Agile Practice Guide (PMI) | Standard |
| Risk | ISO 31000:2018 | Standard |

---

## 7. Status / 状态

**Document Version / 文档版本**: 1.0
**Last Updated / 最后更新**: 2026-02-02
**Status / 状态**: ✅ Complete
**Next Review / 下次审查**: 2026-05-02

**Related Documents / 相关文档**:

- [Spaced Repetition Schedule](./02-spaced-repetition-schedule.md)
- [Retrieval Practice Questions](./03-retrieval-practice-questions.md)
- [Concept Difficulty Ranking](./04-concept-difficulty-ranking.md)
