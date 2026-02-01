# Visual Concept Guide / 概念可视化指南

## 1. Overview / 概述

This document provides intuitive visual explanations and diagrams for abstract formal concepts in the Formal-ProgramManage knowledge system.

本文档为Formal-ProgramManage知识体系中的抽象形式化概念提供直观的可视化解释和图表。

---

## 2. Kripke Structure Visualization / Kripke结构可视化

### 2.1 Abstract Definition / 抽象定义

$$M = (S, S_0, R, L)$$

### 2.2 Visual Representation / 可视化表示

```mermaid
graph LR
    subgraph KripkeExample[Kripke Structure: Project Workflow]
        S0((S0<br>Initiation<br>start=true))
        S1((S1<br>Planning<br>plan=true))
        S2((S2<br>Execution<br>exec=true))
        S3((S3<br>Complete<br>done=true))

        S0 -->|approve| S1
        S1 -->|finalize| S2
        S2 -->|deliver| S3
        S2 -->|rework| S1
    end

    style S0 fill:#90EE90
    style S3 fill:#87CEEB
```

### 2.3 Intuitive Analogy / 直观类比

**Think of it as**: A map of cities (states) and roads (transitions)

- Each city has a sign showing what's true there (labels)
- You start at a specific city (initial state)
- You can only travel on existing roads (transition relation)

---

## 3. Temporal Logic (LTL) Visualization / 时序逻辑可视化

### 3.1 LTL Operators / LTL算子

```mermaid
graph LR
    subgraph LTL_Operators[LTL Operators Visual Guide]
        subgraph Next[○ Next]
            N1[Now] --> N2[Next<br>φ holds here]
        end

        subgraph Eventually[◇ Eventually]
            E1[Now] --> E2[...] --> E3[φ holds<br>somewhere]
        end

        subgraph Always[□ Always]
            A1[φ] --> A2[φ] --> A3[φ] --> A4[φ<br>forever]
        end

        subgraph Until[U Until]
            U1[φ] --> U2[φ] --> U3[ψ<br>then stops]
        end
    end
```

### 3.2 Project Examples / 项目示例

| LTL Formula | Visual | Meaning |
|-------------|--------|---------|
| ○ planning | `[init] → [planning]` | Next state is planning |
| ◇ complete | `[...] → [...] → [complete]` | Eventually reaches complete |
| □ (budget ≥ 0) | `[✓] → [✓] → [✓] → ...` | Budget always non-negative |
| planning U execution | `[plan] → [plan] → [exec]` | Planning until execution starts |

### 3.3 Intuitive Analogy / 直观类比

**Think of it as**: Describing a movie plot

- ○ (Next): "In the next scene..."
- ◇ (Eventually): "At some point..."
- □ (Always): "Throughout the entire movie..."
- U (Until): "This continues until..."

---

## 4. Markov Decision Process (MDP) Visualization / MDP可视化

### 4.1 MDP Structure / MDP结构

```mermaid
graph TD
    subgraph MDP[Markov Decision Process]
        S1[State: Low Risk]
        S2[State: Medium Risk]
        S3[State: High Risk]

        S1 -->|Action: Continue<br>P=0.8| S1
        S1 -->|Action: Continue<br>P=0.2| S2
        S2 -->|Action: Mitigate<br>P=0.7| S1
        S2 -->|Action: Mitigate<br>P=0.3| S2
        S2 -->|Action: Ignore<br>P=0.4| S3
        S3 -->|Action: Crisis Mgmt<br>P=0.5| S2
    end
```

### 4.2 Decision Tree View / 决策树视图

```mermaid
graph TD
    ROOT[Current State] --> A1{Action 1}
    ROOT --> A2{Action 2}

    A1 -->|P=0.6, R=10| S1[State A]
    A1 -->|P=0.4, R=5| S2[State B]

    A2 -->|P=0.3, R=15| S3[State C]
    A2 -->|P=0.7, R=2| S4[State D]
```

### 4.3 Intuitive Analogy / 直观类比

**Think of it as**: A board game with dice

- States = squares on the board
- Actions = moves you can make
- Probabilities = dice outcomes
- Rewards = points you earn
- Goal = maximize total points

---

## 5. Category Theory Concepts / 范畴论概念

### 5.1 Morphism (Arrow) / 态射

```mermaid
graph LR
    subgraph Morphism[Morphism: Transition Function]
        A[Object A<br>Project State 1] -->|f: morphism| B[Object B<br>Project State 2]
    end
```

**Intuitive**: A morphism is just a "transformation" or "process" that takes you from one thing to another.

### 5.2 Functor / 函子

```mermaid
graph TD
    subgraph Functor[Functor: Structure-Preserving Map]
        subgraph Cat1[Category 1: Abstract]
            A1[A] -->|f| B1[B]
        end

        subgraph Cat2[Category 2: Concrete]
            A2[F(A)] -->|F(f)| B2[F(B)]
        end

        A1 -.->|F| A2
        B1 -.->|F| B2
    end
```

**Intuitive**: A functor is like a "translation" that preserves structure—like translating a story to another language while keeping the plot.

### 5.3 Natural Transformation / 自然变换

```mermaid
graph TD
    subgraph NatTrans[Natural Transformation]
        subgraph FunctorF[Functor F]
            FA[F(A)] -->|F(f)| FB[F(B)]
        end

        subgraph FunctorG[Functor G]
            GA[G(A)] -->|G(f)| GB[G(B)]
        end

        FA -->|η_A| GA
        FB -->|η_B| GB
    end
```

**Intuitive**: A natural transformation is a "consistent way" to convert between two translations.

---

## 6. Model Checking Visualization / 模型检验可视化

### 6.1 State Space Exploration / 状态空间探索

```mermaid
graph TD
    subgraph StateSpace[State Space Exploration]
        S0[Initial] --> S1[State 1]
        S0 --> S2[State 2]
        S1 --> S3[State 3]
        S1 --> S4[State 4]
        S2 --> S4
        S2 --> S5[State 5]
        S4 --> S6[Error!]
        S3 --> S7[Success]
        S5 --> S7

        style S6 fill:#FF6B6B
        style S7 fill:#90EE90
    end
```

### 6.2 Counterexample Path / 反例路径

```mermaid
graph LR
    subgraph Counterexample[Counterexample: Path to Error]
        CE1[S0: init] -->|step1| CE2[S2: plan]
        CE2 -->|step2| CE3[S4: exec]
        CE3 -->|step3| CE4[S6: ERROR]

        style CE4 fill:#FF6B6B
    end
```

**Intuitive**: Model checking is like exploring a maze to check if any path leads to a trap.

---

## 7. Project Lifecycle State Machine / 项目生命周期状态机

### 7.1 Visual State Machine / 可视状态机

```mermaid
stateDiagram-v2
    [*] --> Initiation
    Initiation --> Planning: Approved
    Planning --> Execution: Plan Complete
    Execution --> Monitoring: Work Started
    Monitoring --> Execution: Issues Found
    Monitoring --> Closing: All Done
    Closing --> [*]: Closed

    Execution --> Cancelled: Terminated
    Planning --> Cancelled: Terminated
    Cancelled --> [*]
```

### 7.2 Phase Transition Conditions / 阶段转换条件

```mermaid
graph TD
    subgraph PhaseGate[Phase Gate Visualization]
        P1[Initiation Phase]
        G1{Gate 1<br>Approved?}
        P2[Planning Phase]
        G2{Gate 2<br>Plan OK?}
        P3[Execution Phase]

        P1 --> G1
        G1 -->|Yes| P2
        G1 -->|No| P1
        P2 --> G2
        G2 -->|Yes| P3
        G2 -->|No| P2
    end
```

---

## 8. Risk Management Visualization / 风险管理可视化

### 8.1 Risk Matrix / 风险矩阵

```mermaid
graph TD
    subgraph RiskMatrix[Risk Matrix 风险矩阵]
        subgraph High[High Impact]
            H1[Medium<br>Risk]
            H2[High<br>Risk]
            H3[Critical<br>Risk]
        end

        subgraph Med[Medium Impact]
            M1[Low<br>Risk]
            M2[Medium<br>Risk]
            M3[High<br>Risk]
        end

        subgraph Low[Low Impact]
            L1[Minimal<br>Risk]
            L2[Low<br>Risk]
            L3[Medium<br>Risk]
        end
    end

    style H3 fill:#FF6B6B
    style H2 fill:#FFA07A
    style H1 fill:#FFD700
    style M3 fill:#FFA07A
    style M2 fill:#FFD700
    style M1 fill:#90EE90
    style L3 fill:#FFD700
    style L2 fill:#90EE90
    style L1 fill:#98FB98
```

### 8.2 Risk Flow / 风险流程

```mermaid
flowchart LR
    subgraph RiskFlow[Risk Management Flow]
        I[Identify] --> A[Analyze]
        A --> P[Plan Response]
        P --> M[Monitor]
        M --> I
    end
```

---

## 9. Resource Allocation Visualization / 资源分配可视化

### 9.1 Gantt-Style View / 甘特图视图

```
Resource 1: |████ Task A ████|     |██ Task D ██|
Resource 2:      |██████ Task B ██████|
Resource 3: |██ Task C ██|    |████ Task E ████|
            ─────────────────────────────────────→ Time
            Day 1    Day 5    Day 10   Day 15
```

### 9.2 Resource Constraint Graph / 资源约束图

```mermaid
graph TD
    subgraph ResourceConstraint[Resource Constraint Visualization]
        R1[Resource: Dev Team<br>Capacity: 40h/week]
        T1[Task 1: 20h]
        T2[Task 2: 25h]
        T3[Task 3: 15h]

        R1 --> T1
        R1 --> T2
        R1 --> T3

        OVER[OVERLOAD!<br>60h > 40h capacity]
    end

    style OVER fill:#FF6B6B
```

---

## 10. Quick Reference Cards / 快速参考卡片

### 10.1 Formal Concept Quick Reference / 形式化概念快速参考

| Concept | Symbol | Visual | Intuition |
|---------|--------|--------|-----------|
| Kripke Structure | M=(S,S₀,R,L) | Graph with labeled nodes | City map |
| LTL Next | ○φ | → [φ] | Next scene |
| LTL Eventually | ◇φ | →...→ [φ] | At some point |
| LTL Always | □φ | [φ]→[φ]→[φ]→... | Forever |
| MDP | (S,A,P,R,γ) | Graph with probabilities | Board game |
| Model Checking | M ⊨ φ | Maze exploration | Find all paths |
| Counterexample | Path to violation | Highlighted path | Found the trap |

---

## 11. Status / 状态

**Document Version / 文档版本**: 1.0
**Last Updated / 最后更新**: 2026-02-02
**Status / 状态**: ✅ Complete
**Next Review / 下次审查**: 2026-05-02

**Related Documents / 相关文档**:

- [Concept Linking Index](./CONCEPT_LINKING_INDEX.md)
- [Learning Prerequisites](../docs/12-learning-support/01-learning-prerequisites.md)
- [Formal Methods Practice](../docs/11-formal-methods-practice/README.md)
