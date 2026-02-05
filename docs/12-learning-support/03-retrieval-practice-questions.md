# Retrieval Practice Questions / 检索练习问题库

## 1. Overview / 概述

### 1.1 Introduction / 简介

**English**: This document provides a comprehensive question bank for retrieval practice across all knowledge layers. Questions are designed to enhance memory consolidation through active recall, following the testing effect research. Each section below links to the corresponding content document and section for "study then practice" and spaced review by topic.

**中文**: 本文档为所有知识层次提供全面的检索练习问题库。问题设计遵循测试效应研究，通过主动回忆来增强记忆巩固。各节均标注对应文档与章节，便于“学完即练”和按主题间隔复习。

### 1.2 Question Types / 问题类型

| Type / 类型 | Purpose / 目的 | Difficulty / 难度 |
|-------------|---------------|-------------------|
| Definition Recall | Test basic knowledge | Low |
| Concept Explanation | Test understanding | Medium |
| Application | Test practical skills | Medium-High |
| Analysis | Test analytical thinking | High |
| Comparison | Test relational understanding | High |
| Problem Solving | Test integration | Very High |

---

## 2. Foundation Layer (FL) Questions / 基础理论层问题

### 2.1 FL-1.1 Formal Foundations / 形式化基础

**对应章节 / Source**: [01-foundations/README.md](../01-foundations/README.md) — §2 定义（状态转换系统、Kripke、LTL）、§3 属性（安全性、活性）

#### Definition Recall / 定义回忆

1. **Q**: What is a Kripke structure? What are its four components?

   **A**: A Kripke structure is a tuple M = (S, S₀, R, L) where:
   - S: finite set of states
   - S₀ ⊆ S: set of initial states
   - R ⊆ S × S: transition relation
   - L: S → 2^AP: labeling function

2. **Q**: Define LTL (Linear Temporal Logic). What are its main operators?

   **A**: LTL is a modal temporal logic for expressing properties over infinite paths. Main operators:
   - ○ (Next): holds in next state
   - □ (Always): holds in all future states
   - ◇ (Eventually): holds in some future state
   - U (Until): holds until another property holds

3. **Q**: What is a state transition system in project management context?

   **A**: A state transition system models project as a tuple P = (S, R, T, C) where states represent project configurations, and transitions represent valid changes (phase transitions, resource allocations, etc.).

#### Concept Explanation / 概念解释

1. **Q**: Explain the difference between safety and liveness properties. Give a project management example of each.

   **A**:
   - **Safety**: "Bad things never happen" - e.g., "Budget never exceeds limit"
   - **Liveness**: "Good things eventually happen" - e.g., "Project eventually completes"

2. **Q**: Why is formal specification important for project management?

   **A**: Formal specification provides:
   - Precise, unambiguous descriptions
   - Ability to verify properties mathematically
   - Detection of edge cases and conflicts
   - Documentation as executable specification

#### Application / 应用

1. **Q**: Write an LTL formula expressing: "If a task starts, it will eventually complete."

   **A**: □(task_started → ◇task_completed)

2. **Q**: Model a simple approval workflow as a Kripke structure with 3 states: Pending, Approved, Rejected.

   **A**:
   - S = {Pending, Approved, Rejected}
   - S₀ = {Pending}
   - R = {(Pending, Approved), (Pending, Rejected)}
   - L(Pending) = {waiting}, L(Approved) = {success}, L(Rejected) = {failure}

### 2.2 FL-1.2 Mathematical Models / 数学模型

**对应章节 / Source**: [01-foundations/mathematical-models.md](../01-foundations/mathematical-models.md) — MDP、价值函数、优化模型

#### Definition Recall / 定义回忆

1. **Q**: What is a Markov Decision Process (MDP)? What are its five components?

   **A**: MDP = (S, A, P, R, γ) where:
   - S: set of states
   - A: set of actions
   - P: transition probability function
   - R: reward function
   - γ: discount factor

2. **Q**: Define the value function V(s) in MDP.

   **A**: V(s) = Expected cumulative discounted reward starting from state s following optimal policy: V(s) = max_a [R(s,a) + γ Σ P(s'|s,a)V(s')]

#### Problem Solving / 问题求解

1. **Q**: A project has 3 phases. Transition probabilities: P(Phase1→Phase2) = 0.8, P(Phase2→Phase3) = 0.9. What's the probability of reaching Phase 3?

    **A**: P(Phase3) = P(Phase1→Phase2) × P(Phase2→Phase3) = 0.8 × 0.9 = 0.72

---

## 3. Core Model Layer (CML) Questions / 核心模型层问题

### 3.1 CML-2.1 Lifecycle Models / 生命周期模型

**对应章节 / Source**: [02-project-management/lifecycle-models.md](../02-project-management/lifecycle-models.md) — 过程组、绩效域、形式化生命周期

#### Definition Recall / 定义回忆

1. **Q**: List the five PMBOK process groups and their primary purpose.

    **A**:
    - Initiating: Define and authorize project
    - Planning: Define scope, objectives, action plan
    - Executing: Integrate and perform work
    - Monitoring & Controlling: Track and regulate progress
    - Closing: Formalize acceptance, close project

2. **Q**: What are the 8 Performance Domains in PMBOK 7th Edition?

    **A**: Stakeholders, Team, Development Approach & Life Cycle, Planning, Project Work, Delivery, Measurement, Uncertainty

#### Comparison / 比较

1. **Q**: Compare waterfall and agile lifecycle models in terms of flexibility and risk management.

    **A**:

    | Aspect | Waterfall | Agile |
    |--------|-----------|-------|
    | Flexibility | Low (changes costly) | High (embraces change) |
    | Risk Management | Front-loaded | Distributed throughout |
    | Feedback | Late (after phases) | Early and continuous |

### 3.2 CML-2.2 Resource Models / 资源管理模型

**对应章节 / Source**: [02-project-management/resource-models.md](../02-project-management/resource-models.md) — 资源分配、约束、调度

#### Application / 应用

1. **Q**: You have 3 developers and 5 tasks. Task durations: T1=2d, T2=3d, T3=2d, T4=4d, T5=1d. Dependencies: T2→T4, T3→T5. Find an optimal allocation.

    **A**: One optimal solution:
    - Dev1: T1(d1-2), T4(d4-7)
    - Dev2: T2(d1-3), T5(d4)
    - Dev3: T3(d1-2)
    - Total: 7 days

2. **Q**: Write a formal constraint for: "Total allocated resources cannot exceed capacity."

    **A**: ∀r ∈ Resources: Σ(allocation(t,r) for t ∈ Tasks) ≤ capacity(r)

### 3.3 CML-2.3 Risk Models / 风险管理模型

**对应章节 / Source**: [02-project-management/risk-models.md](../02-project-management/risk-models.md) — 风险暴露、应对策略

#### Definition Recall / 定义回忆

1. **Q**: What is risk exposure and how is it calculated?

    **A**: Risk Exposure = Probability × Impact. It quantifies the expected loss from a risk.

2. **Q**: List the four risk response strategies for negative risks.

    **A**:
    - Avoid: Eliminate the threat
    - Mitigate: Reduce probability or impact
    - Transfer: Shift to third party
    - Accept: Acknowledge without action

#### Analysis / 分析

1. **Q**: A risk has P=0.3, Impact=$100K. Mitigation costs $20K and reduces P to 0.1. Should you mitigate?

    **A**:
    - Without mitigation: Expected loss = 0.3 × $100K = $30K
    - With mitigation: Cost + Expected loss = $20K + 0.1 × $100K = $30K
    - Break-even. Consider other factors (reputation, secondary risks).

### 3.4 CML-2.4 Quality Models / 质量管理模型

**对应章节 / Source**: [02-project-management/quality-models.md](../02-project-management/quality-models.md) — QA/QC、COQ

#### Definition Recall / 定义回忆

1. **Q**: What is the difference between quality assurance and quality control?

    **A**:
    - **QA**: Process-oriented, preventive, focuses on building quality in
    - **QC**: Product-oriented, detective, focuses on finding defects

2. **Q**: Define the Cost of Quality (COQ) and its components.

    **A**: COQ = Cost of Conformance + Cost of Non-Conformance
    - Conformance: Prevention costs, Appraisal costs
    - Non-Conformance: Internal failure, External failure

---

## 4. Verification Layer (VL) Questions / 验证理论层问题

### 4.1 VL-3.1 Model Checking / 模型检验

**对应章节 / Source**: [03-formal-verification/verification-theory.md](../03-formal-verification/verification-theory.md)、[model-checking.md](../03-formal-verification/model-checking.md) — 模型检验问题、反例、LTL/CTL

#### Definition Recall / 定义回忆

1. **Q**: What is the model checking problem?

    **A**: Given a Kripke structure M and a temporal logic formula φ, determine whether M ⊨ φ (M satisfies φ).

2. **Q**: What is a counterexample in model checking?

    **A**: An execution trace that demonstrates a property violation. For safety properties, it's a finite path to an error state. For liveness, it's a path with an infinite loop that never satisfies the property.

#### Application / 应用

1. **Q**: Write a CTL formula for: "From any state, it's possible to reach project completion."

    **A**: AG(EF(phase = "Completed"))

2. **Q**: Write an LTL formula for: "A task cannot be marked complete before it starts."

    **A**: □(task_complete → ○⁻task_started) or □¬(¬task_started ∧ task_complete)

### 4.2 VL-3.2 Theorem Proving / 定理证明

**对应章节 / Source**: [03-formal-verification/verification-theory.md](../03-formal-verification/verification-theory.md)、[theorem-proving.md](../03-formal-verification/theorem-proving.md) — 定理证明、Hoare 逻辑

#### Concept Explanation / 概念解释

1. **Q**: Explain the difference between model checking and theorem proving.

    **A**:

    | Aspect | Model Checking | Theorem Proving |
    |--------|----------------|-----------------|
    | State space | Finite | Can handle infinite |
    | Automation | Fully automatic | Often interactive |
    | Output | Counterexamples | Mathematical proof |
    | Scalability | State explosion | Depends on proof complexity |

---

## 5. Application Layer (AL) Questions / 应用模型层问题

### 5.1 Software Development / 软件开发

**对应章节 / Source**: [04-industry-applications/software-development/agile-models.md](../04-industry-applications/software-development/agile-models.md) 等

1. **Q**: What are the key ceremonies in Scrum?

    **A**: Sprint Planning, Daily Standup, Sprint Review, Sprint Retrospective

2. **Q**: Explain the difference between Continuous Integration and Continuous Deployment.

    **A**:
    - **CI**: Automatically build and test code on every commit
    - **CD**: Automatically deploy tested code to production

### 5.2 Emerging Technologies / 新兴技术

**对应章节 / Source**: [04-industry-applications/ai-management/ai-management.md](../04-industry-applications/ai-management/ai-management.md)、[blockchain-management](../04-industry-applications/blockchain-management/) 等

1. **Q**: What unique challenges does AI project management face?

    **A**:
    - Unpredictable training times
    - Data quality dependencies
    - Model performance uncertainty
    - Ethical considerations
    - Explainability requirements

2. **Q**: How does blockchain technology affect project governance?

    **A**: Enables decentralized decision-making, immutable audit trails, smart contract automation, and transparent stakeholder voting.

---

## 6. Integration Questions / 综合问题

### 6.1 Cross-Layer Integration / 跨层次综合

1. **Q**: How do the five knowledge layers relate to each other? Describe the flow.

    **A**:

    ```
    FL (Foundations) → Mathematical/logical basis
         ↓
    CML (Core Models) → PM-specific formal models
         ↓
    VL (Verification) → Prove properties of models
         ↓
    AL (Application) → Industry-specific adaptations
         ↓
    IL (Implementation) → Executable code
    ```

2. **Q**: Design a verification approach for ensuring a project schedule meets all deadlines.

    **A**:
    1. Model schedule as Kripke structure (states = task completion)
    2. Define property: □(∀t ∈ Tasks: completion_time(t) ≤ deadline(t))
    3. Use model checker (TLC, SPIN) to verify
    4. If counterexample found, adjust schedule

---

## 7. Self-Assessment Scoring / 自我评估评分

### 7.1 Scoring Guide / 评分指南

| Score | Level | Next Action |
|-------|-------|-------------|
| 0-30% | Needs Review | Re-study material, then retry |
| 31-50% | Developing | Focus review on weak areas |
| 51-70% | Proficient | Continue with spaced repetition |
| 71-90% | Advanced | Move to harder problems |
| 91-100% | Expert | Ready for application/teaching |

### 7.2 Practice Protocol / 练习协议

1. **Initial Test**: Answer without notes
2. **Score**: Count correct answers
3. **Review**: Study incorrect answers
4. **Retest**: Focus on missed questions
5. **Space**: Wait before next session

### 7.3 与三层解释 25 概念覆盖 (Coverage of 25 THREE_LAYER Concepts)

以下表格将 [THREE_LAYER_EXPLANATIONS.md](../THREE_LAYER_EXPLANATIONS.md) 中 25 个概念与本问题库对应；**难度**见 [04-concept-difficulty-ranking.md](./04-concept-difficulty-ranking.md)。缺失或薄弱概念已在对应节补题。

| # | 概念 (Concept) | 对应问题节 | 难度 |
|---|----------------|------------|------|
| 1 | Project Definition | §3.1 生命周期（过程组/阶段） | 1 |
| 2 | State Space | §2.1 Kripke（S 即状态集）、§7.4 补题 | 2 |
| 3 | Transition Systems | §2.1 状态转换系统 | 2 |
| 4 | Set Theory Basics | §2.2 数学模型、§7.4 补题 | 1 |
| 5 | Graph Theory | §3.2 资源（依赖/分配）、§7.4 Critical Path | 2 |
| 6 | Project Phases | §3.1 过程组 | 1 |
| 7 | PMBOK Process Groups | §3.1 定义回忆 Q1 | 1 |
| 8 | Risk Definition | §3.3 定义回忆、§7.4 风险五元组 | 2 |
| 9 | Quality Definition | §3.4 QA/QC | 2 |
| 10 | Resource Types | §3.2 资源、§7.4 补题 | 2 |
| 11 | Phase Transitions | §3.1、§7.4 阶段转换 | 3 |
| 12 | Probability Basics | §2.2 MDP、§3.3 风险暴露 | 2 |
| 13 | Syntax vs Semantics | §7.4 补题 | 3 |
| 14 | Kripke Structures | §2.1 定义回忆 Q1 | 3 |
| 15 | LTL | §2.1、§4.1 应用 | 3 |
| 16 | Resource Allocation | §3.2 应用 | 2 |
| 17 | Risk Identification | §3.3、§7.4 补题 | 2 |
| 18 | QA vs QC | §3.4 定义回忆 Q1 | 2 |
| 19 | Formal Lifecycle Model | §3.1、§7.4 补题 | 3 |
| 20 | Model Checking | §4.1 模型检验 | 4 |
| 21 | Critical Path | §7.4 补题 | 3 |
| 22 | WBS | §7.4 补题 | 2 |
| 23 | Theorem Proving | §4.2 定理证明 vs 模型检验 | 4 |
| 24 | CTL | §4.1 应用 Q1 | 4 |
| 25 | Verification vs Validation | §7.4 补题 | 3 |

### 7.4 补题：覆盖 25 概念中薄弱项 (Additional Questions for 25-Concept Coverage)

**对应文档**: [THREE_LAYER_EXPLANATIONS.md](../THREE_LAYER_EXPLANATIONS.md)、[04-concept-difficulty-ranking.md](./04-concept-difficulty-ranking.md)

#### State Space, Set Theory, Phase Transitions (FL/CML)

1. **Q** (State Space): What is the state space of a project in our formal model? **A**: The set of all possible project configurations (e.g. phase, resource allocation, risk register). In Kripke structure it is the set S. See [01-foundations/README.md](../01-foundations/README.md) and [verification-theory.md](../03-formal-verification/verification-theory.md).

2. **Q** (Set Theory): How are sets used in project models (tasks, resources, phases)? **A**: Tasks T, Resources R, Phases P are sets; relations (e.g. allocation, dependency) are subsets of Cartesian products (e.g. D ⊆ T×T). See [mathematical-models.md](../01-foundations/mathematical-models.md).

3. **Q** (Phase Transitions): What is a phase transition in the formal lifecycle? **A**: A valid move from one project phase to another under defined conditions (e.g. gate criteria, deliverables). Formally: transition relation or function between states/phases. See [lifecycle-models.md](../02-project-management/lifecycle-models.md) formal lifecycle.

4. **Q** (Formal Lifecycle Model): Name one safety and one liveness property for a project lifecycle. **A**: Safety: "Never exceed budget" (□¬over_budget). Liveness: "Project eventually reaches Closing" (◇phase=Closing). See [lifecycle-models.md](../02-project-management/lifecycle-models.md) §3 Properties.

#### Risk, WBS, Critical Path (CML)

1. **Q** (Risk Definition): What are the five components of the risk quintuple in risk-models? **A**: Event E, Probability P, Impact I, Time window T, Response cost C (or equivalent per [risk-models.md](../02-project-management/risk-models.md) definition 2.3.1).

2. **Q** (WBS): What is a WBS and why is it important for scope and schedule? **A**: Work Breakdown Structure: hierarchical decomposition of deliverables into work packages. Basis for scope baseline, scheduling, and resource estimation. See [lifecycle-models.md](../02-project-management/lifecycle-models.md), [PMBOK_8_ALIGNMENT_PLAN.md](../PMBOK_8_ALIGNMENT_PLAN.md) Scope.

3. **Q** (Critical Path): What is the critical path and why does it determine project duration? **A**: The longest path from start to finish with zero slack; any delay on this path delays the project. See [resource-models.md](../02-project-management/resource-models.md) §2.2.3 CPM/PERT, [THREE_LAYER_EXPLANATIONS.md](../THREE_LAYER_EXPLANATIONS.md) §21.

#### VL: Syntax vs Semantics, Verification vs Validation

1. **Q** (Syntax vs Semantics): In formal specification, what is the difference between syntax and semantics? **A**: Syntax: rules for well-formed formulas (how to write). Semantics: meaning of formulas in a structure (what they denote). See [semantic-models.md](../01-foundations/semantic-models.md).

2. **Q** (Verification vs Validation): State the difference between verification and validation in one sentence each. **A**: Verification: "Did we build it right?" (conformance to spec). Validation: "Did we build the right thing?" (meeting needs). See [verification-theory.md](../03-formal-verification/verification-theory.md), [quality-models.md](../02-project-management/quality-models.md).

---

## 8. Status / 状态

**Document Version / 文档版本**: 1.1
**Last Updated / 最后更新**: 2026-02-04
**Status / 状态**: ✅ Complete；25 概念覆盖表与补题已加入（§7.3–7.4）
**Total Questions / 总问题数**: 40+ across all layers（含 §7.4 补题 9 道）
**Next Review / 下次审查**: 2026-05-02（季度审查时核对 25 概念与 [04-concept-difficulty-ranking.md](./04-concept-difficulty-ranking.md) 一致性）

**Related Documents / 相关文档**:

- [Learning Prerequisites](./01-learning-prerequisites.md)
- [Spaced Repetition Schedule](./02-spaced-repetition-schedule.md)
- [Concept Difficulty Ranking](./04-concept-difficulty-ranking.md)
- [Interleaved Learning Paths](./05-interleaved-learning-paths.md)

**Bidirectional links**: Each question section above links to the source doc (FL/CML/VL/AL). To link from content docs back to questions, add in their "Related Documents": [12-learning-support/03-retrieval-practice-questions.md](../12-learning-support/03-retrieval-practice-questions.md).
