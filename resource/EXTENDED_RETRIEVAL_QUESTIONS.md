# Extended Retrieval Practice Questions / 扩展检索练习问题库

## 1. Overview / 概述

This document extends the retrieval practice question bank to provide comprehensive coverage of all core concepts (5-10 questions per concept).

本文档扩展检索练习问题库，为所有核心概念提供全面覆盖（每个概念5-10题）。

---

## 2. Foundation Layer (FL) Extended Questions / 基础理论层扩展问题

### 2.1 FL-1.1 Kripke Structure / Kripke结构 (10 Questions)

1. **Q**: What are the four components of a Kripke structure?
   **A**: S (states), S₀ (initial states), R (transition relation), L (labeling function)

2. **Q**: What does the labeling function L do in a Kripke structure?
   **A**: It maps each state to the set of atomic propositions true in that state.

3. **Q**: Can a Kripke structure have multiple initial states?
   **A**: Yes, S₀ is a subset of S, so it can contain multiple initial states.

4. **Q**: What is the difference between a Kripke structure and a finite state machine?
   **A**: Kripke structures have state labels (propositions), while FSMs have edge labels (input/output).

5. **Q**: Draw a Kripke structure for a 3-phase project (init, exec, done).
   **A**: States: {init, exec, done}; S₀={init}; R={(init,exec),(exec,done)}; L(done)={complete}

6. **Q**: What does it mean for a transition relation R to be total?
   **A**: Every state has at least one successor: ∀s∈S: ∃s'∈S: (s,s')∈R

7. **Q**: How do you represent a deadlock state in a Kripke structure?
   **A**: A state with no outgoing transitions (violates totality, or add self-loop)

8. **Q**: What atomic propositions would you use for a risk management Kripke structure?
   **A**: Examples: risk_identified, risk_analyzed, risk_mitigated, risk_closed

9. **Q**: How does a Kripke structure relate to model checking?
   **A**: Model checking verifies whether a Kripke structure satisfies a temporal logic property.

10. **Q**: What is the state explosion problem in Kripke structures?
    **A**: The number of states grows exponentially with the number of variables/components.

### 2.2 FL-1.1 LTL Temporal Logic / LTL时序逻辑 (10 Questions)

1. **Q**: What does LTL stand for?
   **A**: Linear Temporal Logic

2. **Q**: What are the four main temporal operators in LTL?
   **A**: ○ (Next), ◇ (Eventually), □ (Always), U (Until)

3. **Q**: Write an LTL formula for "The project will eventually complete."
   **A**: ◇ complete

4. **Q**: Write an LTL formula for "Budget is always non-negative."
   **A**: □ (budget ≥ 0)

5. **Q**: What is the difference between ◇φ and □◇φ?
   **A**: ◇φ means φ holds at least once; □◇φ means φ holds infinitely often.

6. **Q**: Write an LTL formula for "If a risk is identified, it will eventually be mitigated."
   **A**: □ (risk_identified → ◇ risk_mitigated)

7. **Q**: What type of property is □(budget ≥ 0)?
   **A**: Safety property (something bad never happens)

8. **Q**: What type of property is ◇ complete?
   **A**: Liveness property (something good eventually happens)

9. **Q**: Write the LTL for "Task A must complete before Task B starts."
   **A**: ¬task_B_started U task_A_complete

10. **Q**: How does LTL differ from CTL?
    **A**: LTL has path quantifiers implicit (all paths), CTL has explicit path quantifiers (A/E).

### 2.3 FL-1.2 MDP / 马尔可夫决策过程 (10 Questions)

1. **Q**: What are the five components of an MDP?
   **A**: S (states), A (actions), P (transition probabilities), R (rewards), γ (discount factor)

2. **Q**: What is the purpose of the discount factor γ?
   **A**: To weight immediate rewards more than future rewards (0 < γ ≤ 1).

3. **Q**: What is a policy in an MDP?
   **A**: A mapping from states to actions: π: S → A

4. **Q**: What does the value function V(s) represent?
   **A**: Expected cumulative discounted reward starting from state s.

5. **Q**: Write the Bellman equation for V(s).
   **A**: V(s) = max_a [R(s,a) + γ Σ P(s'|s,a)V(s')]

6. **Q**: What is the difference between value iteration and policy iteration?
   **A**: Value iteration updates value function; policy iteration evaluates then improves policy.

7. **Q**: How can MDP be applied to project risk decisions?
   **A**: States = risk levels, Actions = mitigation options, Rewards = risk reduction.

8. **Q**: What makes an MDP "Markov"?
   **A**: Future depends only on current state, not history (memoryless property).

9. **Q**: What is a reward function in project context?
   **A**: Could be: -cost for actions, +value for deliverables, -penalty for delays.

10. **Q**: How do you find the optimal policy?
    **A**: Solve Bellman equations using value iteration, policy iteration, or linear programming.

---

## 3. Core Model Layer (CML) Extended Questions / 核心模型层扩展问题

### 3.1 CML-2.1 Project Lifecycle / 项目生命周期 (10 Questions)

1. **Q**: What are the five PMBOK process groups?
   **A**: Initiating, Planning, Executing, Monitoring & Controlling, Closing

2. **Q**: What is the primary deliverable of the Initiating process group?
   **A**: Project charter

3. **Q**: What is a phase gate?
   **A**: A decision point between phases where go/no-go decisions are made.

4. **Q**: What is the difference between a phase and a process group?
   **A**: Phases are sequential project divisions; process groups can repeat within phases.

5. **Q**: What are the 8 Performance Domains in PMBOK 7th Edition?
   **A**: Stakeholders, Team, Development Approach, Planning, Project Work, Delivery, Measurement, Uncertainty

6. **Q**: What is progressive elaboration?
   **A**: The iterative refinement of project details as more information becomes available.

7. **Q**: How do you formally model a lifecycle as a state machine?
   **A**: States = phases, Transitions = phase gates, Properties = phase completion criteria.

8. **Q**: What triggers the transition from Planning to Execution?
   **A**: Approval of the project management plan.

9. **Q**: What activities occur in the Monitoring & Controlling process group?
   **A**: Performance measurement, variance analysis, change control, corrective actions.

10. **Q**: What is the relationship between lifecycle and methodology?
    **A**: Lifecycle defines phases; methodology defines how work is done within phases.

### 3.2 CML-2.2 Resource Management / 资源管理 (8 Questions)

1. **Q**: What are the four main types of project resources?
   **A**: Human, material, equipment, financial

2. **Q**: What is resource leveling?
   **A**: Adjusting schedule to resolve resource over-allocations while respecting constraints.

3. **Q**: What is resource smoothing?
   **A**: Adjusting activities within float to reduce peaks, without changing critical path.

4. **Q**: Write a constraint for "Resource capacity cannot be exceeded."
   **A**: ∀r∈Resources: Σ(allocation(t,r) for t∈ActiveTasks) ≤ capacity(r)

5. **Q**: What is the difference between assignment and allocation?
   **A**: Assignment = which resource does the task; Allocation = how much capacity.

6. **Q**: What is a responsibility assignment matrix (RAM)?
   **A**: Matrix showing who is Responsible, Accountable, Consulted, Informed (RACI).

7. **Q**: How does multi-project resource management differ from single project?
   **A**: Must consider resource sharing, priority, and cross-project dependencies.

8. **Q**: What is resource calendar?
   **A**: Shows when resources are available (working days, hours, vacations).

### 3.3 CML-2.3 Risk Management / 风险管理 (10 Questions)

1. **Q**: What is the definition of project risk?
   **A**: An uncertain event that, if it occurs, has an effect on project objectives.

2. **Q**: What are the four risk response strategies for threats?
   **A**: Avoid, Mitigate, Transfer, Accept

3. **Q**: What are the three risk response strategies for opportunities?
   **A**: Exploit, Enhance, Share

4. **Q**: How is risk exposure calculated?
   **A**: Risk Exposure = Probability × Impact

5. **Q**: What is a risk register?
   **A**: A document listing identified risks with analysis and response plans.

6. **Q**: What is the difference between qualitative and quantitative risk analysis?
   **A**: Qualitative = categorical (H/M/L); Quantitative = numerical (probabilities, values).

7. **Q**: What is Monte Carlo simulation used for in risk management?
   **A**: To model probability distributions of project outcomes (cost, schedule).

8. **Q**: What is secondary risk?
   **A**: A risk that arises as a direct result of implementing a risk response.

9. **Q**: What is residual risk?
   **A**: Risk remaining after risk responses have been implemented.

10. **Q**: How does ISO 31000 define risk?
    **A**: The effect of uncertainty on objectives.

### 3.4 CML-2.4 Quality Management / 质量管理 (8 Questions)

1. **Q**: What is the difference between quality assurance and quality control?
   **A**: QA = process-focused, preventive; QC = product-focused, detective.

2. **Q**: What is the Cost of Quality (COQ)?
   **A**: Total cost of conformance (prevention + appraisal) and non-conformance (internal + external failure).

3. **Q**: What is a control chart?
   **A**: A graph showing process variation over time with control limits.

4. **Q**: What does Six Sigma refer to?
   **A**: 3.4 defects per million opportunities (6 standard deviations from mean).

5. **Q**: What is the PDCA cycle?
   **A**: Plan-Do-Check-Act, also known as Deming cycle.

6. **Q**: What is a quality audit?
   **A**: Independent review to verify compliance with quality standards and processes.

7. **Q**: What is the difference between grade and quality?
   **A**: Grade = category/ranking; Quality = degree to which requirements are met.

8. **Q**: What is continuous improvement?
   **A**: Ongoing effort to improve products, services, or processes incrementally.

---

## 4. Verification Layer (VL) Extended Questions / 验证理论层扩展问题

### 4.1 VL-3.1 Model Checking / 模型检验 (10 Questions)

1. **Q**: What is the model checking problem?
   **A**: Given model M and property φ, determine if M ⊨ φ.

2. **Q**: What are the two main types of properties checked?
   **A**: Safety (bad things don't happen) and Liveness (good things happen).

3. **Q**: What is a counterexample?
   **A**: An execution trace that violates the property being checked.

4. **Q**: What is state explosion?
   **A**: Exponential growth in states making exhaustive checking infeasible.

5. **Q**: Name two techniques to handle state explosion.
   **A**: Abstraction, symbolic representation (BDDs), bounded model checking.

6. **Q**: What is the difference between explicit and symbolic model checking?
   **A**: Explicit enumerates states; symbolic uses BDDs to represent state sets.

7. **Q**: What is bounded model checking?
   **A**: Checking properties up to a fixed depth/bound, often using SAT solvers.

8. **Q**: What does the TLC model checker check?
   **A**: TLA+ specifications for safety and liveness properties.

9. **Q**: How do you specify a safety property in LTL?
   **A**: □¬bad_state or □good_condition

10. **Q**: What is the advantage of model checking over testing?
    **A**: Exhaustive coverage of all reachable states, not just sampled paths.

### 4.2 VL-3.2 Theorem Proving / 定理证明 (8 Questions)

1. **Q**: How does theorem proving differ from model checking?
   **A**: Theorem proving handles infinite states; model checking is for finite states.

2. **Q**: What is an interactive theorem prover?
   **A**: A tool where the user guides the proof with tactics (e.g., Coq, Isabelle, Lean).

3. **Q**: What is the Curry-Howard correspondence?
   **A**: The isomorphism between proofs and programs, types and propositions.

4. **Q**: What is a proof tactic?
   **A**: A command that transforms proof goals (e.g., intro, apply, induction).

5. **Q**: Name three interactive theorem provers.
   **A**: Coq, Isabelle/HOL, Lean

6. **Q**: What is an SMT solver?
   **A**: Satisfiability Modulo Theories solver - checks satisfiability with theories.

7. **Q**: How can Z3 be used for project verification?
   **A**: To verify constraints, find optimal allocations, check schedule feasibility.

8. **Q**: What is the advantage of theorem proving over model checking?
   **A**: Can handle infinite state spaces and prove universal properties.

---

## 5. Application Layer (AL) Extended Questions / 应用模型层扩展问题

### 5.1 AL-4.1 Software Development / 软件开发 (8 Questions)

1. **Q**: What are the four values of the Agile Manifesto?
   **A**: Individuals/interactions, Working software, Customer collaboration, Responding to change

2. **Q**: What are the Scrum ceremonies?
   **A**: Sprint Planning, Daily Standup, Sprint Review, Sprint Retrospective

3. **Q**: What is a Sprint?
   **A**: A time-boxed iteration (1-4 weeks) producing a potentially shippable increment.

4. **Q**: What is DevOps?
   **A**: Culture and practices unifying development and operations for continuous delivery.

5. **Q**: What is CI/CD?
   **A**: Continuous Integration / Continuous Deployment - automated build, test, deploy.

6. **Q**: What is technical debt?
   **A**: Implied cost of future rework caused by choosing quick solutions over better approaches.

7. **Q**: What is the difference between Kanban and Scrum?
   **A**: Kanban = continuous flow, WIP limits; Scrum = time-boxed sprints, roles.

8. **Q**: What is Definition of Done?
   **A**: Shared understanding of what "complete" means for a work item.

### 5.2 AL-4.4+ Emerging Technologies / 新兴技术 (6 Questions)

1. **Q**: What unique challenges does AI project management face?
   **A**: Data quality, unpredictable training, ethical concerns, explainability.

2. **Q**: What is MLOps?
   **A**: DevOps practices applied to machine learning model lifecycle.

3. **Q**: How does blockchain affect project governance?
   **A**: Enables decentralized decisions, immutable records, smart contracts.

4. **Q**: What is the challenge of quantum computing projects?
   **A**: Hardware limitations, error correction, new algorithms, talent scarcity.

5. **Q**: What is edge computing project management?
   **A**: Managing distributed systems across edge nodes with latency constraints.

6. **Q**: What are the unique aspects of Web3/metaverse projects?
   **A**: Decentralization, tokenomics, virtual collaboration, DAO governance.

---

## 6. Complexity & Systems Questions / 复杂性与系统问题

### 6.1 Cynefin Framework (6 Questions)

1. **Q**: What are the five domains of the Cynefin framework?
   **A**: Clear (Obvious), Complicated, Complex, Chaotic, Confused (Disorder)

2. **Q**: What is the approach for the Complex domain?
   **A**: Probe → Sense → Respond

3. **Q**: What is the approach for the Complicated domain?
   **A**: Sense → Analyze → Respond

4. **Q**: When should you use Agile methods according to Cynefin?
   **A**: In the Complex domain where outcomes are emergent.

5. **Q**: What is the "cliff" between Complex and Chaotic?
   **A**: The boundary where loss of control causes sudden transition to chaos.

6. **Q**: What project management approach fits the Clear domain?
   **A**: Standardized processes, best practices, traditional PM.

### 6.2 Systems Dynamics (6 Questions)

1. **Q**: What are the basic elements of systems dynamics?
   **A**: Stocks, Flows, Feedback loops, Delays

2. **Q**: What is a reinforcing loop?
   **A**: A feedback loop that amplifies change (positive feedback).

3. **Q**: What is a balancing loop?
   **A**: A feedback loop that seeks equilibrium (negative feedback).

4. **Q**: What is the "Rework Cycle" archetype?
   **A**: Pressure → Speed up → Lower quality → More rework → More pressure.

5. **Q**: How does adding people to a late project make it later (Brooks's Law)?
   **A**: Training overhead, communication overhead, productivity dip.

6. **Q**: What is a system archetype?
   **A**: A common pattern of system behavior (e.g., Fixes that Fail, Limits to Growth).

---

## 7. Self-Assessment / 自我评估

### 7.1 Scoring Guide / 评分指南

| Score | Interpretation | Next Steps |
|-------|----------------|------------|
| 0-40% | Needs significant review | Re-study fundamentals |
| 41-60% | Developing | Focus on weak areas |
| 61-80% | Proficient | Continue practice, try advanced |
| 81-100% | Expert | Ready to apply/teach |

### 7.2 Topic Coverage / 主题覆盖

| Topic | Questions | Target Score |
|-------|-----------|--------------|
| FL: Kripke/LTL | 20 | 16+ |
| FL: MDP | 10 | 8+ |
| CML: Lifecycle | 10 | 8+ |
| CML: Resources | 8 | 6+ |
| CML: Risk | 10 | 8+ |
| CML: Quality | 8 | 6+ |
| VL: Model Checking | 10 | 8+ |
| VL: Theorem Proving | 8 | 6+ |
| AL: Software | 8 | 6+ |
| AL: Emerging | 6 | 4+ |
| Complexity | 12 | 10+ |
| **Total** | **110** | **86+** |

---

## 8. Status / 状态

**Document Version / 文档版本**: 1.0
**Last Updated / 最后更新**: 2026-02-02
**Status / 状态**: ✅ Complete
**Total Questions / 总问题数**: 110+
**Next Review / 下次审查**: 2026-05-02

**Related Documents / 相关文档**:

- [Retrieval Practice Questions (Core)](../docs/12-learning-support/03-retrieval-practice-questions.md)
- [Learning Prerequisites](../docs/12-learning-support/01-learning-prerequisites.md)
- [Spaced Repetition Schedule](../docs/12-learning-support/02-spaced-repetition-schedule.md)
