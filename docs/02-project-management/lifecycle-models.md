# 2.1 项目生命周期模型

## 概述

项目生命周期模型是Formal-ProgramManage的核心理论之一，定义了项目从启动到收尾的完整演进过程。本理论体系严格对标PMBOK 7th Edition、ISO 21500:2012、PRINCE2 2017、APM Body of Knowledge 7th Edition等国际项目管理标准。

## 2.1.1 生命周期基础理论

### 基本定义

**定义 2.1.1** (项目生命周期 - PMBOK 7th Edition) 项目生命周期是一个四元组：
$$\mathcal{L} = (P, T, G, C)$$

其中：

- $P = \{p_1, p_2, \ldots, p_n\}$ 是阶段集合，满足 $p_i \cap p_j = \emptyset$ 对于 $i \neq j$
- $T = \{t_1, t_2, \ldots, t_m\}$ 是转换点集合，满足 $t_i < t_{i+1}$
- $G = \{g_1, g_2, \ldots, g_k\}$ 是关口集合，满足 $g_i \subseteq P \times P$
- $C: P \times T \rightarrow \mathbb{R}^+$ 是成本函数，满足 $C(p,t) \geq 0$

**定义 2.1.2** (项目阶段) 项目阶段是一个五元组：
$$p = (S, A, D, O, M)$$

其中：

- $S$ 是阶段状态集合，满足 $S = \{\text{Initiated}, \text{Planning}, \text{Executing}, \text{Monitoring}, \text{Closing}\}$
- $A$ 是阶段活动集合，满足 $A \subseteq \mathcal{A}$
- $D$ 是阶段交付物集合，满足 $D \subseteq \mathcal{D}$
- $O$ 是阶段目标集合，满足 $O \subseteq \mathcal{O}$
- $M$ 是阶段度量指标集合，满足 $M: \mathcal{M} \rightarrow \mathbb{R}$

**定义 2.1.3** (生命周期转换) 生命周期转换是一个函数：
$$\text{transition}: P \times E \rightarrow P$$

其中 $E$ 是事件集合，包含：

- $\text{phase\_complete}$: 阶段完成事件
- $\text{gate\_approved}$: 关口批准事件
- $\text{change\_requested}$: 变更请求事件
- $\text{risk\_triggered}$: 风险触发事件

## 2.1.2 标准生命周期模型

### PMBOK 7th Edition 生命周期

**定义 2.1.4** (PMBOK生命周期) PMBOK生命周期包含五个过程组：
$$\mathcal{L}_{PMBOK} = (\text{Initiating}, \text{Planning}, \text{Executing}, \text{Monitoring \& Controlling}, \text{Closing})$$

**阶段 2.1.1** (启动过程组) 启动过程组 $I$ 满足：
$$I = \{i_1, i_2, \ldots, i_n\}$$

其中：

- $i_1$: 制定项目章程
- $i_2$: 识别相关方
- $i_3$: 启动项目

**阶段 2.1.2** (规划过程组) 规划过程组 $P$ 满足：
$$P = \{p_1, p_2, \ldots, p_m\}$$

其中：

- $p_1$: 制定项目管理计划
- $p_2$: 规划范围管理
- $p_3$: 收集需求
- $p_4$: 定义范围
- $p_5$: 创建工作分解结构
- $p_6$: 规划进度管理
- $p_7$: 定义活动
- $p_8$: 排列活动顺序
- $p_9$: 估算活动持续时间
- $p_{10}$: 制定进度计划
- $p_{11}$: 规划成本管理
- $p_{12}$: 估算成本
- $p_{13}$: 制定预算
- $p_{14}$: 规划质量管理
- $p_{15}$: 规划资源管理
- $p_{16}$: 估算活动资源
- $p_{17}$: 规划沟通管理
- $p_{18}$: 规划风险管理
- $p_{19}$: 识别风险
- $p_{20}$: 实施定性风险分析
- $p_{21}$: 实施定量风险分析
- $p_{22}$: 规划风险应对
- $p_{23}$: 规划采购管理
- $p_{24}$: 规划相关方参与

**阶段 2.1.3** (执行过程组) 执行过程组 $E$ 满足：
$$E = \{e_1, e_2, \ldots, e_k\}$$

其中：

- $e_1$: 指导与管理项目工作
- $e_2$: 管理项目知识
- $e_3$: 管理质量
- $e_4$: 获取资源
- $e_5$: 建设团队
- $e_6$: 管理团队
- $e_7$: 管理沟通
- $e_8$: 实施风险应对
- $e_9$: 实施采购
- $e_{10}$: 管理相关方参与

**阶段 2.1.4** (监控过程组) 监控过程组 $M$ 满足：
$$M = \{m_1, m_2, \ldots, m_l\}$$

其中：

- $m_1$: 监控项目工作
- $m_2$: 执行整体变更控制
- $m_3$: 确认范围
- $m_4$: 控制范围
- $m_5$: 控制进度
- $m_6$: 控制成本
- $m_7$: 控制质量
- $m_8$: 控制资源
- $m_9$: 监督沟通
- $m_{10}$: 监督风险
- $m_{11}$: 控制采购
- $m_{12}$: 监督相关方参与

**阶段 2.1.5** (收尾过程组) 收尾过程组 $C$ 满足：
$$C = \{c_1, c_2\}$$

其中：

- $c_1$: 结束项目或阶段
- $c_2$: 结束采购

### ISO 21500 生命周期

**定义 2.1.5** (ISO 21500生命周期) ISO 21500生命周期包含五个过程组：
$$\mathcal{L}_{ISO} = (\text{Initiating}, \text{Planning}, \text{Implementing}, \text{Controlling}, \text{Closing})$$

**定理 2.1.1** (生命周期等价性) PMBOK和ISO 21500生命周期在语义上等价：
$$\mathcal{L}_{PMBOK} \equiv \mathcal{L}_{ISO}$$

### PRINCE2 生命周期

**定义 2.1.6** (PRINCE2生命周期) PRINCE2生命周期包含七个主题：
$$\mathcal{L}_{PRINCE2} = (\text{Business Case}, \text{Organization}, \text{Quality}, \text{Plans}, \text{Risk}, \text{Change}, \text{Progress})$$

**阶段 2.1.6** (PRINCE2阶段) PRINCE2包含七个过程：

1. **Starting Up a Project (SU)**: 项目启动
2. **Initiating a Project (IP)**: 项目初始化
3. **Directing a Project (DP)**: 项目指导
4. **Controlling a Stage (CS)**: 阶段控制
5. **Managing Product Delivery (MP)**: 产品交付管理
6. **Managing a Stage Boundary (SB)**: 阶段边界管理
7. **Closing a Project (CP)**: 项目收尾

## 2.1.3 形式化生命周期模型

### 状态转换系统

**定义 2.1.7** (生命周期状态转换系统) 生命周期状态转换系统是一个五元组：
$$LTS = (S, S_0, \Sigma, \delta, F)$$

其中：

- $S$ 是状态集合，满足 $S = \{\text{Initiated}, \text{Planning}, \text{Executing}, \text{Monitoring}, \text{Closing}, \text{Completed}\}$
- $S_0 = \{\text{Initiated}\}$ 是初始状态集合
- $\Sigma$ 是事件字母表，包含生命周期事件
- $\delta: S \times \Sigma \rightarrow S$ 是状态转换函数
- $F = \{\text{Completed}\}$ 是最终状态集合

**定义 2.1.8** (生命周期事件) 生命周期事件集合：
$$\Sigma = \{\text{start\_planning}, \text{planning\_complete}, \text{start\_execution}, \text{execution\_complete}, \text{start\_monitoring}, \text{monitoring\_complete}, \text{start\_closing}, \text{closing\_complete}\}$$

### 转换函数定义

**定义 2.1.9** (生命周期转换函数) 转换函数 $\delta$ 定义为：

$$\begin{align}
\delta(\text{Initiated}, \text{start\_planning}) &= \text{Planning} \\
\delta(\text{Planning}, \text{planning\_complete}) &= \text{Executing} \\
\delta(\text{Executing}, \text{start\_monitoring}) &= \text{Monitoring} \\
\delta(\text{Monitoring}, \text{monitoring\_complete}) &= \text{Executing} \\
\delta(\text{Executing}, \text{execution\_complete}) &= \text{Closing} \\
\delta(\text{Closing}, \text{closing\_complete}) &= \text{Completed}
\end{align}$$

### 生命周期属性

**定义 2.1.10** (生命周期安全性属性) 生命周期安全性属性：
$$\phi_{safety} = \mathbf{G}(\text{Completed} \Rightarrow \text{all\_deliverables\_produced})$$

**定义 2.1.11** (生命周期活性属性) 生命周期活性属性：
$$\phi_{liveness} = \mathbf{G}(\text{Initiated} \Rightarrow \mathbf{F}\text{Completed})$$

**定义 2.1.12** (生命周期公平性属性) 生命周期公平性属性：
$$\phi_{fairness} = \mathbf{G}\mathbf{F}(\text{Monitoring})$$

## 2.1.4 生命周期验证

### 验证方法

**算法 2.1.1** (生命周期验证算法)：

```rust
use std::collections::{HashMap, HashSet};

# [derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LifecycleState {
    Initiated,
    Planning,
    Executing,
    Monitoring,
    Closing,
    Completed,
}

# [derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LifecycleEvent {
    StartPlanning,
    PlanningComplete,
    StartExecution,
    ExecutionComplete,
    StartMonitoring,
    MonitoringComplete,
    StartClosing,
    ClosingComplete,
}

# [derive(Debug, Clone)]
pub struct LifecycleTransition {
    pub from: LifecycleState,
    pub event: LifecycleEvent,
    pub to: LifecycleState,
    pub conditions: Vec<Condition>,
    pub actions: Vec<Action>,
}

# [derive(Debug, Clone)]
pub struct Condition {
    pub name: String,
    pub predicate: Box<dyn Fn(&ProjectState) -> bool>,
}

# [derive(Debug, Clone)]
pub struct Action {
    pub name: String,
    pub operation: Box<dyn Fn(&mut ProjectState)>,
}

# [derive(Debug, Clone)]
pub struct ProjectState {
    pub current_state: LifecycleState,
    pub deliverables: HashSet<String>,
    pub milestones: HashMap<String, bool>,
    pub resources: HashMap<String, f64>,
    pub timeline: HashMap<String, f64>,
    pub risks: Vec<Risk>,
    pub quality_metrics: HashMap<String, f64>,
}

# [derive(Debug, Clone)]
pub struct Risk {
    pub id: String,
    pub description: String,
    pub probability: f64,
    pub impact: f64,
    pub mitigation: String,
}

# [derive(Debug)]
pub struct LifecycleValidator {
    pub transitions: Vec<LifecycleTransition>,
    pub initial_state: ProjectState,
    pub final_states: HashSet<LifecycleState>,
}

impl LifecycleValidator {
    pub fn new() -> Self {
        LifecycleValidator {
            transitions: Vec::new(),
            initial_state: ProjectState {
                current_state: LifecycleState::Initiated,
                deliverables: HashSet::new(),
                milestones: HashMap::new(),
                resources: HashMap::new(),
                timeline: HashMap::new(),
                risks: Vec::new(),
                quality_metrics: HashMap::new(),
            },
            final_states: HashSet::from([LifecycleState::Completed]),
        }
    }

    pub fn add_transition(&mut self, transition: LifecycleTransition) {
        self.transitions.push(transition);
    }

    pub fn verify_safety_property(&self, project: &ProjectState) -> bool {
        // 验证安全性属性：项目完成时所有交付物都已产生
        if project.current_state == LifecycleState::Completed {
            return self.all_deliverables_produced(project);
        }
        true
    }

    pub fn verify_liveness_property(&self, project: &ProjectState) -> bool {
        // 验证活性属性：从启动状态最终能到达完成状态
        self.can_reach_completion(project)
    }

    pub fn verify_fairness_property(&self, project: &ProjectState) -> bool {
        // 验证公平性属性：监控状态会无限次出现
        self.monitoring_fairness(project)
    }

    fn all_deliverables_produced(&self, project: &ProjectState) -> bool {
        // 检查所有必需的交付物是否都已产生
        let required_deliverables = self.get_required_deliverables();
        required_deliverables.iter().all(|d| project.deliverables.contains(d))
    }

    fn get_required_deliverables(&self) -> HashSet<String> {
        // 定义必需的交付物
        HashSet::from([
            "Project Charter".to_string(),
            "Project Management Plan".to_string(),
            "Work Breakdown Structure".to_string(),
            "Schedule".to_string(),
            "Budget".to_string(),
            "Quality Plan".to_string(),
            "Risk Register".to_string(),
            "Final Report".to_string(),
        ])
    }

    fn can_reach_completion(&self, project: &ProjectState) -> bool {
        // 使用可达性分析检查是否能到达完成状态
        let mut visited = HashSet::new();
        self.dfs_reach_completion(project, &mut visited)
    }

    fn dfs_reach_completion(&self, project: &ProjectState, visited: &mut HashSet<LifecycleState>) -> bool {
        if project.current_state == LifecycleState::Completed {
            return true;
        }

        if visited.contains(&project.current_state) {
            return false;
        }

        visited.insert(project.current_state.clone());

        for transition in &self.transitions {
            if transition.from == project.current_state {
                // 检查转换条件是否满足
                if self.check_transition_conditions(transition, project) {
                    let mut new_state = project.clone();
                    new_state.current_state = transition.to.clone();

                    if self.dfs_reach_completion(&new_state, visited) {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn check_transition_conditions(&self, transition: &LifecycleTransition, project: &ProjectState) -> bool {
        transition.conditions.iter().all(|condition| (condition.predicate)(project))
    }

    fn monitoring_fairness(&self, project: &ProjectState) -> bool {
        // 检查监控公平性：确保监控状态会无限次出现
        // 这需要分析无限路径，简化实现
        true
    }

    pub fn execute_transition(&mut self, project: &mut ProjectState, event: LifecycleEvent) -> Result<(), String> {
        for transition in &self.transitions {
            if transition.from == project.current_state && transition.event == event {
                // 检查转换条件
                if !self.check_transition_conditions(transition, project) {
                    return Err("转换条件不满足".to_string());
                }

                // 执行转换动作
                for action in &transition.actions {
                    (action.operation)(project);
                }

                // 更新状态
                project.current_state = transition.to.clone();
                return Ok(());
            }
        }

        Err("无效的转换".to_string())
    }
}
```

## 2.1.5 生命周期优化

### 优化目标

**定义 2.1.13** (生命周期优化目标) 生命周期优化目标函数：
$$f(\mathcal{L}) = \alpha \cdot \text{Time}(\mathcal{L}) + \beta \cdot \text{Cost}(\mathcal{L}) + \gamma \cdot \text{Quality}(\mathcal{L})$$

其中：
- $\text{Time}(\mathcal{L})$ 是生命周期总时间
- $\text{Cost}(\mathcal{L})$ 是生命周期总成本
- $\text{Quality}(\mathcal{L})$ 是生命周期质量指标
- $\alpha, \beta, \gamma$ 是权重系数，满足 $\alpha + \beta + \gamma = 1$

### 优化算法

**算法 2.1.2** (生命周期优化算法)：

```rust
use std::collections::HashMap;

# [derive(Debug)]
pub struct LifecycleOptimizer {
    pub optimization_objectives: Vec<OptimizationObjective>,
    pub optimization_constraints: Vec<OptimizationConstraint>,
    pub optimization_history: Vec<OptimizationStep>,
}

# [derive(Debug)]
pub struct OptimizationObjective {
    pub name: String,
    pub weight: f64,
    pub function: Box<dyn Fn(&LifecycleModel) -> f64>,
}

# [derive(Debug)]
pub struct OptimizationConstraint {
    pub name: String,
    pub condition: Box<dyn Fn(&LifecycleModel) -> bool>,
}

# [derive(Debug)]
pub struct OptimizationStep {
    pub iteration: usize,
    pub objective_value: f64,
    pub constraint_violations: Vec<String>,
    pub lifecycle_model: LifecycleModel,
}

# [derive(Debug, Clone)]
pub struct LifecycleModel {
    pub phases: Vec<Phase>,
    pub transitions: Vec<Transition>,
    pub resources: HashMap<String, f64>,
    pub timeline: HashMap<String, f64>,
    pub quality_metrics: HashMap<String, f64>,
}

# [derive(Debug, Clone)]
pub struct Phase {
    pub id: String,
    pub name: String,
    pub duration: f64,
    pub cost: f64,
    pub quality_target: f64,
    pub dependencies: Vec<String>,
}

# [derive(Debug, Clone)]
pub struct Transition {
    pub from: String,
    pub to: String,
    pub conditions: Vec<String>,
    pub actions: Vec<String>,
}

impl LifecycleOptimizer {
    pub fn new() -> Self {
        LifecycleOptimizer {
            optimization_objectives: Vec::new(),
            optimization_constraints: Vec::new(),
            optimization_history: Vec::new(),
        }
    }

    pub fn add_objective(&mut self, objective: OptimizationObjective) {
        self.optimization_objectives.push(objective);
    }

    pub fn add_constraint(&mut self, constraint: OptimizationConstraint) {
        self.optimization_constraints.push(constraint);
    }

    pub fn optimize_lifecycle(&mut self, initial_model: LifecycleModel) -> LifecycleModel {
        let mut current_model = initial_model;
        let mut iteration = 0;
        let max_iterations = 1000;

        while iteration < max_iterations {
            let objective_value = self.calculate_objective_value(&current_model);
            let constraint_violations = self.check_constraint_violations(&current_model);

            let step = OptimizationStep {
                iteration,
                objective_value,
                constraint_violations: constraint_violations.clone(),
                lifecycle_model: current_model.clone(),
            };

            self.optimization_history.push(step);

            if constraint_violations.is_empty() {
                // 如果没有约束违反，尝试优化目标
                let improved_model = self.improve_model(&current_model);
                if self.calculate_objective_value(&improved_model) > objective_value {
                    current_model = improved_model;
                } else {
                    break; // 收敛
                }
            } else {
                // 如果有约束违反，修复约束
                current_model = self.repair_constraints(&current_model, &constraint_violations);
            }

            iteration += 1;
        }

        current_model
    }

    fn calculate_objective_value(&self, model: &LifecycleModel) -> f64 {
        let mut total_value = 0.0;

        for objective in &self.optimization_objectives {
            let value = (objective.function)(model);
            total_value += objective.weight * value;
        }

        total_value
    }

    fn check_constraint_violations(&self, model: &LifecycleModel) -> Vec<String> {
        let mut violations = Vec::new();

        for constraint in &self.optimization_constraints {
            if !(constraint.condition)(model) {
                violations.push(constraint.name.clone());
            }
        }

        violations
    }

    fn improve_model(&self, model: &LifecycleModel) -> LifecycleModel {
        let mut improved_model = model.clone();

        // 实现模型改进策略
        // 1. 优化阶段持续时间
        for phase in &mut improved_model.phases {
            if phase.duration > 10.0 {
                phase.duration *= 0.9; // 减少10%
            }
        }

        // 2. 优化资源分配
        for (resource, amount) in &mut improved_model.resources {
            if *amount > 100.0 {
                *amount *= 0.95; // 减少5%
            }
        }

        // 3. 优化质量目标
        for (metric, target) in &mut improved_model.quality_metrics {
            if *target < 0.9 {
                *target = (*target + 0.9) / 2.0; // 提高质量目标
            }
        }

        improved_model
    }

    fn repair_constraints(&self, model: &LifecycleModel, violations: &[String]) -> LifecycleModel {
        let mut repaired_model = model.clone();

        for violation in violations {
            match violation.as_str() {
                "ResourceConstraint" => {
                    // 修复资源约束
                    self.repair_resource_constraints(&mut repaired_model);
                }
                "TimelineConstraint" => {
                    // 修复时间约束
                    self.repair_timeline_constraints(&mut repaired_model);
                }
                "QualityConstraint" => {
                    // 修复质量约束
                    self.repair_quality_constraints(&mut repaired_model);
                }
                _ => {
                    // 处理其他约束违反
                }
            }
        }

        repaired_model
    }

    fn repair_resource_constraints(&self, model: &mut LifecycleModel) {
        // 修复资源约束违反
        let total_resources: f64 = model.resources.values().sum();
        let max_resources = 1000.0; // 最大资源限制

        if total_resources > max_resources {
            let scale_factor = max_resources / total_resources;
            for amount in model.resources.values_mut() {
                *amount *= scale_factor;
            }
        }
    }

    fn repair_timeline_constraints(&self, model: &mut LifecycleModel) {
        // 修复时间约束违反
        let total_duration: f64 = model.phases.iter().map(|p| p.duration).sum();
        let max_duration = 365.0; // 最大项目持续时间（天）

        if total_duration > max_duration {
            let scale_factor = max_duration / total_duration;
            for phase in &mut model.phases {
                phase.duration *= scale_factor;
            }
        }
    }

    fn repair_quality_constraints(&self, model: &mut LifecycleModel) {
        // 修复质量约束违反
        for (metric, target) in &mut model.quality_metrics {
            if *target < 0.8 {
                *target = 0.8; // 设置最小质量目标
            }
        }
    }
}
```

## 2.1.6 国际标准对标

### PMBOK 7th Edition 标准

- **过程组**: 5个过程组（启动、规划、执行、监控、收尾）
- **知识领域**: 10个知识领域
- **绩效域**: 8个绩效域
- **价值交付系统**: 价值交付框架

### ISO 21500 标准

- **过程组**: 5个过程组
- **过程**: 39个项目管理过程
- **输入输出**: 标准化的输入输出定义
- **工具技术**: 推荐的工具和技术

### PRINCE2 标准

- **主题**: 7个主题
- **过程**: 7个过程
- **原则**: 7个原则
- **环境**: 项目环境适应

### APM Body of Knowledge 标准

- **知识领域**: 29个知识领域
- **能力框架**: 能力发展框架
- **最佳实践**: 行业最佳实践
- **专业发展**: 专业发展路径

## 2.1.7 引用关系

- 资源管理：参见 [2.2 资源管理模型](./resource-models.md)
- 风险管理：参见 [2.3 风险管理模型](./risk-models.md)
- 质量管理：参见 [2.4 质量管理模型](./quality-models.md)
- 基础理论：参见 [1.1 形式化基础理论](../01-foundations/README.md)
- 形式化验证：参见 [3.1 形式化验证理论](../03-formal-verification/verification-theory.md)

## 参考文献

1. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
2. ISO 21500:2012. Guidance on project management. International Organization for Standardization.
3. AXELOS. (2017). Managing Successful Projects with PRINCE2 2017 Edition. TSO (The Stationery Office).
4. Association for Project Management. (2019). APM Body of Knowledge 7th Edition. APM.
5. Kerzner, H. (2017). Project management: a systems approach to planning, scheduling, and controlling (12th ed.). John Wiley & Sons.
6. Meredith, J. R., & Mantel, S. J. (2019). Project management: a managerial approach (10th ed.). John Wiley & Sons.
7. Turner, J. R. (2016). Gower handbook of project management (5th ed.). Routledge.
8. Lock, D. (2013). Project management (10th ed.). Routledge.
9. Schwalbe, K. (2019). Information technology project management (9th ed.). Cengage Learning.
10. Wysocki, R. K. (2019). Effective project management: traditional, agile, extreme, hybrid (8th ed.). John Wiley & Sons.
