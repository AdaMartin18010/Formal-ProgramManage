# 2.3 风险管理模型

## 概述

风险管理模型是Formal-ProgramManage的核心组成部分，提供形式化的数学框架来识别、评估、监控和缓解项目中的各种风险。

## 2.3.1 风险基础定义

### 风险定义

**定义 2.3.1** 项目风险是一个五元组 $R = (E, P, I, T, C)$，其中：

- $E$ 是风险事件 (Event)
- $P$ 是发生概率 (Probability)
- $I$ 是影响程度 (Impact)
- $T$ 是时间窗口 (Time Window)
- $C$ 是风险类别 (Category)

### 风险分类

**定义 2.3.2** 风险类别集合 $\mathcal{C} = \{Technical, Schedule, Cost, Quality, Resource, External\}$

**定义 2.3.3** 技术风险 $R_{tech} = (E_{tech}, P_{tech}, I_{tech}, T_{tech}, Technical)$

**定义 2.3.4** 进度风险 $R_{schedule} = (E_{schedule}, P_{schedule}, I_{schedule}, T_{schedule}, Schedule)$

**定义 2.3.5** 成本风险 $R_{cost} = (E_{cost}, P_{cost}, I_{cost}, T_{cost}, Cost)$

**定义 2.3.6** 质量风险 $R_{quality} = (E_{quality}, P_{quality}, I_{quality}, T_{quality}, Quality)$

**定义 2.3.7** 资源风险 $R_{resource} = (E_{resource}, P_{resource}, I_{resource}, T_{resource}, Resource)$

**定义 2.3.8** 外部风险 $R_{external} = (E_{external}, P_{external}, I_{external}, T_{external}, External)$

## 2.3.2 风险评估理论

### 风险值计算

**定义 2.3.9** 风险值函数：
$$RV(r) = P(r) \times I(r)$$

其中 $r$ 是风险，$P(r)$ 是概率，$I(r)$ 是影响。

### 风险矩阵

**定义 2.3.10** 风险矩阵是一个函数 $RM: \mathcal{P} \times \mathcal{I} \rightarrow \mathcal{L}$，其中：

- $\mathcal{P}$ 是概率等级集合
- $\mathcal{I}$ 是影响等级集合
- $\mathcal{L}$ 是风险等级集合

### 风险等级定义

**定义 2.3.11** 风险等级集合 $\mathcal{L} = \{Low, Medium, High, Critical\}$

**定义 2.3.12** 风险等级映射：
$$RM(p, i) = \begin{cases}
Low & \text{if } p \times i < 0.1 \\
Medium & \text{if } 0.1 \leq p \times i < 0.3 \\
High & \text{if } 0.3 \leq p \times i < 0.7 \\
Critical & \text{if } p \times i \geq 0.7
\end{cases}$$

## 2.3.3 风险识别模型

### 风险识别函数

**定义 2.3.13** 风险识别函数 $RI: \mathcal{P} \rightarrow 2^{\mathcal{R}}$，其中：
- $\mathcal{P}$ 是项目集合
- $\mathcal{R}$ 是风险集合

### 风险识别算法

**算法 2.3.1** 基于检查表的风险识别：

```rust
use std::collections::{HashMap, HashSet};

# [derive(Debug, Clone)]
pub struct Risk {
    pub id: String,
    pub name: String,
    pub description: String,
    pub event: String,
    pub probability: f64,
    pub impact: f64,
    pub category: RiskCategory,
    pub time_window: TimeWindow,
    pub triggers: Vec<String>,
    pub mitigation_plan: Option<String>,
}

# [derive(Debug, Clone)]
pub enum RiskCategory {
    Technical,
    Schedule,
    Cost,
    Quality,
    Resource,
    External,
}

# [derive(Debug, Clone)]
pub struct TimeWindow {
    pub start_time: u64,
    pub end_time: u64,
    pub probability_distribution: ProbabilityDistribution,
}

# [derive(Debug, Clone)]
pub enum ProbabilityDistribution {
    Uniform,
    Normal { mean: f64, std_dev: f64 },
    Exponential { lambda: f64 },
    Custom(Vec<(f64, f64)>), // (time, probability)
}

pub struct RiskIdentifier {
    pub checklists: HashMap<RiskCategory, Vec<RiskCheckItem>>,
    pub historical_data: HashMap<String, Vec<HistoricalRisk>>,
    pub expert_knowledge: Vec<ExpertRule>,
}

# [derive(Debug, Clone)]
pub struct RiskCheckItem {
    pub id: String,
    pub question: String,
    pub category: RiskCategory,
    pub risk_template: RiskTemplate,
}

# [derive(Debug, Clone)]
pub struct RiskTemplate {
    pub name: String,
    pub description: String,
    pub base_probability: f64,
    pub base_impact: f64,
    pub category: RiskCategory,
}

# [derive(Debug, Clone)]
pub struct HistoricalRisk {
    pub project_id: String,
    pub risk_id: String,
    pub occurred: bool,
    pub actual_impact: f64,
    pub timestamp: u64,
}

# [derive(Debug, Clone)]
pub struct ExpertRule {
    pub id: String,
    pub condition: RiskCondition,
    pub risk_template: RiskTemplate,
    pub confidence: f64,
}

# [derive(Debug, Clone)]
pub enum RiskCondition {
    ProjectComplexity(f64),
    TeamExperience(u32),
    TechnologyMaturity(f64),
    BudgetConstraint(f64),
    TimeConstraint(f64),
    ResourceAvailability(f64),
}

impl RiskIdentifier {
    pub fn new() -> Self {
        RiskIdentifier {
            checklists: Self::initialize_checklists(),
            historical_data: HashMap::new(),
            expert_knowledge: Self::initialize_expert_rules(),
        }
    }

    fn initialize_checklists() -> HashMap<RiskCategory, Vec<RiskCheckItem>> {
        let mut checklists = HashMap::new();

        // 技术风险检查表
        checklists.insert(RiskCategory::Technical, vec![
            RiskCheckItem {
                id: "TECH-001".to_string(),
                question: "项目是否使用了新技术？".to_string(),
                category: RiskCategory::Technical,
                risk_template: RiskTemplate {
                    name: "新技术风险".to_string(),
                    description: "使用新技术可能导致技术风险".to_string(),
                    base_probability: 0.6,
                    base_impact: 0.7,
                    category: RiskCategory::Technical,
                },
            },
            RiskCheckItem {
                id: "TECH-002".to_string(),
                question: "团队是否有相关技术经验？".to_string(),
                category: RiskCategory::Technical,
                risk_template: RiskTemplate {
                    name: "技术经验风险".to_string(),
                    description: "团队缺乏相关技术经验".to_string(),
                    base_probability: 0.4,
                    base_impact: 0.8,
                    category: RiskCategory::Technical,
                },
            },
        ]);

        // 进度风险检查表
        checklists.insert(RiskCategory::Schedule, vec![
            RiskCheckItem {
                id: "SCHED-001".to_string(),
                question: "项目时间是否紧张？".to_string(),
                category: RiskCategory::Schedule,
                risk_template: RiskTemplate {
                    name: "时间压力风险".to_string(),
                    description: "项目时间紧张可能导致进度风险".to_string(),
                    base_probability: 0.5,
                    base_impact: 0.6,
                    category: RiskCategory::Schedule,
                },
            },
        ]);

        // 成本风险检查表
        checklists.insert(RiskCategory::Cost, vec![
            RiskCheckItem {
                id: "COST-001".to_string(),
                question: "预算是否充足？".to_string(),
                category: RiskCategory::Cost,
                risk_template: RiskTemplate {
                    name: "预算不足风险".to_string(),
                    description: "预算不足可能导致成本超支".to_string(),
                    base_probability: 0.3,
                    base_impact: 0.9,
                    category: RiskCategory::Cost,
                },
            },
        ]);

        checklists
    }

    fn initialize_expert_rules() -> Vec<ExpertRule> {
        vec![
            ExpertRule {
                id: "EXP-001".to_string(),
                condition: RiskCondition::ProjectComplexity(0.8),
                risk_template: RiskTemplate {
                    name: "高复杂度风险".to_string(),
                    description: "项目复杂度高增加风险".to_string(),
                    base_probability: 0.7,
                    base_impact: 0.8,
                    category: RiskCategory::Technical,
                },
                confidence: 0.9,
            },
            ExpertRule {
                id: "EXP-002".to_string(),
                condition: RiskCondition::TeamExperience(2),
                risk_template: RiskTemplate {
                    name: "团队经验不足风险".to_string(),
                    description: "团队经验不足增加风险".to_string(),
                    base_probability: 0.6,
                    base_impact: 0.7,
                    category: RiskCategory::Resource,
                },
                confidence: 0.8,
            },
        ]
    }

    pub fn identify_risks(&self, project: &Project) -> Vec<Risk> {
        let mut risks = Vec::new();

        // 基于检查表识别风险
        risks.extend(self.checklist_based_identification(project));

        // 基于历史数据识别风险
        risks.extend(self.historical_based_identification(project));

        // 基于专家规则识别风险
        risks.extend(self.expert_based_identification(project));

        // 去重和合并
        self.deduplicate_risks(risks)
    }

    fn checklist_based_identification(&self, project: &Project) -> Vec<Risk> {
        let mut risks = Vec::new();

        for (category, checklist) in &self.checklists {
            for item in checklist {
                if self.evaluate_check_item(project, item) {
                    let risk = self.create_risk_from_template(&item.risk_template, project);
                    risks.push(risk);
                }
            }
        }

        risks
    }

    fn evaluate_check_item(&self, project: &Project, item: &RiskCheckItem) -> bool {
        // 简化实现：根据问题类型评估
        match item.id.as_str() {
            "TECH-001" => {
                // 检查是否使用新技术
                project.uses_new_technology()
            },
            "TECH-002" => {
                // 检查团队技术经验
                !project.team_has_experience()
            },
            "SCHED-001" => {
                // 检查时间压力
                project.has_time_pressure()
            },
            "COST-001" => {
                // 检查预算充足性
                !project.has_sufficient_budget()
            },
            _ => false,
        }
    }

    fn historical_based_identification(&self, project: &Project) -> Vec<Risk> {
        let mut risks = Vec::new();

        // 分析历史数据中的风险模式
        for (project_type, historical_risks) in &self.historical_data {
            if self.is_similar_project(project, project_type) {
                let common_risks = self.find_common_risks(historical_risks);
                for risk_pattern in common_risks {
                    let risk = self.create_risk_from_pattern(risk_pattern, project);
                    risks.push(risk);
                }
            }
        }

        risks
    }

    fn expert_based_identification(&self, project: &Project) -> Vec<Risk> {
        let mut risks = Vec::new();

        for rule in &self.expert_knowledge {
            if self.evaluate_expert_rule(project, rule) {
                let risk = self.create_risk_from_template(&rule.risk_template, project);
                risks.push(risk);
            }
        }

        risks
    }

    fn evaluate_expert_rule(&self, project: &Project, rule: &ExpertRule) -> bool {
        match &rule.condition {
            RiskCondition::ProjectComplexity(threshold) => {
                project.complexity() > *threshold
            },
            RiskCondition::TeamExperience(min_experience) => {
                project.team_experience() < *min_experience
            },
            RiskCondition::TechnologyMaturity(threshold) => {
                project.technology_maturity() < *threshold
            },
            RiskCondition::BudgetConstraint(threshold) => {
                project.budget_utilization() > *threshold
            },
            RiskCondition::TimeConstraint(threshold) => {
                project.time_utilization() > *threshold
            },
            RiskCondition::ResourceAvailability(threshold) => {
                project.resource_availability() < *threshold
            },
        }
    }

    fn create_risk_from_template(&self, template: &RiskTemplate, project: &Project) -> Risk {
        Risk {
            id: format!("{}-{}", template.name, project.id),
            name: template.name.clone(),
            description: template.description.clone(),
            event: format!("{} in project {}", template.name, project.id),
            probability: template.base_probability,
            impact: template.base_impact,
            category: template.category.clone(),
            time_window: TimeWindow {
                start_time: 0,
                end_time: project.duration(),
                probability_distribution: ProbabilityDistribution::Uniform,
            },
            triggers: Vec::new(),
            mitigation_plan: None,
        }
    }

    fn deduplicate_risks(&self, risks: Vec<Risk>) -> Vec<Risk> {
        let mut unique_risks = Vec::new();
        let mut seen_names = HashSet::new();

        for risk in risks {
            if !seen_names.contains(&risk.name) {
                seen_names.insert(risk.name.clone());
                unique_risks.push(risk);
            }
        }

        unique_risks
    }
}

// 项目扩展方法（简化实现）
impl Project {
    fn uses_new_technology(&self) -> bool {
        // 简化实现
        true
    }

    fn team_has_experience(&self) -> bool {
        // 简化实现
        false
    }

    fn has_time_pressure(&self) -> bool {
        // 简化实现
        true
    }

    fn has_sufficient_budget(&self) -> bool {
        // 简化实现
        false
    }

    fn complexity(&self) -> f64 {
        // 简化实现
        0.8
    }

    fn team_experience(&self) -> u32 {
        // 简化实现
        1
    }

    fn technology_maturity(&self) -> f64 {
        // 简化实现
        0.6
    }

    fn budget_utilization(&self) -> f64 {
        // 简化实现
        0.9
    }

    fn time_utilization(&self) -> f64 {
        // 简化实现
        0.8
    }

    fn resource_availability(&self) -> f64 {
        // 简化实现
        0.7
    }

    fn duration(&self) -> u64 {
        // 简化实现
        100
    }
}
```

## 2.3.4 风险量化模型

### 蒙特卡洛模拟

**定义 2.3.14** 风险蒙特卡洛模拟：
$$E[Loss] = \frac{1}{N} \sum_{i=1}^{N} \sum_{r \in \mathcal{R}} I_r^{(i)} \times P_r^{(i)}$$

其中：
- $N$ 是模拟次数
- $I_r^{(i)}$ 是第 $i$ 次模拟中风险 $r$ 的影响
- $P_r^{(i)}$ 是第 $i$ 次模拟中风险 $r$ 的发生概率

### 风险量化实现

```rust
use rand::Rng;

pub struct RiskQuantifier {
    pub simulation_iterations: usize,
    pub risk_models: HashMap<String, RiskModel>,
}

# [derive(Debug, Clone)]
pub struct RiskModel {
    pub risk_id: String,
    pub probability_distribution: ProbabilityDistribution,
    pub impact_distribution: ImpactDistribution,
    pub correlation_matrix: Option<Vec<Vec<f64>>>,
}

# [derive(Debug, Clone)]
pub enum ImpactDistribution {
    Fixed(f64),
    Normal { mean: f64, std_dev: f64 },
    Uniform { min: f64, max: f64 },
    Triangular { min: f64, mode: f64, max: f64 },
}

impl RiskQuantifier {
    pub fn new(simulation_iterations: usize) -> Self {
        RiskQuantifier {
            simulation_iterations,
            risk_models: HashMap::new(),
        }
    }

    pub fn add_risk_model(&mut self, risk_id: String, model: RiskModel) {
        self.risk_models.insert(risk_id, model);
    }

    pub fn run_monte_carlo_simulation(&self) -> MonteCarloResult {
        let mut rng = rand::thread_rng();
        let mut total_losses = Vec::new();
        let mut risk_occurrences = HashMap::new();

        for iteration in 0..self.simulation_iterations {
            let mut iteration_loss = 0.0;
            let mut iteration_risks = Vec::new();

            // 模拟每个风险
            for (risk_id, model) in &self.risk_models {
                let probability = self.sample_probability(&model.probability_distribution, &mut rng);
                let impact = self.sample_impact(&model.impact_distribution, &mut rng);

                // 检查风险是否发生
                if rng.gen::<f64>() < probability {
                    iteration_loss += impact;
                    iteration_risks.push(risk_id.clone());
                }
            }

            total_losses.push(iteration_loss);

            // 记录风险发生次数
            for risk_id in iteration_risks {
                *risk_occurrences.entry(risk_id).or_insert(0) += 1;
            }
        }

        // 计算统计指标
        let mean_loss = total_losses.iter().sum::<f64>() / total_losses.len() as f64;
        let variance = total_losses.iter()
            .map(|&x| (x - mean_loss).powi(2))
            .sum::<f64>() / total_losses.len() as f64;
        let std_dev = variance.sqrt();

        // 计算分位数
        total_losses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_50 = total_losses[total_losses.len() / 2];
        let percentile_90 = total_losses[(total_losses.len() * 9) / 10];
        let percentile_95 = total_losses[(total_losses.len() * 19) / 20];

        MonteCarloResult {
            mean_loss,
            std_dev,
            percentile_50,
            percentile_90,
            percentile_95,
            risk_occurrences,
            total_iterations: self.simulation_iterations,
        }
    }

    fn sample_probability(&self, distribution: &ProbabilityDistribution, rng: &mut impl Rng) -> f64 {
        match distribution {
            ProbabilityDistribution::Uniform => rng.gen::<f64>(),
            ProbabilityDistribution::Normal { mean, std_dev } => {
                // 使用Box-Muller变换生成正态分布
                let u1 = rng.gen::<f64>();
                let u2 = rng.gen::<f64>();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + std_dev * z0
            },
            ProbabilityDistribution::Exponential { lambda } => {
                -rng.gen::<f64>().ln() / lambda
            },
            ProbabilityDistribution::Custom(probabilities) => {
                // 从自定义概率分布中采样
                let random = rng.gen::<f64>();
                let mut cumulative = 0.0;

                for (time, prob) in probabilities {
                    cumulative += prob;
                    if random <= cumulative {
                        return *time;
                    }
                }

                probabilities.last().map(|(_, prob)| *prob).unwrap_or(0.0)
            },
        }
    }

    fn sample_impact(&self, distribution: &ImpactDistribution, rng: &mut impl Rng) -> f64 {
        match distribution {
            ImpactDistribution::Fixed(value) => *value,
            ImpactDistribution::Normal { mean, std_dev } => {
                let u1 = rng.gen::<f64>();
                let u2 = rng.gen::<f64>();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + std_dev * z0
            },
            ImpactDistribution::Uniform { min, max } => {
                min + (max - min) * rng.gen::<f64>()
            },
            ImpactDistribution::Triangular { min, mode, max } => {
                let u = rng.gen::<f64>();
                if u < (mode - min) / (max - min) {
                    min + (u * (max - min) * (mode - min)).sqrt()
                } else {
                    max - ((1.0 - u) * (max - min) * (max - mode)).sqrt()
                }
            },
        }
    }
}

# [derive(Debug, Clone)]
pub struct MonteCarloResult {
    pub mean_loss: f64,
    pub std_dev: f64,
    pub percentile_50: f64,
    pub percentile_90: f64,
    pub percentile_95: f64,
    pub risk_occurrences: HashMap<String, usize>,
    pub total_iterations: usize,
}
```

## 2.3.5 风险缓解模型

### 缓解策略定义

**定义 2.3.15** 风险缓解策略是一个四元组 $MS = (R, S, C, E)$，其中：
- $R$ 是目标风险
- $S$ 是缓解策略
- $C$ 是实施成本
- $E$ 是预期效果

### 缓解策略类型

**定义 2.3.16** 缓解策略类型：
- **避免** (Avoid): 完全消除风险
- **转移** (Transfer): 将风险转移给第三方
- **缓解** (Mitigate): 降低风险概率或影响
- **接受** (Accept): 接受风险并准备应对

### 缓解策略优化

**定义 2.3.17** 缓解策略优化问题：
$$\min_{MS} \sum_{ms \in MS} C(ms)$$

约束条件：
$$\sum_{ms \in MS} E(ms) \geq T$$

其中 $T$ 是目标风险降低阈值。

### 缓解策略实现

```rust
pub struct RiskMitigation {
    pub strategies: HashMap<String, MitigationStrategy>,
    pub effectiveness_matrix: HashMap<String, HashMap<String, f64>>,
    pub cost_matrix: HashMap<String, f64>,
}

# [derive(Debug, Clone)]
pub struct MitigationStrategy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub strategy_type: MitigationType,
    pub target_risks: Vec<String>,
    pub cost: f64,
    pub effectiveness: f64,
    pub implementation_time: u64,
    pub dependencies: Vec<String>,
}

# [derive(Debug, Clone)]
pub enum MitigationType {
    Avoid,
    Transfer,
    Mitigate,
    Accept,
}

impl RiskMitigation {
    pub fn new() -> Self {
        RiskMitigation {
            strategies: HashMap::new(),
            effectiveness_matrix: HashMap::new(),
            cost_matrix: HashMap::new(),
        }
    }

    pub fn add_strategy(&mut self, strategy: MitigationStrategy) {
        self.strategies.insert(strategy.id.clone(), strategy);
    }

    pub fn optimize_mitigation_plan(&self, risks: &[Risk], budget: f64) -> MitigationPlan {
        let mut plan = MitigationPlan::new();
        let mut remaining_budget = budget;
        let mut covered_risks = HashSet::new();

        // 按成本效益比排序策略
        let mut strategies: Vec<&MitigationStrategy> = self.strategies.values().collect();
        strategies.sort_by(|a, b| {
            let ratio_a = a.effectiveness / a.cost;
            let ratio_b = b.effectiveness / b.cost;
            ratio_b.partial_cmp(&ratio_a).unwrap()
        });

        for strategy in strategies {
            if strategy.cost <= remaining_budget {
                // 检查策略是否覆盖未处理的风险
                let uncovered_risks = strategy.target_risks.iter()
                    .filter(|risk_id| !covered_risks.contains(*risk_id))
                    .count();

                if uncovered_risks > 0 {
                    plan.add_strategy(strategy.clone());
                    remaining_budget -= strategy.cost;

                    for risk_id in &strategy.target_risks {
                        covered_risks.insert(risk_id.clone());
                    }
                }
            }
        }

        plan
    }

    pub fn calculate_mitigation_effectiveness(&self, plan: &MitigationPlan, risks: &[Risk]) -> f64 {
        let mut total_risk_reduction = 0.0;
        let mut total_original_risk = 0.0;

        for risk in risks {
            let original_risk_value = risk.probability * risk.impact;
            total_original_risk += original_risk_value;

            let risk_reduction = self.calculate_risk_reduction(risk, plan);
            total_risk_reduction += risk_reduction;
        }

        if total_original_risk > 0.0 {
            total_risk_reduction / total_original_risk
        } else {
            0.0
        }
    }

    fn calculate_risk_reduction(&self, risk: &Risk, plan: &MitigationPlan) -> f64 {
        let mut risk_reduction = 0.0;

        for strategy in &plan.strategies {
            if strategy.target_risks.contains(&risk.id) {
                match strategy.strategy_type {
                    MitigationType::Avoid => {
                        risk_reduction = risk.probability * risk.impact;
                    },
                    MitigationType::Transfer => {
                        risk_reduction = risk.probability * risk.impact * 0.8;
                    },
                    MitigationType::Mitigate => {
                        risk_reduction = risk.probability * risk.impact * strategy.effectiveness;
                    },
                    MitigationType::Accept => {
                        risk_reduction = 0.0;
                    },
                }
            }
        }

        risk_reduction
    }
}

# [derive(Debug, Clone)]
pub struct MitigationPlan {
    pub strategies: Vec<MitigationStrategy>,
    pub total_cost: f64,
    pub expected_effectiveness: f64,
}

impl MitigationPlan {
    pub fn new() -> Self {
        MitigationPlan {
            strategies: Vec::new(),
            total_cost: 0.0,
            expected_effectiveness: 0.0,
        }
    }

    pub fn add_strategy(&mut self, strategy: MitigationStrategy) {
        self.total_cost += strategy.cost;
        self.strategies.push(strategy);
    }

    pub fn calculate_roi(&self, original_risk_value: f64) -> f64 {
        if self.total_cost > 0.0 {
            (original_risk_value - self.expected_effectiveness) / self.total_cost
        } else {
            0.0
        }
    }
}
```

## 2.3.6 风险监控模型

### 风险监控指标

**定义 2.3.18** 风险监控指标：
- **风险暴露度** (Risk Exposure): $RE = \sum_{r \in \mathcal{R}} P(r) \times I(r)$
- **风险趋势** (Risk Trend): $RT = \frac{d}{dt} RE(t)$
- **风险集中度** (Risk Concentration): $RC = \max_{r \in \mathcal{R}} P(r) \times I(r)$

### 风险监控实现

```rust
pub struct RiskMonitor {
    pub risk_metrics: HashMap<String, RiskMetrics>,
    pub alert_thresholds: HashMap<String, f64>,
    pub monitoring_history: Vec<MonitoringRecord>,
}

# [derive(Debug, Clone)]
pub struct RiskMetrics {
    pub risk_id: String,
    pub current_probability: f64,
    pub current_impact: f64,
    pub risk_value: f64,
    pub trend: f64,
    pub last_updated: u64,
}

# [derive(Debug, Clone)]
pub struct MonitoringRecord {
    pub timestamp: u64,
    pub risk_id: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub alert_triggered: bool,
}

impl RiskMonitor {
    pub fn new() -> Self {
        RiskMonitor {
            risk_metrics: HashMap::new(),
            alert_thresholds: HashMap::new(),
            monitoring_history: Vec::new(),
        }
    }

    pub fn update_risk_metrics(&mut self, risk_id: &str, probability: f64, impact: f64) {
        let risk_value = probability * impact;
        let trend = self.calculate_trend(risk_id, risk_value);

        let metrics = RiskMetrics {
            risk_id: risk_id.to_string(),
            current_probability: probability,
            current_impact: impact,
            risk_value,
            trend,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.risk_metrics.insert(risk_id.to_string(), metrics);

        // 检查告警
        self.check_alerts(risk_id, risk_value);
    }

    fn calculate_trend(&self, risk_id: &str, current_value: f64) -> f64 {
        // 计算风险趋势（简化实现）
        let history = self.monitoring_history.iter()
            .filter(|record| record.risk_id == risk_id)
            .collect::<Vec<_>>();

        if history.len() >= 2 {
            let recent = history[history.len() - 1].metric_value;
            let previous = history[history.len() - 2].metric_value;
            recent - previous
        } else {
            0.0
        }
    }

    fn check_alerts(&mut self, risk_id: &str, risk_value: f64) {
        if let Some(threshold) = self.alert_thresholds.get(risk_id) {
            if risk_value > *threshold {
                let record = MonitoringRecord {
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    risk_id: risk_id.to_string(),
                    metric_value: risk_value,
                    threshold: *threshold,
                    alert_triggered: true,
                };

                self.monitoring_history.push(record);
            }
        }
    }

    pub fn get_risk_report(&self, risk_id: &str) -> Option<RiskReport> {
        if let Some(metrics) = self.risk_metrics.get(risk_id) {
            let alerts = self.monitoring_history.iter()
                .filter(|record| record.risk_id == risk_id && record.alert_triggered)
                .cloned()
                .collect();

            Some(RiskReport {
                risk_id: risk_id.to_string(),
                metrics: metrics.clone(),
                alerts,
            })
        } else {
            None
        }
    }
}

# [derive(Debug, Clone)]
pub struct RiskReport {
    pub risk_id: String,
    pub metrics: RiskMetrics,
    pub alerts: Vec<MonitoringRecord>,
}
```

## 2.3.7 相关链接

- [1.1 形式化基础理论](../01-foundations/README.md)
- [1.2 数学模型基础](../01-foundations/mathematical-models.md)
- [2.1 项目生命周期模型](./lifecycle-models.md)
- [2.2 资源管理模型](./resource-models.md)
- [2.4 质量管理模型](./quality-models.md)

## 参考文献

1. PMI. (2017). A Guide to the Project Management Body of Knowledge (PMBOK Guide). Project Management Institute.
2. Chapman, C., & Ward, S. (2011). Project risk management: processes, techniques and insights. John Wiley & Sons.
3. Hillson, D. (2009). Managing risk in projects. Gower Publishing, Ltd.
4. Kerzner, H. (2017). Project management: a systems approach to planning, scheduling, and controlling. John Wiley & Sons.
