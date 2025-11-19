# 2.3 风险管理模型

## 概述

风险管理模型是Formal-ProgramManage的核心理论之一，定义了项目风险的识别、分析、应对和监控机制。本理论体系严格对标PMBOK 7th Edition、ISO 31000、PRINCE2等国际风险管理标准。

## 2.3.1 风险管理基础理论

### 基本定义

**定义 2.3.1** (项目风险 - PMBOK 7th Edition) 项目风险是一个五元组：
$$\mathcal{R} = (E, P, I, T, C)$$

其中：

- $E$ 是风险事件集合，满足 $E = \{e_1, e_2, \ldots, e_n\}$
- $P: E \rightarrow [0,1]$ 是概率函数，满足 $\sum_{e \in E} P(e) \leq 1$
- $I: E \rightarrow \mathbb{R}^+$ 是影响函数，表示风险影响程度
- $T: E \rightarrow \mathbb{R}^+$ 是时间函数，表示风险发生时间
- $C: E \rightarrow \mathbb{R}^+$ 是成本函数，表示风险应对成本

**定义 2.3.2** (风险暴露度) 风险暴露度是一个函数：
$$\text{Exposure}: E \rightarrow \mathbb{R}^+$$

定义为：
$$\text{Exposure}(e) = P(e) \times I(e)$$

**定义 2.3.3** (风险优先级) 风险优先级是一个函数：
$$\text{Priority}: E \rightarrow \mathbb{N}$$

定义为：
$$\text{Priority}(e) = \text{rank}(\text{Exposure}(e))$$

其中 $\text{rank}$ 是排序函数。

## 2.3.2 风险识别模型

### 风险分类体系

**定义 2.3.4** (风险分类) 风险分类是一个层次结构：
$$\mathcal{C} = \{C_1, C_2, \ldots, C_k\}$$

其中每个类别 $C_i$ 包含：

- **技术风险**: 技术实现、技术依赖、技术过时
- **管理风险**: 计划变更、资源不足、沟通问题
- **外部风险**: 市场变化、政策变化、自然灾害
- **财务风险**: 成本超支、资金不足、汇率波动
- **质量风险**: 质量缺陷、返工、客户满意度

### 风险识别算法

**算法 2.3.1** (风险识别算法)：

```rust
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Risk {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: RiskCategory,
    pub probability: f64,
    pub impact: f64,
    pub exposure: f64,
    pub priority: u32,
    pub triggers: Vec<String>,
    pub indicators: Vec<String>,
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

#[derive(Debug, Clone)]
pub enum RiskCategory {
    Technical,
    Management,
    External,
    Financial,
    Quality,
    Schedule,
    Resource,
    Communication,
}

#[derive(Debug, Clone)]
pub struct MitigationStrategy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub cost: f64,
    pub effectiveness: f64,
    pub implementation_time: f64,
}

#[derive(Debug)]
pub struct RiskIdentifier {
    pub risk_templates: HashMap<RiskCategory, Vec<RiskTemplate>>,
    pub historical_risks: Vec<Risk>,
    pub project_context: ProjectContext,
}

#[derive(Debug, Clone)]
pub struct RiskTemplate {
    pub name: String,
    pub description: String,
    pub typical_probability: f64,
    pub typical_impact: f64,
    pub triggers: Vec<String>,
    pub indicators: Vec<String>,
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

#[derive(Debug, Clone)]
pub struct ProjectContext {
    pub project_type: String,
    pub team_size: u32,
    pub budget: f64,
    pub duration: f64,
    pub complexity: f64,
    pub technology_stack: Vec<String>,
    pub stakeholders: Vec<String>,
}

impl RiskIdentifier {
    pub fn new() -> Self {
        RiskIdentifier {
            risk_templates: Self::initialize_risk_templates(),
            historical_risks: Vec::new(),
            project_context: ProjectContext {
                project_type: "software".to_string(),
                team_size: 10,
                budget: 1000000.0,
                duration: 365.0,
                complexity: 0.7,
                technology_stack: vec!["rust".to_string(), "haskell".to_string()],
                stakeholders: vec!["client".to_string(), "users".to_string()],
            },
        }
    }

    fn initialize_risk_templates() -> HashMap<RiskCategory, Vec<RiskTemplate>> {
        let mut templates = HashMap::new();

        // 技术风险模板
        templates.insert(RiskCategory::Technical, vec![
            RiskTemplate {
                name: "技术实现风险".to_string(),
                description: "新技术或复杂技术的实现困难".to_string(),
                typical_probability: 0.6,
                typical_impact: 0.8,
                triggers: vec!["新技术采用".to_string(), "技术复杂度高".to_string()],
                indicators: vec!["技术债务增加".to_string(), "开发速度下降".to_string()],
                mitigation_strategies: vec![
                    MitigationStrategy {
                        id: "tech_1".to_string(),
                        name: "技术预研".to_string(),
                        description: "提前进行技术可行性研究".to_string(),
                        cost: 50000.0,
                        effectiveness: 0.8,
                        implementation_time: 30.0,
                    }
                ],
            },
            RiskTemplate {
                name: "技术依赖风险".to_string(),
                description: "第三方技术或工具的依赖问题".to_string(),
                typical_probability: 0.4,
                typical_impact: 0.7,
                triggers: vec!["第三方组件更新".to_string(), "许可证变更".to_string()],
                indicators: vec!["依赖组件停止维护".to_string(), "安全漏洞发现".to_string()],
                mitigation_strategies: vec![
                    MitigationStrategy {
                        id: "tech_2".to_string(),
                        name: "依赖管理".to_string(),
                        description: "建立依赖管理和备份方案".to_string(),
                        cost: 20000.0,
                        effectiveness: 0.7,
                        implementation_time: 15.0,
                    }
                ],
            },
        ]);

        // 管理风险模板
        templates.insert(RiskCategory::Management, vec![
            RiskTemplate {
                name: "需求变更风险".to_string(),
                description: "项目需求频繁变更导致的范围蔓延".to_string(),
                typical_probability: 0.7,
                typical_impact: 0.9,
                triggers: vec!["客户需求不明确".to_string(), "市场变化".to_string()],
                indicators: vec!["需求文档频繁更新".to_string(), "开发计划调整".to_string()],
                mitigation_strategies: vec![
                    MitigationStrategy {
                        id: "mgmt_1".to_string(),
                        name: "需求管理".to_string(),
                        description: "建立需求变更控制流程".to_string(),
                        cost: 30000.0,
                        effectiveness: 0.8,
                        implementation_time: 20.0,
                    }
                ],
            },
        ]);

        templates
    }

    pub fn identify_risks(&self, project_context: &ProjectContext) -> Vec<Risk> {
        let mut identified_risks = Vec::new();

        // 基于模板识别风险
        for (category, templates) in &self.risk_templates {
            for template in templates {
                let risk = self.create_risk_from_template(template, project_context);
                identified_risks.push(risk);
            }
        }

        // 基于历史数据识别风险
        let historical_risks = self.identify_historical_risks(project_context);
        identified_risks.extend(historical_risks);

        // 基于项目特征识别风险
        let contextual_risks = self.identify_contextual_risks(project_context);
        identified_risks.extend(contextual_risks);

        // 计算风险暴露度和优先级
        for risk in &mut identified_risks {
            risk.exposure = risk.probability * risk.impact;
        }

        // 按暴露度排序并分配优先级
        identified_risks.sort_by(|a, b| b.exposure.partial_cmp(&a.exposure).unwrap());
        for (i, risk) in identified_risks.iter_mut().enumerate() {
            risk.priority = i as u32 + 1;
        }

        identified_risks
    }

    fn create_risk_from_template(&self, template: &RiskTemplate, context: &ProjectContext) -> Risk {
        // 根据项目上下文调整概率和影响
        let adjusted_probability = self.adjust_probability(template.typical_probability, context);
        let adjusted_impact = self.adjust_impact(template.typical_impact, context);

        Risk {
            id: format!("risk_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
            name: template.name.clone(),
            description: template.description.clone(),
            category: self.determine_category(&template.name),
            probability: adjusted_probability,
            impact: adjusted_impact,
            exposure: adjusted_probability * adjusted_impact,
            priority: 0, // 稍后设置
            triggers: template.triggers.clone(),
            indicators: template.indicators.clone(),
            mitigation_strategies: template.mitigation_strategies.clone(),
        }
    }

    fn adjust_probability(&self, base_probability: f64, context: &ProjectContext) -> f64 {
        let mut adjusted = base_probability;

        // 根据项目复杂度调整
        adjusted *= (1.0 + context.complexity * 0.3);

        // 根据团队规模调整
        if context.team_size > 20 {
            adjusted *= 1.2; // 大团队沟通风险增加
        }

        // 根据技术栈调整
        if context.technology_stack.len() > 3 {
            adjusted *= 1.1; // 多技术栈集成风险增加
        }

        adjusted.min(1.0)
    }

    fn adjust_impact(&self, base_impact: f64, context: &ProjectContext) -> f64 {
        let mut adjusted = base_impact;

        // 根据项目预算调整
        if context.budget > 5000000.0 {
            adjusted *= 1.2; // 大项目影响更大
        }

        // 根据项目持续时间调整
        if context.duration > 730.0 {
            adjusted *= 1.1; // 长期项目影响更大
        }

        adjusted.min(1.0)
    }

    fn determine_category(&self, risk_name: &str) -> RiskCategory {
        if risk_name.contains("技术") || risk_name.contains("技术") {
            RiskCategory::Technical
        } else if risk_name.contains("管理") || risk_name.contains("需求") {
            RiskCategory::Management
        } else if risk_name.contains("外部") || risk_name.contains("市场") {
            RiskCategory::External
        } else if risk_name.contains("财务") || risk_name.contains("成本") {
            RiskCategory::Financial
        } else if risk_name.contains("质量") || risk_name.contains("缺陷") {
            RiskCategory::Quality
        } else {
            RiskCategory::Management // 默认分类
        }
    }

    fn identify_historical_risks(&self, context: &ProjectContext) -> Vec<Risk> {
        let mut historical_risks = Vec::new();

        // 基于历史数据识别相似项目的风险
        for historical_risk in &self.historical_risks {
            if self.is_similar_project(historical_risk, context) {
                let adapted_risk = self.adapt_historical_risk(historical_risk, context);
                historical_risks.push(adapted_risk);
            }
        }

        historical_risks
    }

    fn is_similar_project(&self, risk: &Risk, context: &ProjectContext) -> bool {
        // 简化的相似性判断
        context.project_type == "software" &&
        context.team_size >= 5 &&
        context.team_size <= 50
    }

    fn adapt_historical_risk(&self, historical_risk: &Risk, context: &ProjectContext) -> Risk {
        let mut adapted = historical_risk.clone();
        adapted.id = format!("hist_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap());
        adapted.probability = self.adjust_probability(historical_risk.probability, context);
        adapted.impact = self.adjust_impact(historical_risk.impact, context);
        adapted
    }

    fn identify_contextual_risks(&self, context: &ProjectContext) -> Vec<Risk> {
        let mut contextual_risks = Vec::new();

        // 基于项目特征识别特定风险
        if context.team_size > 20 {
            contextual_risks.push(Risk {
                id: format!("context_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
                name: "大团队协调风险".to_string(),
                description: "大团队规模导致的沟通和协调困难".to_string(),
                category: RiskCategory::Communication,
                probability: 0.6,
                impact: 0.7,
                exposure: 0.42,
                priority: 0,
                triggers: vec!["团队规模增长".to_string(), "跨部门协作".to_string()],
                indicators: vec!["沟通效率下降".to_string(), "决策延迟".to_string()],
                mitigation_strategies: vec![
                    MitigationStrategy {
                        id: "comm_1".to_string(),
                        name: "沟通管理".to_string(),
                        description: "建立有效的沟通机制和工具".to_string(),
                        cost: 25000.0,
                        effectiveness: 0.8,
                        implementation_time: 10.0,
                    }
                ],
            });
        }

        if context.complexity > 0.8 {
            contextual_risks.push(Risk {
                id: format!("context_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
                name: "高复杂度项目风险".to_string(),
                description: "项目复杂度高导致的实施困难".to_string(),
                category: RiskCategory::Technical,
                probability: 0.8,
                impact: 0.9,
                exposure: 0.72,
                priority: 0,
                triggers: vec!["技术复杂度高".to_string(), "业务逻辑复杂".to_string()],
                indicators: vec!["开发进度延迟".to_string(), "缺陷率上升".to_string()],
                mitigation_strategies: vec![
                    MitigationStrategy {
                        id: "tech_3".to_string(),
                        name: "分阶段实施".to_string(),
                        description: "将复杂项目分解为可管理的阶段".to_string(),
                        cost: 40000.0,
                        effectiveness: 0.9,
                        implementation_time: 25.0,
                    }
                ],
            });
        }

        contextual_risks
    }
}
```

## 2.3.3 风险分析模型

### 定性风险分析

**定义 2.3.5** (风险矩阵) 风险矩阵是一个二维表：
$$M = [m_{ij}]_{n \times m}$$

其中：

- $m_{ij}$ 表示概率等级 $i$ 和影响等级 $j$ 的风险等级
- $n$ 是概率等级数
- $m$ 是影响等级数

**定义 2.3.6** (风险等级) 风险等级函数：
$$\text{RiskLevel}: [0,1] \times [0,1] \rightarrow \{Low, Medium, High, Critical\}$$

定义为：
$$\text{RiskLevel}(p, i) = \begin{cases}
\text{Low} & \text{if } p \times i < 0.1 \\
\text{Medium} & \text{if } 0.1 \leq p \times i < 0.3 \\
\text{High} & \text{if } 0.3 \leq p \times i < 0.6 \\
\text{Critical} & \text{if } p \times i \geq 0.6
\end{cases}$$

### 定量风险分析

**定义 2.3.7** (蒙特卡洛模拟) 蒙特卡洛模拟的风险评估：
$$\text{ExpectedLoss} = \frac{1}{N} \sum_{i=1}^{N} \text{Loss}_i$$

其中 $N$ 是模拟次数，$\text{Loss}_i$ 是第 $i$ 次模拟的损失。

**算法 2.3.2** (蒙特卡洛风险分析)：

```rust
use std::collections::HashMap;
use rand::Rng;

# [derive(Debug)]
pub struct MonteCarloAnalyzer {
    pub simulation_count: usize,
    pub risk_scenarios: Vec<RiskScenario>,
    pub project_parameters: ProjectParameters,
}

# [derive(Debug, Clone)]
pub struct RiskScenario {
    pub id: String,
    pub risks: Vec<Risk>,
    pub probability: f64,
    pub impact_distribution: ImpactDistribution,
}

# [derive(Debug, Clone)]
pub enum ImpactDistribution {
    Normal { mean: f64, std_dev: f64 },
    Uniform { min: f64, max: f64 },
    Triangular { min: f64, mode: f64, max: f64 },
    Exponential { lambda: f64 },
}

# [derive(Debug, Clone)]
pub struct ProjectParameters {
    pub base_cost: f64,
    pub base_duration: f64,
    pub base_quality: f64,
    pub cost_risk_factor: f64,
    pub duration_risk_factor: f64,
    pub quality_risk_factor: f64,
}

# [derive(Debug)]
pub struct SimulationResult {
    pub iterations: Vec<SimulationIteration>,
    pub summary: SimulationSummary,
}

# [derive(Debug, Clone)]
pub struct SimulationIteration {
    pub iteration: usize,
    pub cost: f64,
    pub duration: f64,
    pub quality: f64,
    pub triggered_risks: Vec<String>,
    pub total_impact: f64,
}

# [derive(Debug)]
pub struct SimulationSummary {
    pub mean_cost: f64,
    pub std_cost: f64,
    pub min_cost: f64,
    pub max_cost: f64,
    pub mean_duration: f64,
    pub std_duration: f64,
    pub min_duration: f64,
    pub max_duration: f64,
    pub mean_quality: f64,
    pub std_quality: f64,
    pub min_quality: f64,
    pub max_quality: f64,
    pub risk_probabilities: HashMap<String, f64>,
}

impl MonteCarloAnalyzer {
    pub fn new(simulation_count: usize, project_parameters: ProjectParameters) -> Self {
        MonteCarloAnalyzer {
            simulation_count,
            risk_scenarios: Vec::new(),
            project_parameters,
        }
    }

    pub fn add_risk_scenario(&mut self, scenario: RiskScenario) {
        self.risk_scenarios.push(scenario);
    }

    pub fn run_simulation(&self) -> SimulationResult {
        let mut iterations = Vec::new();
        let mut risk_trigger_counts: HashMap<String, u32> = HashMap::new();

        for i in 0..self.simulation_count {
            let iteration = self.run_single_iteration(i);

            // 统计风险触发次数
            for risk_id in &iteration.triggered_risks {
                *risk_trigger_counts.entry(risk_id.clone()).or_insert(0) += 1;
            }

            iterations.push(iteration);
        }

        // 计算统计摘要
        let summary = self.calculate_summary(&iterations, &risk_trigger_counts);

        SimulationResult {
            iterations,
            summary,
        }
    }

    fn run_single_iteration(&self, iteration: usize) -> SimulationIteration {
        let mut rng = rand::thread_rng();
        let mut triggered_risks = Vec::new();
        let mut total_impact = 0.0;

        let mut cost = self.project_parameters.base_cost;
        let mut duration = self.project_parameters.base_duration;
        let mut quality = self.project_parameters.base_quality;

        // 模拟每个风险场景
        for scenario in &self.risk_scenarios {
            // 检查风险是否触发
            if rng.gen::<f64>() < scenario.probability {
                triggered_risks.push(scenario.id.clone());

                // 计算风险影响
                let impact = self.calculate_impact(&scenario.impact_distribution, &mut rng);
                total_impact += impact;

                // 应用影响到项目参数
                cost += impact * self.project_parameters.cost_risk_factor;
                duration += impact * self.project_parameters.duration_risk_factor;
                quality -= impact * self.project_parameters.quality_risk_factor;
            }
        }

        // 确保质量在合理范围内
        quality = quality.max(0.0).min(1.0);

        SimulationIteration {
            iteration,
            cost,
            duration,
            quality,
            triggered_risks,
            total_impact,
        }
    }

    fn calculate_impact(&self, distribution: &ImpactDistribution, rng: &mut rand::rngs::ThreadRng) -> f64 {
        match distribution {
            ImpactDistribution::Normal { mean, std_dev } => {
                // 使用Box-Muller变换生成正态分布
                let u1 = rng.gen::<f64>();
                let u2 = rng.gen::<f64>();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + z0 * std_dev
            }
            ImpactDistribution::Uniform { min, max } => {
                rng.gen_range(*min..*max)
            }
            ImpactDistribution::Triangular { min, mode, max } => {
                let u = rng.gen::<f64>();
                let f = (*mode - *min) / (*max - *min);

                if u < f {
                    *min + (u * (*max - *min) * (*mode - *min)).sqrt()
                } else {
                    *max - ((1.0 - u) * (*max - *min) * (*max - *mode)).sqrt()
                }
            }
            ImpactDistribution::Exponential { lambda } => {
                -u.ln() / lambda
            }
        }
    }

    fn calculate_summary(&self, iterations: &[SimulationIteration], risk_trigger_counts: &HashMap<String, u32>) -> SimulationSummary {
        let costs: Vec<f64> = iterations.iter().map(|i| i.cost).collect();
        let durations: Vec<f64> = iterations.iter().map(|i| i.duration).collect();
        let qualities: Vec<f64> = iterations.iter().map(|i| i.quality).collect();

        let mean_cost = costs.iter().sum::<f64>() / costs.len() as f64;
        let mean_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let mean_quality = qualities.iter().sum::<f64>() / qualities.len() as f64;

        let std_cost = self.calculate_std(&costs, mean_cost);
        let std_duration = self.calculate_std(&durations, mean_duration);
        let std_quality = self.calculate_std(&qualities, mean_quality);

        let min_cost = costs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_cost = costs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_duration = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_duration = durations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_quality = qualities.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_quality = qualities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut risk_probabilities = HashMap::new();
        for (risk_id, count) in risk_trigger_counts {
            risk_probabilities.insert(risk_id.clone(), *count as f64 / self.simulation_count as f64);
        }

        SimulationSummary {
            mean_cost,
            std_cost,
            min_cost,
            max_cost,
            mean_duration,
            std_duration,
            min_duration,
            max_duration,
            mean_quality,
            std_quality,
            min_quality,
            max_quality,
            risk_probabilities,
        }
    }

    fn calculate_std(&self, values: &[f64], mean: f64) -> f64 {
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}
```

## 2.3.4 风险应对模型

### 风险应对策略

**定义 2.3.8** (风险应对策略) 风险应对策略是一个四元组：
$$\text{Strategy} = (T, A, C, E)$$

其中：
- $T$ 是策略类型：$\{Avoid, Transfer, Mitigate, Accept\}$
- $A$ 是行动集合
- $C$ 是成本函数
- $E$ 是效果函数

**定义 2.3.9** (风险应对优化) 风险应对优化问题：
$$\min_{\text{Strategy}} \sum_{i=1}^{n} C_i(\text{Strategy}_i)$$

约束条件：
$$\sum_{i=1}^{n} E_i(\text{Strategy}_i) \geq \text{TargetRiskReduction}$$

**算法 2.3.3** (风险应对优化算法)：

```rust
use std::collections::HashMap;

# [derive(Debug, Clone)]
pub struct RiskResponse {
    pub risk_id: String,
    pub strategy: ResponseStrategy,
    pub actions: Vec<ResponseAction>,
    pub cost: f64,
    pub effectiveness: f64,
    pub implementation_time: f64,
}

# [derive(Debug, Clone)]
pub enum ResponseStrategy {
    Avoid,
    Transfer,
    Mitigate,
    Accept,
}

# [derive(Debug, Clone)]
pub struct ResponseAction {
    pub id: String,
    pub name: String,
    pub description: String,
    pub cost: f64,
    pub effectiveness: f64,
    pub duration: f64,
    pub dependencies: Vec<String>,
}

# [derive(Debug)]
pub struct RiskResponseOptimizer {
    pub risks: Vec<Risk>,
    pub available_budget: f64,
    pub target_risk_reduction: f64,
    pub response_options: HashMap<String, Vec<RiskResponse>>,
}

impl RiskResponseOptimizer {
    pub fn new(risks: Vec<Risk>, available_budget: f64, target_risk_reduction: f64) -> Self {
        RiskResponseOptimizer {
            risks,
            available_budget,
            target_risk_reduction,
            response_options: HashMap::new(),
        }
    }

    pub fn add_response_option(&mut self, risk_id: String, response: RiskResponse) {
        self.response_options.entry(risk_id).or_insert_with(Vec::new).push(response);
    }

    pub fn optimize_responses(&self) -> Vec<RiskResponse> {
        // 使用动态规划优化风险应对策略
        let mut dp = vec![vec![0.0; (self.available_budget * 100.0) as usize + 1]; self.risks.len() + 1];
        let mut decisions = vec![vec![None; (self.available_budget * 100.0) as usize + 1]; self.risks.len()];

        // 初始化
        for j in 0..=(self.available_budget * 100.0) as usize {
            dp[0][j] = 0.0;
        }

        // 动态规划
        for i in 1..=self.risks.len() {
            let risk = &self.risks[i - 1];
            let responses = self.response_options.get(&risk.id).unwrap_or(&Vec::new());

            for budget in 0..=(self.available_budget * 100.0) as usize {
                dp[i][budget] = dp[i - 1][budget]; // 不采取任何措施
                decisions[i - 1][budget] = None;

                for response in responses {
                    let cost_budget = (response.cost * 100.0) as usize;
                    if cost_budget <= budget {
                        let risk_reduction = risk.exposure * response.effectiveness;
                        let total_value = dp[i - 1][budget - cost_budget] + risk_reduction;

                        if total_value > dp[i][budget] {
                            dp[i][budget] = total_value;
                            decisions[i - 1][budget] = Some(response.clone());
                        }
                    }
                }
            }
        }

        // 回溯最优解
        let mut selected_responses = Vec::new();
        let mut remaining_budget = (self.available_budget * 100.0) as usize;

        for i in (0..self.risks.len()).rev() {
            if let Some(response) = &decisions[i][remaining_budget] {
                selected_responses.push(response.clone());
                remaining_budget -= (response.cost * 100.0) as usize;
            }
        }

        selected_responses
    }

    pub fn calculate_risk_reduction(&self, responses: &[RiskResponse]) -> f64 {
        let mut total_reduction = 0.0;

        for response in responses {
            if let Some(risk) = self.risks.iter().find(|r| r.id == response.risk_id) {
                total_reduction += risk.exposure * response.effectiveness;
            }
        }

        total_reduction
    }

    pub fn calculate_total_cost(&self, responses: &[RiskResponse]) -> f64 {
        responses.iter().map(|r| r.cost).sum()
    }

    pub fn generate_response_plan(&self, responses: &[RiskResponse]) -> ResponsePlan {
        let mut plan = ResponsePlan {
            responses: responses.to_vec(),
            total_cost: self.calculate_total_cost(responses),
            total_risk_reduction: self.calculate_risk_reduction(responses),
            implementation_schedule: Vec::new(),
        };

        // 生成实施计划
        plan.implementation_schedule = self.generate_implementation_schedule(responses);

        plan
    }

    fn generate_implementation_schedule(&self, responses: &[RiskResponse]) -> Vec<ScheduledAction> {
        let mut scheduled_actions = Vec::new();
        let mut current_time = 0.0;

        // 按优先级排序响应
        let mut sorted_responses = responses.to_vec();
        sorted_responses.sort_by(|a, b| {
            let risk_a = self.risks.iter().find(|r| r.id == a.risk_id).unwrap();
            let risk_b = self.risks.iter().find(|r| r.id == b.risk_id).unwrap();
            risk_a.priority.cmp(&risk_b.priority)
        });

        for response in sorted_responses {
            for action in &response.actions {
                let scheduled_action = ScheduledAction {
                    action_id: action.id.clone(),
                    action_name: action.name.clone(),
                    start_time: current_time,
                    end_time: current_time + action.duration,
                    cost: action.cost,
                    risk_id: response.risk_id.clone(),
                };

                scheduled_actions.push(scheduled_action);
                current_time += action.duration;
            }
        }

        scheduled_actions
    }
}

# [derive(Debug)]
pub struct ResponsePlan {
    pub responses: Vec<RiskResponse>,
    pub total_cost: f64,
    pub total_risk_reduction: f64,
    pub implementation_schedule: Vec<ScheduledAction>,
}

# [derive(Debug, Clone)]
pub struct ScheduledAction {
    pub action_id: String,
    pub action_name: String,
    pub start_time: f64,
    pub end_time: f64,
    pub cost: f64,
    pub risk_id: String,
}
```

## 2.3.5 风险监控模型

### 风险监控系统

**定义 2.3.10** (风险监控指标) 风险监控指标包括：
- **风险触发率**: $\text{TriggerRate} = \frac{\text{TriggeredRisks}}{\text{TotalRisks}} \times 100\%$
- **风险应对效果**: $\text{Effectiveness} = \frac{\text{RiskReduction}}{\text{ResponseCost}}$
- **风险趋势**: $\text{Trend} = \frac{\text{CurrentRiskLevel} - \text{PreviousRiskLevel}}{\text{TimeInterval}}$

**算法 2.3.4** (风险监控算法)：

```rust
use std::collections::HashMap;

# [derive(Debug)]
pub struct RiskMonitor {
    pub risks: HashMap<String, Risk>,
    pub risk_indicators: HashMap<String, Vec<RiskIndicator>>,
    pub monitoring_thresholds: HashMap<String, f64>,
    pub alert_history: Vec<RiskAlert>,
}

# [derive(Debug, Clone)]
pub struct RiskIndicator {
    pub id: String,
    pub name: String,
    pub current_value: f64,
    pub threshold: f64,
    pub trend: f64,
    pub last_updated: f64,
}

# [derive(Debug, Clone)]
pub struct RiskAlert {
    pub id: String,
    pub risk_id: String,
    pub alert_type: AlertType,
    pub severity: Severity,
    pub message: String,
    pub timestamp: f64,
    pub value: f64,
    pub threshold: f64,
}

# [derive(Debug, Clone)]
pub enum AlertType {
    ThresholdExceeded,
    TrendWarning,
    RiskTriggered,
    ResponseIneffective,
}

# [derive(Debug, Clone)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl RiskMonitor {
    pub fn new() -> Self {
        RiskMonitor {
            risks: HashMap::new(),
            risk_indicators: HashMap::new(),
            monitoring_thresholds: HashMap::new(),
            alert_history: Vec::new(),
        }
    }

    pub fn add_risk(&mut self, risk: Risk) {
        self.risks.insert(risk.id.clone(), risk);
    }

    pub fn add_indicator(&mut self, risk_id: String, indicator: RiskIndicator) {
        self.risk_indicators.entry(risk_id).or_insert_with(Vec::new).push(indicator);
    }

    pub fn set_threshold(&mut self, risk_id: String, threshold: f64) {
        self.monitoring_thresholds.insert(risk_id, threshold);
    }

    pub fn monitor_risks(&mut self) -> Vec<RiskAlert> {
        let mut new_alerts = Vec::new();

        for (risk_id, risk) in &self.risks {
            // 检查风险指标
            if let Some(indicators) = self.risk_indicators.get(risk_id) {
                for indicator in indicators {
                    let alert = self.check_indicator(risk_id, indicator);
                    if let Some(alert) = alert {
                        new_alerts.push(alert.clone());
                        self.alert_history.push(alert);
                    }
                }
            }

            // 检查风险概率变化
            let probability_alert = self.check_probability_change(risk_id, risk);
            if let Some(alert) = probability_alert {
                        new_alerts.push(alert.clone());
                        self.alert_history.push(alert);
                    }

            // 检查风险影响变化
            let impact_alert = self.check_impact_change(risk_id, risk);
            if let Some(alert) = impact_alert {
                new_alerts.push(alert.clone());
                self.alert_history.push(alert);
            }
        }

        new_alerts
    }

    fn check_indicator(&self, risk_id: &str, indicator: &RiskIndicator) -> Option<RiskAlert> {
        if indicator.current_value > indicator.threshold {
            let severity = self.determine_severity(indicator.current_value, indicator.threshold);

            Some(RiskAlert {
                id: format!("alert_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
                risk_id: risk_id.to_string(),
                alert_type: AlertType::ThresholdExceeded,
                severity,
                message: format!("指标 '{}' 超过阈值: {:.2} > {:.2}",
                               indicator.name, indicator.current_value, indicator.threshold),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as f64,
                value: indicator.current_value,
                threshold: indicator.threshold,
            })
        } else if indicator.trend > 0.1 {
            // 趋势警告
            Some(RiskAlert {
                id: format!("alert_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
                risk_id: risk_id.to_string(),
                alert_type: AlertType::TrendWarning,
                severity: Severity::Medium,
                message: format!("指标 '{}' 呈上升趋势: {:.2}", indicator.name, indicator.trend),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as f64,
                value: indicator.trend,
                threshold: 0.1,
            })
        } else {
            None
        }
    }

    fn check_probability_change(&self, risk_id: &str, risk: &Risk) -> Option<RiskAlert> {
        // 这里应该比较当前概率与历史概率
        // 简化实现：检查概率是否超过某个阈值
        if risk.probability > 0.8 {
            Some(RiskAlert {
                id: format!("alert_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
                risk_id: risk_id.to_string(),
                alert_type: AlertType::RiskTriggered,
                severity: Severity::High,
                message: format!("风险 '{}' 概率过高: {:.2}", risk.name, risk.probability),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as f64,
                value: risk.probability,
                threshold: 0.8,
            })
        } else {
            None
        }
    }

    fn check_impact_change(&self, risk_id: &str, risk: &Risk) -> Option<RiskAlert> {
        // 检查风险影响是否显著增加
        if risk.impact > 0.9 {
            Some(RiskAlert {
                id: format!("alert_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
                risk_id: risk_id.to_string(),
                alert_type: AlertType::RiskTriggered,
                severity: Severity::Critical,
                message: format!("风险 '{}' 影响程度极高: {:.2}", risk.name, risk.impact),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as f64,
                value: risk.impact,
                threshold: 0.9,
            })
        } else {
            None
        }
    }

    fn determine_severity(&self, value: f64, threshold: f64) -> Severity {
        let ratio = value / threshold;

        if ratio > 2.0 {
            Severity::Critical
        } else if ratio > 1.5 {
            Severity::High
        } else if ratio > 1.2 {
            Severity::Medium
        } else {
            Severity::Low
        }
    }

    pub fn generate_risk_report(&self) -> RiskReport {
        let mut report = RiskReport {
            total_risks: self.risks.len(),
            high_priority_risks: 0,
            triggered_risks: 0,
            total_alerts: self.alert_history.len(),
            risk_distribution: HashMap::new(),
            alert_summary: HashMap::new(),
        };

        // 统计风险分布
        for risk in self.risks.values() {
            let category = format!("{:?}", risk.category);
            *report.risk_distribution.entry(category).or_insert(0) += 1;

            if risk.priority <= 5 {
                report.high_priority_risks += 1;
            }
        }

        // 统计告警摘要
        for alert in &self.alert_history {
            let alert_type = format!("{:?}", alert.alert_type);
            *report.alert_summary.entry(alert_type).or_insert(0) += 1;
        }

        report
    }
}

# [derive(Debug)]
pub struct RiskReport {
    pub total_risks: usize,
    pub high_priority_risks: usize,
    pub triggered_risks: usize,
    pub total_alerts: usize,
    pub risk_distribution: HashMap<String, usize>,
    pub alert_summary: HashMap<String, usize>,
}
```

## 2.3.6 国际标准对标

### PMBOK 7th Edition 标准

- **风险管理知识领域**: 项目风险管理过程
- **风险识别**: 识别风险过程
- **风险分析**: 实施定性风险分析、实施定量风险分析
- **风险应对**: 规划风险应对过程
- **风险监控**: 监督风险过程

### ISO 31000 标准

- **风险管理原则**: 风险管理框架和过程
- **风险评估**: 风险识别、风险分析、风险评价
- **风险处理**: 风险应对策略选择和实施
- **监控和评审**: 风险监控和持续改进

### PRINCE2 标准

- **风险主题**: 风险管理主题
- **风险识别**: 风险识别和评估
- **风险应对**: 风险应对策略
- **风险监控**: 风险监控和控制

## 2.3.7 引用关系

- 生命周期模型：参见 [2.1 项目生命周期模型](./lifecycle-models.md)
- 资源管理：参见 [2.2 资源管理模型](./resource-models.md)
- 质量管理：参见 [2.4 质量管理模型](./quality-models.md)
- 基础理论：参见 [1.1 形式化基础理论](../01-foundations/README.md)
- 形式化验证：参见 [3.1 形式化验证理论](../03-formal-verification/verification-theory.md)

## 参考文献

1. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
2. ISO 31000:2018. Risk management - Guidelines. International Organization for Standardization.
3. AXELOS. (2017). Managing Successful Projects with PRINCE2 2017 Edition. TSO (The Stationery Office).
4. Kerzner, H. (2017). Project management: a systems approach to planning, scheduling, and controlling (12th ed.). John Wiley & Sons.
5. Meredith, J. R., & Mantel, S. J. (2019). Project management: a managerial approach (10th ed.). John Wiley & Sons.
6. Turner, J. R. (2016). Gower handbook of project management (5th ed.). Routledge.
7. Lock, D. (2013). Project management (10th ed.). Routledge.
8. Schwalbe, K. (2019). Information technology project management (9th ed.). Cengage Learning.
9. Wysocki, R. K. (2019). Effective project management: traditional, agile, extreme, hybrid (8th ed.). John Wiley & Sons.
10. Chapman, C., & Ward, S. (2011). Project risk management: processes, techniques and insights. John Wiley & Sons.
