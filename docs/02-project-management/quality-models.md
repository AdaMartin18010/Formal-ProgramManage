# 2.4 质量管理模型

## 概述

质量管理模型是Formal-ProgramManage的核心理论之一，定义了项目质量的规划、保证、控制和改进机制。本理论体系严格对标ISO/IEC 25010、ISO 9001、CMMI-DEV等国际质量管理标准。

## 2.4.1 质量管理基础理论

### 基本定义

**定义 2.4.1** (项目质量 - ISO/IEC 25010) 项目质量是一个六元组：
$$\mathcal{Q} = (F, E, M, P, S, U)$$

其中：
- $F$ 是功能性质量属性，满足 $F: \mathcal{F} \rightarrow [0,1]$
- $E$ 是效率性质量属性，满足 $E: \mathcal{E} \rightarrow [0,1]$
- $M$ 是维护性质量属性，满足 $M: \mathcal{M} \rightarrow [0,1]$
- $P$ 是可移植性质量属性，满足 $P: \mathcal{P} \rightarrow [0,1]$
- $S$ 是安全性质量属性，满足 $S: \mathcal{S} \rightarrow [0,1]$
- $U$ 是可用性质量属性，满足 $U: \mathcal{U} \rightarrow [0,1]$

**定义 2.4.2** (质量函数) 质量函数是一个映射：
$$\text{Quality}: \mathcal{Q} \rightarrow [0,1]$$

定义为：
$$\text{Quality}(q) = \alpha \cdot F + \beta \cdot E + \gamma \cdot M + \delta \cdot P + \epsilon \cdot S + \zeta \cdot U$$

其中 $\alpha + \beta + \gamma + \delta + \epsilon + \zeta = 1$ 是权重系数。

**定义 2.4.3** (质量约束) 质量约束是一个三元组：
$$C = (Q, L, U)$$

其中：
- $Q$ 是质量属性
- $L$ 是下界约束，满足 $L \in [0,1]$
- $U$ 是上界约束，满足 $U \in [0,1]$ 且 $U \geq L$

## 2.4.2 质量规划模型

### 质量目标设定

**定义 2.4.4** (质量目标) 质量目标是一个函数：
$$\text{QualityGoal}: \mathcal{P} \times \mathcal{T} \rightarrow [0,1]$$

其中 $\mathcal{P}$ 是项目集合，$\mathcal{T}$ 是时间集合。

**定义 2.4.5** (质量基准) 质量基准是一个四元组：
$$B = (M, T, V, C)$$

其中：
- $M$ 是度量指标集合
- $T$ 是目标值集合
- $V$ 是验证方法集合
- $C$ 是控制机制集合

### 质量规划算法

**算法 2.4.1** (质量规划算法)：

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct QualityAttribute {
    pub name: String,
    pub value: f64,
    pub weight: f64,
    pub target: f64,
    pub tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct QualityMetric {
    pub id: String,
    pub name: String,
    pub description: String,
    pub measurement_method: String,
    pub unit: String,
    pub target_value: f64,
    pub acceptable_range: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct QualityPlan {
    pub project_id: String,
    pub quality_attributes: Vec<QualityAttribute>,
    pub quality_metrics: Vec<QualityMetric>,
    pub quality_goals: HashMap<String, f64>,
    pub quality_controls: Vec<QualityControl>,
    pub quality_improvements: Vec<QualityImprovement>,
}

#[derive(Debug, Clone)]
pub struct QualityControl {
    pub id: String,
    pub name: String,
    pub description: String,
    pub control_type: ControlType,
    pub frequency: String,
    pub responsible: String,
    pub tools: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ControlType {
    Preventive,
    Detective,
    Corrective,
}

#[derive(Debug, Clone)]
pub struct QualityImprovement {
    pub id: String,
    pub name: String,
    pub description: String,
    pub improvement_type: ImprovementType,
    pub priority: u32,
    pub cost: f64,
    pub expected_benefit: f64,
}

#[derive(Debug, Clone)]
pub enum ImprovementType {
    Process,
    Technology,
    Training,
    Tool,
}

#[derive(Debug)]
pub struct QualityPlanner {
    pub quality_standards: HashMap<String, QualityStandard>,
    pub quality_templates: HashMap<String, QualityTemplate>,
    pub historical_data: Vec<QualityData>,
}

#[derive(Debug, Clone)]
pub struct QualityStandard {
    pub name: String,
    pub version: String,
    pub description: String,
    pub requirements: Vec<QualityRequirement>,
    pub metrics: Vec<QualityMetric>,
}

#[derive(Debug, Clone)]
pub struct QualityRequirement {
    pub id: String,
    pub description: String,
    pub category: String,
    pub priority: u32,
    pub acceptance_criteria: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct QualityTemplate {
    pub name: String,
    pub project_type: String,
    pub quality_attributes: Vec<QualityAttribute>,
    pub quality_metrics: Vec<QualityMetric>,
    pub quality_controls: Vec<QualityControl>,
}

#[derive(Debug, Clone)]
pub struct QualityData {
    pub project_id: String,
    pub timestamp: f64,
    pub quality_score: f64,
    pub quality_attributes: HashMap<String, f64>,
    pub issues: Vec<QualityIssue>,
}

#[derive(Debug, Clone)]
pub struct QualityIssue {
    pub id: String,
    pub description: String,
    pub severity: Severity,
    pub category: String,
    pub status: IssueStatus,
}

#[derive(Debug, Clone)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub enum IssueStatus {
    Open,
    InProgress,
    Resolved,
    Closed,
}

impl QualityPlanner {
    pub fn new() -> Self {
        QualityPlanner {
            quality_standards: Self::initialize_standards(),
            quality_templates: Self::initialize_templates(),
            historical_data: Vec::new(),
        }
    }
    
    fn initialize_standards() -> HashMap<String, QualityStandard> {
        let mut standards = HashMap::new();
        
        // ISO/IEC 25010 标准
        standards.insert("ISO25010".to_string(), QualityStandard {
            name: "ISO/IEC 25010".to_string(),
            version: "2011".to_string(),
            description: "Systems and software Quality Requirements and Evaluation (SQuaRE)".to_string(),
            requirements: vec![
                QualityRequirement {
                    id: "FUNC_001".to_string(),
                    description: "功能完整性".to_string(),
                    category: "Functionality".to_string(),
                    priority: 1,
                    acceptance_criteria: vec!["所有必需功能都已实现".to_string()],
                },
                QualityRequirement {
                    id: "PERF_001".to_string(),
                    description: "性能效率".to_string(),
                    category: "Performance".to_string(),
                    priority: 2,
                    acceptance_criteria: vec!["响应时间小于2秒".to_string()],
                },
                QualityRequirement {
                    id: "SEC_001".to_string(),
                    description: "安全性".to_string(),
                    category: "Security".to_string(),
                    priority: 1,
                    acceptance_criteria: vec!["通过安全测试".to_string()],
                },
            ],
            metrics: vec![
                QualityMetric {
                    id: "FUNC_COV".to_string(),
                    name: "功能覆盖率".to_string(),
                    description: "已实现功能与需求功能的比率".to_string(),
                    measurement_method: "功能测试".to_string(),
                    unit: "%".to_string(),
                    target_value: 100.0,
                    acceptable_range: (95.0, 100.0),
                },
                QualityMetric {
                    id: "PERF_RESP".to_string(),
                    name: "响应时间".to_string(),
                    description: "系统响应时间".to_string(),
                    measurement_method: "性能测试".to_string(),
                    unit: "秒".to_string(),
                    target_value: 1.0,
                    acceptable_range: (0.5, 2.0),
                },
            ],
        });
        
        standards
    }
    
    fn initialize_templates() -> HashMap<String, QualityTemplate> {
        let mut templates = HashMap::new();
        
        // 软件开发质量模板
        templates.insert("software_development".to_string(), QualityTemplate {
            name: "软件开发质量模板".to_string(),
            project_type: "software".to_string(),
            quality_attributes: vec![
                QualityAttribute {
                    name: "功能性".to_string(),
                    value: 0.0,
                    weight: 0.25,
                    target: 0.95,
                    tolerance: 0.05,
                },
                QualityAttribute {
                    name: "性能效率".to_string(),
                    value: 0.0,
                    weight: 0.20,
                    target: 0.90,
                    tolerance: 0.10,
                },
                QualityAttribute {
                    name: "安全性".to_string(),
                    value: 0.0,
                    weight: 0.20,
                    target: 0.95,
                    tolerance: 0.05,
                },
                QualityAttribute {
                    name: "可用性".to_string(),
                    value: 0.0,
                    weight: 0.15,
                    target: 0.85,
                    tolerance: 0.15,
                },
                QualityAttribute {
                    name: "维护性".to_string(),
                    value: 0.0,
                    weight: 0.10,
                    target: 0.80,
                    tolerance: 0.20,
                },
                QualityAttribute {
                    name: "可移植性".to_string(),
                    value: 0.0,
                    weight: 0.10,
                    target: 0.75,
                    tolerance: 0.25,
                },
            ],
            quality_metrics: vec![
                QualityMetric {
                    id: "CODE_COV".to_string(),
                    name: "代码覆盖率".to_string(),
                    description: "单元测试代码覆盖率".to_string(),
                    measurement_method: "代码覆盖率工具".to_string(),
                    unit: "%".to_string(),
                    target_value: 90.0,
                    acceptable_range: (80.0, 100.0),
                },
                QualityMetric {
                    id: "DEFECT_DENSITY".to_string(),
                    name: "缺陷密度".to_string(),
                    description: "每千行代码的缺陷数".to_string(),
                    measurement_method: "缺陷跟踪系统".to_string(),
                    unit: "defects/KLOC".to_string(),
                    target_value: 1.0,
                    acceptable_range: (0.0, 2.0),
                },
            ],
            quality_controls: vec![
                QualityControl {
                    id: "CODE_REVIEW".to_string(),
                    name: "代码审查".to_string(),
                    description: "同行代码审查".to_string(),
                    control_type: ControlType::Preventive,
                    frequency: "每个功能完成时".to_string(),
                    responsible: "开发团队".to_string(),
                    tools: vec!["GitHub PR".to_string(), "SonarQube".to_string()],
                },
                QualityControl {
                    id: "UNIT_TEST".to_string(),
                    name: "单元测试".to_string(),
                    description: "自动化单元测试".to_string(),
                    control_type: ControlType::Detective,
                    frequency: "每次代码提交".to_string(),
                    responsible: "开发人员".to_string(),
                    tools: vec!["JUnit".to_string(), "pytest".to_string()],
                },
            ],
        });
        
        templates
    }
    
    pub fn create_quality_plan(&self, project_type: &str, project_id: &str) -> QualityPlan {
        let template = self.quality_templates.get(project_type)
            .expect("Quality template not found");
        
        let mut quality_goals = HashMap::new();
        for attr in &template.quality_attributes {
            quality_goals.insert(attr.name.clone(), attr.target);
        }
        
        QualityPlan {
            project_id: project_id.to_string(),
            quality_attributes: template.quality_attributes.clone(),
            quality_metrics: template.quality_metrics.clone(),
            quality_goals,
            quality_controls: template.quality_controls.clone(),
            quality_improvements: Vec::new(),
        }
    }
    
    pub fn add_quality_improvement(&mut self, plan: &mut QualityPlan, improvement: QualityImprovement) {
        plan.quality_improvements.push(improvement);
    }
    
    pub fn calculate_quality_score(&self, plan: &QualityPlan) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        for attr in &plan.quality_attributes {
            total_score += attr.value * attr.weight;
            total_weight += attr.weight;
        }
        
        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        }
    }
    
    pub fn check_quality_compliance(&self, plan: &QualityPlan) -> Vec<QualityIssue> {
        let mut issues = Vec::new();
        
        for attr in &plan.quality_attributes {
            let deviation = (attr.value - attr.target).abs();
            if deviation > attr.tolerance {
                issues.push(QualityIssue {
                    id: format!("issue_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
                    description: format!("质量属性 '{}' 不符合要求: 当前值 {:.2}, 目标值 {:.2}", 
                                       attr.name, attr.value, attr.target),
                    severity: if deviation > attr.tolerance * 2.0 { Severity::High } else { Severity::Medium },
                    category: attr.name.clone(),
                    status: IssueStatus::Open,
                });
            }
        }
        
        issues
    }
}
```

## 2.4.3 质量保证模型

### 质量保证体系

**定义 2.4.6** (质量保证) 质量保证是一个函数：
$$\text{QualityAssurance}: \mathcal{P} \times \mathcal{Q} \rightarrow \{True, False\}$$

定义为：
$$\text{QualityAssurance}(p, q) = \text{Quality}(q) \geq \text{QualityGoal}(p)$$

**定义 2.4.7** (质量保证活动) 质量保证活动集合：
$$\mathcal{QA} = \{\text{Planning}, \text{Review}, \text{Testing}, \text{Monitoring}, \text{Reporting}\}$$

### 质量保证算法

**算法 2.4.2** (质量保证算法)：

```rust
use std::collections::HashMap;

#[derive(Debug)]
pub struct QualityAssurance {
    pub quality_plan: QualityPlan,
    pub quality_activities: Vec<QualityActivity>,
    pub quality_reports: Vec<QualityReport>,
    pub quality_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct QualityActivity {
    pub id: String,
    pub name: String,
    pub activity_type: ActivityType,
    pub status: ActivityStatus,
    pub start_date: f64,
    pub end_date: f64,
    pub responsible: String,
    pub results: Vec<ActivityResult>,
}

#[derive(Debug, Clone)]
pub enum ActivityType {
    Planning,
    Review,
    Testing,
    Monitoring,
    Reporting,
}

#[derive(Debug, Clone)]
pub enum ActivityStatus {
    Planned,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct ActivityResult {
    pub metric_id: String,
    pub measured_value: f64,
    pub target_value: f64,
    pub status: ResultStatus,
    pub comments: String,
}

#[derive(Debug, Clone)]
pub enum ResultStatus {
    Pass,
    Fail,
    Warning,
}

#[derive(Debug, Clone)]
pub struct QualityReport {
    pub id: String,
    pub report_date: f64,
    pub quality_score: f64,
    pub quality_metrics: HashMap<String, f64>,
    pub issues: Vec<QualityIssue>,
    pub recommendations: Vec<String>,
}

impl QualityAssurance {
    pub fn new(quality_plan: QualityPlan) -> Self {
        QualityAssurance {
            quality_plan,
            quality_activities: Vec::new(),
            quality_reports: Vec::new(),
            quality_metrics: HashMap::new(),
        }
    }
    
    pub fn add_activity(&mut self, activity: QualityActivity) {
        self.quality_activities.push(activity);
    }
    
    pub fn execute_activity(&mut self, activity_id: &str) -> Result<Vec<ActivityResult>, String> {
        if let Some(activity) = self.quality_activities.iter_mut().find(|a| a.id == activity_id) {
            activity.status = ActivityStatus::InProgress;
            
            let results = match activity.activity_type {
                ActivityType::Planning => self.execute_planning_activity(activity),
                ActivityType::Review => self.execute_review_activity(activity),
                ActivityType::Testing => self.execute_testing_activity(activity),
                ActivityType::Monitoring => self.execute_monitoring_activity(activity),
                ActivityType::Reporting => self.execute_reporting_activity(activity),
            };
            
            activity.results = results.clone();
            activity.status = ActivityStatus::Completed;
            
            Ok(results)
        } else {
            Err("Activity not found".to_string())
        }
    }
    
    fn execute_planning_activity(&self, activity: &QualityActivity) -> Vec<ActivityResult> {
        let mut results = Vec::new();
        
        // 质量规划活动
        for metric in &self.quality_plan.quality_metrics {
            let result = ActivityResult {
                metric_id: metric.id.clone(),
                measured_value: 0.0, // 规划阶段为0
                target_value: metric.target_value,
                status: ResultStatus::Pass,
                comments: "质量目标已设定".to_string(),
            };
            results.push(result);
        }
        
        results
    }
    
    fn execute_review_activity(&self, activity: &QualityActivity) -> Vec<ActivityResult> {
        let mut results = Vec::new();
        
        // 代码审查活动
        let code_review_metrics = vec![
            ("CODE_QUALITY", 0.85, 0.80),
            ("DOCUMENTATION", 0.90, 0.85),
            ("STANDARDS_COMPLIANCE", 0.95, 0.90),
        ];
        
        for (metric_name, measured_value, target_value) in code_review_metrics {
            let status = if measured_value >= target_value {
                ResultStatus::Pass
            } else if measured_value >= target_value * 0.9 {
                ResultStatus::Warning
            } else {
                ResultStatus::Fail
            };
            
            let result = ActivityResult {
                metric_id: metric_name.to_string(),
                measured_value,
                target_value,
                status,
                comments: "代码审查完成".to_string(),
            };
            results.push(result);
        }
        
        results
    }
    
    fn execute_testing_activity(&self, activity: &QualityActivity) -> Vec<ActivityResult> {
        let mut results = Vec::new();
        
        // 测试活动
        let testing_metrics = vec![
            ("CODE_COVERAGE", 92.5, 90.0),
            ("FUNCTIONAL_TEST_PASS_RATE", 98.0, 95.0),
            ("PERFORMANCE_TEST_PASS_RATE", 96.0, 90.0),
            ("SECURITY_TEST_PASS_RATE", 100.0, 95.0),
        ];
        
        for (metric_name, measured_value, target_value) in testing_metrics {
            let status = if measured_value >= target_value {
                ResultStatus::Pass
            } else if measured_value >= target_value * 0.9 {
                ResultStatus::Warning
            } else {
                ResultStatus::Fail
            };
            
            let result = ActivityResult {
                metric_id: metric_name.to_string(),
                measured_value,
                target_value,
                status,
                comments: "测试执行完成".to_string(),
            };
            results.push(result);
        }
        
        results
    }
    
    fn execute_monitoring_activity(&self, activity: &QualityActivity) -> Vec<ActivityResult> {
        let mut results = Vec::new();
        
        // 质量监控活动
        let monitoring_metrics = vec![
            ("DEFECT_DENSITY", 0.8, 1.0),
            ("MEAN_TIME_TO_RESOLVE", 2.5, 3.0),
            ("CUSTOMER_SATISFACTION", 4.2, 4.0),
        ];
        
        for (metric_name, measured_value, target_value) in monitoring_metrics {
            let status = if measured_value >= target_value {
                ResultStatus::Pass
            } else if measured_value >= target_value * 0.9 {
                ResultStatus::Warning
            } else {
                ResultStatus::Fail
            };
            
            let result = ActivityResult {
                metric_id: metric_name.to_string(),
                measured_value,
                target_value,
                status,
                comments: "质量监控完成".to_string(),
            };
            results.push(result);
        }
        
        results
    }
    
    fn execute_reporting_activity(&self, activity: &QualityActivity) -> Vec<ActivityResult> {
        let mut results = Vec::new();
        
        // 质量报告活动
        let overall_quality_score = self.calculate_overall_quality_score();
        let target_score = 0.85;
        
        let result = ActivityResult {
            metric_id: "OVERALL_QUALITY_SCORE".to_string(),
            measured_value: overall_quality_score,
            target_value: target_score,
            status: if overall_quality_score >= target_score {
                ResultStatus::Pass
            } else {
                ResultStatus::Fail
            },
            comments: "质量报告生成完成".to_string(),
        };
        results.push(result);
        
        results
    }
    
    fn calculate_overall_quality_score(&self) -> f64 {
        // 计算整体质量分数
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        for attr in &self.quality_plan.quality_attributes {
            let current_value = self.quality_metrics.get(&attr.name).unwrap_or(&0.0);
            total_score += current_value * attr.weight;
            total_weight += attr.weight;
        }
        
        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        }
    }
    
    pub fn generate_quality_report(&mut self) -> QualityReport {
        let quality_score = self.calculate_overall_quality_score();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        
        // 分析质量问题
        for attr in &self.quality_plan.quality_attributes {
            let current_value = self.quality_metrics.get(&attr.name).unwrap_or(&0.0);
            if current_value < &attr.target {
                issues.push(QualityIssue {
                    id: format!("issue_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
                    description: format!("质量属性 '{}' 未达标: {:.2} < {:.2}", 
                                       attr.name, current_value, attr.target),
                    severity: if current_value < &(attr.target * 0.8) { Severity::High } else { Severity::Medium },
                    category: attr.name.clone(),
                    status: IssueStatus::Open,
                });
                
                recommendations.push(format!("改进质量属性 '{}' 到目标值 {:.2}", attr.name, attr.target));
            }
        }
        
        let report = QualityReport {
            id: format!("report_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
            report_date: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as f64,
            quality_score,
            quality_metrics: self.quality_metrics.clone(),
            issues,
            recommendations,
        };
        
        self.quality_reports.push(report.clone());
        report
    }
}
```

## 2.4.4 质量控制模型

### 质量控制体系

**定义 2.4.8** (质量控制) 质量控制是一个函数：
$$\text{QualityControl}: \mathcal{P} \times \mathcal{M} \rightarrow \mathcal{A}$$

其中 $\mathcal{A}$ 是控制动作集合。

**定义 2.4.9** (控制图) 控制图是一个三元组：
$$CC = (D, LCL, UCL)$$

其中：
- $D$ 是数据点集合
- $LCL$ 是下控制限
- $UCL$ 是上控制限

### 质量控制算法

**算法 2.4.3** (质量控制算法)：

```rust
use std::collections::VecDeque;

#[derive(Debug)]
pub struct QualityController {
    pub control_charts: HashMap<String, ControlChart>,
    pub control_rules: Vec<ControlRule>,
    pub control_actions: Vec<ControlAction>,
}

#[derive(Debug, Clone)]
pub struct ControlChart {
    pub metric_id: String,
    pub data_points: VecDeque<DataPoint>,
    pub center_line: f64,
    pub upper_control_limit: f64,
    pub lower_control_limit: f64,
    pub warning_limits: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct DataPoint {
    pub timestamp: f64,
    pub value: f64,
    pub sample_size: usize,
}

#[derive(Debug, Clone)]
pub struct ControlRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub condition: RuleCondition,
    pub action: String,
}

#[derive(Debug, Clone)]
pub enum RuleCondition {
    PointAboveUCL,
    PointBelowLCL,
    TrendUp,
    TrendDown,
    RunAboveCenter,
    RunBelowCenter,
}

#[derive(Debug, Clone)]
pub struct ControlAction {
    pub id: String,
    pub name: String,
    pub description: String,
    pub action_type: ActionType,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum ActionType {
    Adjust,
    Stop,
    Investigate,
    Notify,
}

impl QualityController {
    pub fn new() -> Self {
        QualityController {
            control_charts: HashMap::new(),
            control_rules: Self::initialize_control_rules(),
            control_actions: Vec::new(),
        }
    }
    
    fn initialize_control_rules() -> Vec<ControlRule> {
        vec![
            ControlRule {
                id: "RULE_001".to_string(),
                name: "超出控制限".to_string(),
                description: "数据点超出上控制限或下控制限".to_string(),
                condition: RuleCondition::PointAboveUCL,
                action: "立即调查并采取纠正措施".to_string(),
            },
            ControlRule {
                id: "RULE_002".to_string(),
                name: "上升趋势".to_string(),
                description: "连续7个点呈上升趋势".to_string(),
                condition: RuleCondition::TrendUp,
                action: "分析趋势原因并调整过程".to_string(),
            },
            ControlRule {
                id: "RULE_003".to_string(),
                name: "中心线偏移".to_string(),
                description: "连续8个点在中心线同一侧".to_string(),
                condition: RuleCondition::RunAboveCenter,
                action: "检查过程是否发生系统性变化".to_string(),
            },
        ]
    }
    
    pub fn add_control_chart(&mut self, metric_id: String, chart: ControlChart) {
        self.control_charts.insert(metric_id, chart);
    }
    
    pub fn add_data_point(&mut self, metric_id: &str, data_point: DataPoint) -> Vec<ControlAction> {
        if let Some(chart) = self.control_charts.get_mut(metric_id) {
            chart.data_points.push_back(data_point.clone());
            
            // 保持控制图大小
            if chart.data_points.len() > 100 {
                chart.data_points.pop_front();
            }
            
            // 检查控制规则
            self.check_control_rules(chart, &data_point)
        } else {
            Vec::new()
        }
    }
    
    fn check_control_rules(&mut self, chart: &ControlChart, data_point: &DataPoint) -> Vec<ControlAction> {
        let mut actions = Vec::new();
        
        for rule in &self.control_rules {
            if self.evaluate_rule(rule, chart, data_point) {
                let action = self.create_control_action(rule, data_point);
                actions.push(action);
            }
        }
        
        actions
    }
    
    fn evaluate_rule(&self, rule: &ControlRule, chart: &ControlChart, data_point: &DataPoint) -> bool {
        match rule.condition {
            RuleCondition::PointAboveUCL => {
                data_point.value > chart.upper_control_limit
            }
            RuleCondition::PointBelowLCL => {
                data_point.value < chart.lower_control_limit
            }
            RuleCondition::TrendUp => {
                self.check_trend(chart, true)
            }
            RuleCondition::TrendDown => {
                self.check_trend(chart, false)
            }
            RuleCondition::RunAboveCenter => {
                self.check_run(chart, true)
            }
            RuleCondition::RunBelowCenter => {
                self.check_run(chart, false)
            }
        }
    }
    
    fn check_trend(&self, chart: &ControlChart, upward: bool) -> bool {
        if chart.data_points.len() < 7 {
            return false;
        }
        
        let recent_points: Vec<f64> = chart.data_points.iter()
            .rev()
            .take(7)
            .map(|p| p.value)
            .collect();
        
        let mut trend_count = 0;
        for i in 1..recent_points.len() {
            if upward && recent_points[i] > recent_points[i-1] {
                trend_count += 1;
            } else if !upward && recent_points[i] < recent_points[i-1] {
                trend_count += 1;
            }
        }
        
        trend_count >= 6 // 至少6个点呈趋势
    }
    
    fn check_run(&self, chart: &ControlChart, above_center: bool) -> bool {
        if chart.data_points.len() < 8 {
            return false;
        }
        
        let recent_points: Vec<f64> = chart.data_points.iter()
            .rev()
            .take(8)
            .map(|p| p.value)
            .collect();
        
        let mut run_count = 0;
        for &value in &recent_points {
            if above_center && value > chart.center_line {
                run_count += 1;
            } else if !above_center && value < chart.center_line {
                run_count += 1;
            } else {
                break;
            }
        }
        
        run_count >= 8
    }
    
    fn create_control_action(&self, rule: &ControlRule, data_point: &DataPoint) -> ControlAction {
        let mut parameters = HashMap::new();
        parameters.insert("value".to_string(), data_point.value);
        parameters.insert("timestamp".to_string(), data_point.timestamp);
        
        ControlAction {
            id: format!("action_{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
            name: rule.name.clone(),
            description: rule.action.clone(),
            action_type: ActionType::Investigate,
            parameters,
        }
    }
    
    pub fn calculate_control_limits(&mut self, metric_id: &str) -> Result<(f64, f64, f64), String> {
        if let Some(chart) = self.control_charts.get(metric_id) {
            if chart.data_points.len() < 20 {
                return Err("Insufficient data for control limit calculation".to_string());
            }
            
            let values: Vec<f64> = chart.data_points.iter().map(|p| p.value).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            
            let variance = values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();
            
            let ucl = mean + 3.0 * std_dev;
            let lcl = mean - 3.0 * std_dev;
            
            Ok((mean, lcl, ucl))
        } else {
            Err("Control chart not found".to_string())
        }
    }
}
```

## 2.4.5 质量改进模型

### 质量改进体系

**定义 2.4.10** (质量改进) 质量改进是一个函数：
$$\text{QualityImprovement}: \mathcal{Q} \times \mathcal{I} \rightarrow \mathcal{Q}$$

其中 $\mathcal{I}$ 是改进措施集合。

**定义 2.4.11** (改进效果) 改进效果是一个函数：
$$\text{ImprovementEffect}: \mathcal{I} \times \mathcal{Q} \rightarrow \mathbb{R}^+$$

### 质量改进算法

**算法 2.4.4** (质量改进算法)：

```rust
use std::collections::HashMap;

#[derive(Debug)]
pub struct QualityImprovement {
    pub improvement_projects: Vec<ImprovementProject>,
    pub improvement_metrics: HashMap<String, f64>,
    pub improvement_history: Vec<ImprovementRecord>,
}

#[derive(Debug, Clone)]
pub struct ImprovementProject {
    pub id: String,
    pub name: String,
    pub description: String,
    pub target_metric: String,
    pub current_value: f64,
    pub target_value: f64,
    pub improvement_actions: Vec<ImprovementAction>,
    pub status: ProjectStatus,
    pub start_date: f64,
    pub end_date: f64,
}

#[derive(Debug, Clone)]
pub struct ImprovementAction {
    pub id: String,
    pub name: String,
    pub description: String,
    pub action_type: ActionType,
    pub cost: f64,
    pub expected_improvement: f64,
    pub implementation_time: f64,
    pub status: ActionStatus,
}

#[derive(Debug, Clone)]
pub enum ProjectStatus {
    Planning,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum ActionStatus {
    Planned,
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ImprovementRecord {
    pub project_id: String,
    pub metric_id: String,
    pub before_value: f64,
    pub after_value: f64,
    pub improvement: f64,
    pub cost: f64,
    pub roi: f64,
    pub completion_date: f64,
}

impl QualityImprovement {
    pub fn new() -> Self {
        QualityImprovement {
            improvement_projects: Vec::new(),
            improvement_metrics: HashMap::new(),
            improvement_history: Vec::new(),
        }
    }
    
    pub fn add_improvement_project(&mut self, project: ImprovementProject) {
        self.improvement_projects.push(project);
    }
    
    pub fn execute_improvement_project(&mut self, project_id: &str) -> Result<ImprovementRecord, String> {
        if let Some(project) = self.improvement_projects.iter_mut().find(|p| p.id == project_id) {
            project.status = ProjectStatus::InProgress;
            
            let before_value = project.current_value;
            let mut total_cost = 0.0;
            let mut total_improvement = 0.0;
            
            // 执行改进措施
            for action in &mut project.improvement_actions {
                action.status = ActionStatus::InProgress;
                
                let improvement = self.execute_improvement_action(action);
                total_improvement += improvement;
                total_cost += action.cost;
                
                action.status = ActionStatus::Completed;
            }
            
            let after_value = before_value + total_improvement;
            let roi = if total_cost > 0.0 {
                total_improvement / total_cost
            } else {
                0.0
            };
            
            project.status = ProjectStatus::Completed;
            project.current_value = after_value;
            
            let record = ImprovementRecord {
                project_id: project_id.to_string(),
                metric_id: project.target_metric.clone(),
                before_value,
                after_value,
                improvement: total_improvement,
                cost: total_cost,
                roi,
                completion_date: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as f64,
            };
            
            self.improvement_history.push(record.clone());
            self.improvement_metrics.insert(project.target_metric.clone(), after_value);
            
            Ok(record)
        } else {
            Err("Improvement project not found".to_string())
        }
    }
    
    fn execute_improvement_action(&self, action: &ImprovementAction) -> f64 {
        // 模拟改进措施的执行效果
        match action.action_type {
            ActionType::Adjust => {
                action.expected_improvement * 0.8 // 80%的预期效果
            }
            ActionType::Stop => {
                action.expected_improvement * 0.9 // 90%的预期效果
            }
            ActionType::Investigate => {
                action.expected_improvement * 0.7 // 70%的预期效果
            }
            ActionType::Notify => {
                action.expected_improvement * 0.5 // 50%的预期效果
            }
        }
    }
    
    pub fn calculate_improvement_roi(&self) -> f64 {
        let total_improvement: f64 = self.improvement_history.iter()
            .map(|r| r.improvement)
            .sum();
        
        let total_cost: f64 = self.improvement_history.iter()
            .map(|r| r.cost)
            .sum();
        
        if total_cost > 0.0 {
            total_improvement / total_cost
        } else {
            0.0
        }
    }
    
    pub fn get_improvement_trend(&self, metric_id: &str) -> Vec<(f64, f64)> {
        let mut trend = Vec::new();
        
        for record in &self.improvement_history {
            if record.metric_id == metric_id {
                trend.push((record.completion_date, record.after_value));
            }
        }
        
        trend.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        trend
    }
}
```

## 2.4.6 国际标准对标

### ISO/IEC 25010 标准

- **质量模型**: 8个质量特性（功能性、性能效率、兼容性、易用性、可靠性、安全性、可维护性、可移植性）
- **质量度量**: 标准化的质量度量方法
- **质量评估**: 质量评估过程和标准

### ISO 9001 标准

- **质量管理体系**: 质量管理体系要求
- **质量方针**: 质量方针和目标
- **质量策划**: 质量策划和控制
- **质量改进**: 持续改进机制

### CMMI-DEV 标准

- **过程域**: 过程改进和能力评估
- **成熟度等级**: 5个成熟度等级
- **最佳实践**: 软件工程最佳实践

## 2.4.7 相关链接

- [2.1 项目生命周期模型](./lifecycle-models.md)
- [2.2 资源管理模型](./resource-models.md)
- [2.3 风险管理模型](./risk-models.md)
- [1.1 形式化基础理论](../01-foundations/README.md)
- [3.1 形式化验证理论](../03-formal-verification/verification-theory.md)

## 参考文献

1. ISO/IEC 25010:2011. Systems and software Quality Requirements and Evaluation (SQuaRE) - System and software quality models.
2. ISO 9001:2015. Quality management systems - Requirements.
3. CMMI Product Team. (2010). CMMI for Development, Version 1.3. Software Engineering Institute.
4. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
5. ISO 21500:2012. Guidance on project management. International Organization for Standardization.
6. Kerzner, H. (2017). Project management: a systems approach to planning, scheduling, and controlling (12th ed.). John Wiley & Sons.
7. Meredith, J. R., & Mantel, S. J. (2019). Project management: a managerial approach (10th ed.). John Wiley & Sons.
8. Turner, J. R. (2016). Gower handbook of project management (5th ed.). Routledge.
9. Lock, D. (2013). Project management (10th ed.). Routledge.
10. Schwalbe, K. (2019). Information technology project management (9th ed.). Cengage Learning.
