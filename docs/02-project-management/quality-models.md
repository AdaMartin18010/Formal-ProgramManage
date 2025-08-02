# 2.4 质量管理模型

## 概述

质量管理模型是Formal-ProgramManage的核心组成部分，提供形式化的数学框架来确保项目交付物满足预定的质量标准和要求。

## 2.4.1 质量基础定义

### 质量定义

**定义 2.4.1** 项目质量是一个四元组 $Q = (S, M, T, C)$，其中：

- $S$ 是质量标准 (Standards)
- $M$ 是度量指标 (Metrics)
- $T$ 是目标值 (Targets)
- $C$ 是约束条件 (Constraints)

### 质量维度

**定义 2.4.2** 质量维度集合 $\mathcal{D} = \{Functionality, Reliability, Usability, Efficiency, Maintainability, Portability\}$

**定义 2.4.3** 功能性质量 $Q_{func} = (S_{func}, M_{func}, T_{func}, C_{func})$

**定义 2.4.4** 可靠性质量 $Q_{reli} = (S_{reli}, M_{reli}, T_{reli}, C_{reli})$

**定义 2.4.5** 可用性质量 $Q_{usab} = (S_{usab}, M_{usab}, T_{usab}, C_{usab})$

**定义 2.4.6** 效率质量 $Q_{effi} = (S_{effi}, M_{effi}, T_{effi}, C_{effi})$

**定义 2.4.7** 可维护性质量 $Q_{main} = (S_{main}, M_{main}, T_{main}, C_{main})$

**定义 2.4.8** 可移植性质量 $Q_{port} = (S_{port}, M_{port}, T_{port}, C_{port})$

## 2.4.2 质量规划模型

### 质量规划函数

**定义 2.4.9** 质量规划函数 $QP: \mathcal{P} \rightarrow \mathcal{Q}$，其中：

- $\mathcal{P}$ 是项目集合
- $\mathcal{Q}$ 是质量计划集合

### 质量目标设定

**定义 2.4.10** 质量目标函数：
$$QO(p) = \{(d, t_d, w_d) \mid d \in \mathcal{D}, t_d \in \mathbb{R}, w_d \in [0,1]\}$$

其中：

- $d$ 是质量维度
- $t_d$ 是目标值
- $w_d$ 是权重

### 质量规划算法

**算法 2.4.1** 质量规划算法：

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct QualityPlan {
    pub project_id: String,
    pub quality_objectives: HashMap<QualityDimension, QualityObjective>,
    pub quality_metrics: Vec<QualityMetric>,
    pub quality_gates: Vec<QualityGate>,
    pub quality_controls: Vec<QualityControl>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum QualityDimension {
    Functionality,
    Reliability,
    Usability,
    Efficiency,
    Maintainability,
    Portability,
}

#[derive(Debug, Clone)]
pub struct QualityObjective {
    pub dimension: QualityDimension,
    pub target_value: f64,
    pub weight: f64,
    pub priority: u32,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct QualityMetric {
    pub id: String,
    pub name: String,
    pub dimension: QualityDimension,
    pub metric_type: MetricType,
    pub measurement_unit: String,
    pub target_value: f64,
    pub acceptable_range: (f64, f64),
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub enum MetricType {
    DefectRate,
    CodeCoverage,
    Performance,
    Usability,
    Reliability,
    Maintainability,
    Security,
    Compliance,
}

#[derive(Debug, Clone)]
pub struct QualityGate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub criteria: Vec<QualityCriterion>,
    pub status: GateStatus,
    pub required_approval: bool,
    pub phase: ProjectPhase,
}

#[derive(Debug, Clone)]
pub struct QualityCriterion {
    pub metric_id: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

#[derive(Debug, Clone)]
pub enum GateStatus {
    Pending,
    Passed,
    Failed,
    Conditional,
}

#[derive(Debug, Clone)]
pub enum ProjectPhase {
    Initiation,
    Planning,
    Execution,
    Monitoring,
    Closure,
}

#[derive(Debug, Clone)]
pub struct QualityControl {
    pub id: String,
    pub name: String,
    pub control_type: ControlType,
    pub frequency: ControlFrequency,
    pub responsible: String,
    pub tools: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ControlType {
    Inspection,
    Testing,
    Review,
    Audit,
    Monitoring,
}

#[derive(Debug, Clone)]
pub enum ControlFrequency {
    Continuous,
    Daily,
    Weekly,
    Monthly,
    Milestone,
    Custom(u32),
}

pub struct QualityPlanner {
    pub templates: HashMap<String, QualityTemplate>,
    pub best_practices: Vec<QualityBestPractice>,
    pub industry_standards: Vec<IndustryStandard>,
}

#[derive(Debug, Clone)]
pub struct QualityTemplate {
    pub id: String,
    pub name: String,
    pub project_type: ProjectType,
    pub quality_objectives: Vec<QualityObjective>,
    pub quality_metrics: Vec<QualityMetric>,
    pub quality_gates: Vec<QualityGate>,
}

#[derive(Debug, Clone)]
pub enum ProjectType {
    SoftwareDevelopment,
    Construction,
    Research,
    Manufacturing,
    Service,
}

#[derive(Debug, Clone)]
pub struct QualityBestPractice {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: BestPracticeCategory,
    pub effectiveness: f64,
    pub implementation_cost: f64,
}

#[derive(Debug, Clone)]
pub enum BestPracticeCategory {
    Planning,
    Execution,
    Monitoring,
    Improvement,
}

#[derive(Debug, Clone)]
pub struct IndustryStandard {
    pub id: String,
    pub name: String,
    pub organization: String,
    pub version: String,
    pub requirements: Vec<QualityRequirement>,
}

#[derive(Debug, Clone)]
pub struct QualityRequirement {
    pub id: String,
    pub description: String,
    pub mandatory: bool,
    pub verification_method: String,
}

impl QualityPlanner {
    pub fn new() -> Self {
        QualityPlanner {
            templates: Self::initialize_templates(),
            best_practices: Self::initialize_best_practices(),
            industry_standards: Self::initialize_industry_standards(),
        }
    }
    
    fn initialize_templates() -> HashMap<String, QualityTemplate> {
        let mut templates = HashMap::new();
        
        // 软件开发质量模板
        templates.insert("software_development".to_string(), QualityTemplate {
            id: "SW_QUAL_001".to_string(),
            name: "Software Development Quality Template".to_string(),
            project_type: ProjectType::SoftwareDevelopment,
            quality_objectives: vec![
                QualityObjective {
                    dimension: QualityDimension::Functionality,
                    target_value: 0.95,
                    weight: 0.3,
                    priority: 1,
                    description: "Ensure all functional requirements are met".to_string(),
                },
                QualityObjective {
                    dimension: QualityDimension::Reliability,
                    target_value: 0.99,
                    weight: 0.25,
                    priority: 2,
                    description: "Ensure high system reliability".to_string(),
                },
                QualityObjective {
                    dimension: QualityDimension::Usability,
                    target_value: 0.85,
                    weight: 0.2,
                    priority: 3,
                    description: "Ensure good user experience".to_string(),
                },
                QualityObjective {
                    dimension: QualityDimension::Maintainability,
                    target_value: 0.8,
                    weight: 0.15,
                    priority: 4,
                    description: "Ensure code maintainability".to_string(),
                },
                QualityObjective {
                    dimension: QualityDimension::Efficiency,
                    target_value: 0.9,
                    weight: 0.1,
                    priority: 5,
                    description: "Ensure system performance".to_string(),
                },
            ],
            quality_metrics: vec![
                QualityMetric {
                    id: "METRIC-001".to_string(),
                    name: "Code Coverage".to_string(),
                    dimension: QualityDimension::Reliability,
                    metric_type: MetricType::CodeCoverage,
                    measurement_unit: "percentage".to_string(),
                    target_value: 90.0,
                    acceptable_range: (80.0, 100.0),
                    weight: 0.3,
                },
                QualityMetric {
                    id: "METRIC-002".to_string(),
                    name: "Defect Rate".to_string(),
                    dimension: QualityDimension::Reliability,
                    metric_type: MetricType::DefectRate,
                    measurement_unit: "defects/KLOC".to_string(),
                    target_value: 1.0,
                    acceptable_range: (0.0, 2.0),
                    weight: 0.25,
                },
                QualityMetric {
                    id: "METRIC-003".to_string(),
                    name: "Response Time".to_string(),
                    dimension: QualityDimension::Efficiency,
                    metric_type: MetricType::Performance,
                    measurement_unit: "milliseconds".to_string(),
                    target_value: 100.0,
                    acceptable_range: (0.0, 200.0),
                    weight: 0.2,
                },
            ],
            quality_gates: vec![
                QualityGate {
                    id: "GATE-001".to_string(),
                    name: "Requirements Review".to_string(),
                    description: "Review and approve requirements".to_string(),
                    criteria: vec![
                        QualityCriterion {
                            metric_id: "METRIC-001".to_string(),
                            operator: ComparisonOperator::GreaterThanOrEqual,
                            threshold: 0.8,
                            weight: 1.0,
                        },
                    ],
                    status: GateStatus::Pending,
                    required_approval: true,
                    phase: ProjectPhase::Planning,
                },
                QualityGate {
                    id: "GATE-002".to_string(),
                    name: "Code Review".to_string(),
                    description: "Review code quality and standards".to_string(),
                    criteria: vec![
                        QualityCriterion {
                            metric_id: "METRIC-002".to_string(),
                            operator: ComparisonOperator::LessThan,
                            threshold: 2.0,
                            weight: 1.0,
                        },
                    ],
                    status: GateStatus::Pending,
                    required_approval: true,
                    phase: ProjectPhase::Execution,
                },
            ],
        });
        
        templates
    }
    
    fn initialize_best_practices() -> Vec<QualityBestPractice> {
        vec![
            QualityBestPractice {
                id: "BP-001".to_string(),
                name: "Code Review".to_string(),
                description: "Regular code review sessions".to_string(),
                category: BestPracticeCategory::Execution,
                effectiveness: 0.8,
                implementation_cost: 0.1,
            },
            QualityBestPractice {
                id: "BP-002".to_string(),
                name: "Automated Testing".to_string(),
                description: "Comprehensive automated testing".to_string(),
                category: BestPracticeCategory::Execution,
                effectiveness: 0.9,
                implementation_cost: 0.3,
            },
            QualityBestPractice {
                id: "BP-003".to_string(),
                name: "Continuous Integration".to_string(),
                description: "Continuous integration and deployment".to_string(),
                category: BestPracticeCategory::Monitoring,
                effectiveness: 0.85,
                implementation_cost: 0.2,
            },
        ]
    }
    
    fn initialize_industry_standards() -> Vec<IndustryStandard> {
        vec![
            IndustryStandard {
                id: "ISO-9001".to_string(),
                name: "ISO 9001:2015".to_string(),
                organization: "ISO".to_string(),
                version: "2015".to_string(),
                requirements: vec![
                    QualityRequirement {
                        id: "REQ-001".to_string(),
                        description: "Quality management system".to_string(),
                        mandatory: true,
                        verification_method: "Audit".to_string(),
                    },
                ],
            },
            IndustryStandard {
                id: "CMMI".to_string(),
                name: "CMMI for Development".to_string(),
                organization: "SEI".to_string(),
                version: "1.3".to_string(),
                requirements: vec![
                    QualityRequirement {
                        id: "REQ-002".to_string(),
                        description: "Process improvement framework".to_string(),
                        mandatory: true,
                        verification_method: "Assessment".to_string(),
                    },
                ],
            },
        ]
    }
    
    pub fn create_quality_plan(&self, project: &Project) -> QualityPlan {
        let template = self.select_template(project);
        let objectives = self.adapt_objectives(&template.quality_objectives, project);
        let metrics = self.adapt_metrics(&template.quality_metrics, project);
        let gates = self.adapt_gates(&template.quality_gates, project);
        let controls = self.create_quality_controls(project);
        
        QualityPlan {
            project_id: project.id.clone(),
            quality_objectives: objectives,
            quality_metrics: metrics,
            quality_gates: gates,
            quality_controls: controls,
        }
    }
    
    fn select_template(&self, project: &Project) -> &QualityTemplate {
        // 根据项目类型选择模板
        match project.project_type() {
            ProjectType::SoftwareDevelopment => {
                self.templates.get("software_development").unwrap()
            },
            _ => {
                // 默认模板
                self.templates.get("software_development").unwrap()
            },
        }
    }
    
    fn adapt_objectives(&self, template_objectives: &[QualityObjective], project: &Project) -> HashMap<QualityDimension, QualityObjective> {
        let mut objectives = HashMap::new();
        
        for objective in template_objectives {
            let adapted_objective = QualityObjective {
                dimension: objective.dimension.clone(),
                target_value: self.adjust_target_value(objective.target_value, project),
                weight: objective.weight,
                priority: objective.priority,
                description: objective.description.clone(),
            };
            
            objectives.insert(objective.dimension.clone(), adapted_objective);
        }
        
        objectives
    }
    
    fn adapt_metrics(&self, template_metrics: &[QualityMetric], project: &Project) -> Vec<QualityMetric> {
        template_metrics.iter()
            .map(|metric| QualityMetric {
                id: format!("{}-{}", metric.id, project.id),
                name: metric.name.clone(),
                dimension: metric.dimension.clone(),
                metric_type: metric.metric_type.clone(),
                measurement_unit: metric.measurement_unit.clone(),
                target_value: self.adjust_target_value(metric.target_value, project),
                acceptable_range: metric.acceptable_range,
                weight: metric.weight,
            })
            .collect()
    }
    
    fn adapt_gates(&self, template_gates: &[QualityGate], project: &Project) -> Vec<QualityGate> {
        template_gates.iter()
            .map(|gate| QualityGate {
                id: format!("{}-{}", gate.id, project.id),
                name: gate.name.clone(),
                description: gate.description.clone(),
                criteria: gate.criteria.clone(),
                status: GateStatus::Pending,
                required_approval: gate.required_approval,
                phase: gate.phase.clone(),
            })
            .collect()
    }
    
    fn create_quality_controls(&self, project: &Project) -> Vec<QualityControl> {
        vec![
            QualityControl {
                id: "CONTROL-001".to_string(),
                name: "Code Review".to_string(),
                control_type: ControlType::Review,
                frequency: ControlFrequency::Continuous,
                responsible: "Development Team".to_string(),
                tools: vec!["GitHub".to_string(), "SonarQube".to_string()],
            },
            QualityControl {
                id: "CONTROL-002".to_string(),
                name: "Automated Testing".to_string(),
                control_type: ControlType::Testing,
                frequency: ControlFrequency::Continuous,
                responsible: "QA Team".to_string(),
                tools: vec!["JUnit".to_string(), "Selenium".to_string()],
            },
            QualityControl {
                id: "CONTROL-003".to_string(),
                name: "Performance Testing".to_string(),
                control_type: ControlType::Testing,
                frequency: ControlFrequency::Weekly,
                responsible: "Performance Team".to_string(),
                tools: vec!["JMeter".to_string(), "LoadRunner".to_string()],
            },
        ]
    }
    
    fn adjust_target_value(&self, base_value: f64, project: &Project) -> f64 {
        // 根据项目特征调整目标值
        let complexity_factor = project.complexity_factor();
        let experience_factor = project.team_experience_factor();
        
        base_value * complexity_factor * experience_factor
    }
}

// 项目扩展方法（简化实现）
impl Project {
    fn project_type(&self) -> ProjectType {
        ProjectType::SoftwareDevelopment
    }
    
    fn complexity_factor(&self) -> f64 {
        // 简化实现
        1.0
    }
    
    fn team_experience_factor(&self) -> f64 {
        // 简化实现
        1.0
    }
}
```

## 2.4.3 质量保证模型

### 质量保证函数

**定义 2.4.11** 质量保证函数 $QA: \mathcal{P} \times \mathcal{Q} \rightarrow \mathcal{R}$，其中：

- $\mathcal{P}$ 是项目集合
- $\mathcal{Q}$ 是质量计划集合
- $\mathcal{R}$ 是保证结果集合

### 质量保证活动

**定义 2.4.12** 质量保证活动集合：
$$\mathcal{A} = \{Planning, Training, Auditing, Reviewing, Testing, Monitoring\}$$

### 质量保证实现

```rust
pub struct QualityAssurance {
    pub quality_plan: QualityPlan,
    pub assurance_activities: Vec<AssuranceActivity>,
    pub audit_schedule: Vec<AuditSchedule>,
    pub compliance_checklist: Vec<ComplianceItem>,
}

#[derive(Debug, Clone)]
pub struct AssuranceActivity {
    pub id: String,
    pub name: String,
    pub activity_type: AssuranceActivityType,
    pub frequency: ActivityFrequency,
    pub responsible: String,
    pub tools: Vec<String>,
    pub success_criteria: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum AssuranceActivityType {
    Planning,
    Training,
    Auditing,
    Reviewing,
    Testing,
    Monitoring,
}

#[derive(Debug, Clone)]
pub enum ActivityFrequency {
    Once,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Custom(u32),
}

#[derive(Debug, Clone)]
pub struct AuditSchedule {
    pub id: String,
    pub name: String,
    pub audit_type: AuditType,
    pub scheduled_date: u64,
    pub auditor: String,
    pub scope: Vec<String>,
    pub criteria: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum AuditType {
    Internal,
    External,
    Compliance,
    Process,
    Product,
}

#[derive(Debug, Clone)]
pub struct ComplianceItem {
    pub id: String,
    pub requirement: String,
    pub standard: String,
    pub mandatory: bool,
    pub verification_method: String,
    pub status: ComplianceStatus,
}

#[derive(Debug, Clone)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant,
    Partial,
    NotApplicable,
}

impl QualityAssurance {
    pub fn new(quality_plan: QualityPlan) -> Self {
        QualityAssurance {
            quality_plan,
            assurance_activities: Self::create_assurance_activities(),
            audit_schedule: Self::create_audit_schedule(),
            compliance_checklist: Self::create_compliance_checklist(),
        }
    }
    
    fn create_assurance_activities() -> Vec<AssuranceActivity> {
        vec![
            AssuranceActivity {
                id: "ACT-001".to_string(),
                name: "Quality Planning".to_string(),
                activity_type: AssuranceActivityType::Planning,
                frequency: ActivityFrequency::Once,
                responsible: "Project Manager".to_string(),
                tools: vec!["Quality Plan Template".to_string()],
                success_criteria: vec![
                    "Quality plan approved".to_string(),
                    "Quality objectives defined".to_string(),
                    "Quality metrics established".to_string(),
                ],
            },
            AssuranceActivity {
                id: "ACT-002".to_string(),
                name: "Team Training".to_string(),
                activity_type: AssuranceActivityType::Training,
                frequency: ActivityFrequency::Once,
                responsible: "HR Manager".to_string(),
                tools: vec!["Training Materials".to_string(), "Online Courses".to_string()],
                success_criteria: vec![
                    "Team trained on quality standards".to_string(),
                    "Quality awareness increased".to_string(),
                ],
            },
            AssuranceActivity {
                id: "ACT-003".to_string(),
                name: "Code Review".to_string(),
                activity_type: AssuranceActivityType::Reviewing,
                frequency: ActivityFrequency::Continuous,
                responsible: "Development Team".to_string(),
                tools: vec!["GitHub".to_string(), "SonarQube".to_string()],
                success_criteria: vec![
                    "All code reviewed".to_string(),
                    "Code quality standards met".to_string(),
                ],
            },
            AssuranceActivity {
                id: "ACT-004".to_string(),
                name: "Testing".to_string(),
                activity_type: AssuranceActivityType::Testing,
                frequency: ActivityFrequency::Continuous,
                responsible: "QA Team".to_string(),
                tools: vec!["JUnit".to_string(), "Selenium".to_string()],
                success_criteria: vec![
                    "All tests passed".to_string(),
                    "Coverage targets met".to_string(),
                ],
            },
        ]
    }
    
    fn create_audit_schedule() -> Vec<AuditSchedule> {
        vec![
            AuditSchedule {
                id: "AUDIT-001".to_string(),
                name: "Process Audit".to_string(),
                audit_type: AuditType::Process,
                scheduled_date: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() + 30 * 24 * 60 * 60, // 30 days from now
                auditor: "Internal Auditor".to_string(),
                scope: vec!["Development Process".to_string(), "Testing Process".to_string()],
                criteria: vec!["ISO 9001".to_string(), "CMMI".to_string()],
            },
            AuditSchedule {
                id: "AUDIT-002".to_string(),
                name: "Product Audit".to_string(),
                audit_type: AuditType::Product,
                scheduled_date: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() + 60 * 24 * 60 * 60, // 60 days from now
                auditor: "External Auditor".to_string(),
                scope: vec!["Software Quality".to_string(), "Documentation".to_string()],
                criteria: vec!["Functional Requirements".to_string(), "Performance Requirements".to_string()],
            },
        ]
    }
    
    fn create_compliance_checklist() -> Vec<ComplianceItem> {
        vec![
            ComplianceItem {
                id: "COMP-001".to_string(),
                requirement: "Quality management system in place".to_string(),
                standard: "ISO 9001:2015".to_string(),
                mandatory: true,
                verification_method: "Documentation Review".to_string(),
                status: ComplianceStatus::Compliant,
            },
            ComplianceItem {
                id: "COMP-002".to_string(),
                requirement: "Process improvement framework".to_string(),
                standard: "CMMI".to_string(),
                mandatory: true,
                verification_method: "Assessment".to_string(),
                status: ComplianceStatus::Partial,
            },
            ComplianceItem {
                id: "COMP-003".to_string(),
                requirement: "Code quality standards".to_string(),
                standard: "Internal Standards".to_string(),
                mandatory: true,
                verification_method: "Code Review".to_string(),
                status: ComplianceStatus::Compliant,
            },
        ]
    }
    
    pub fn execute_assurance_activity(&mut self, activity_id: &str) -> AssuranceResult {
        if let Some(activity) = self.assurance_activities.iter_mut()
            .find(|a| a.id == activity_id) {
            
            match activity.activity_type {
                AssuranceActivityType::Planning => self.execute_planning_activity(activity),
                AssuranceActivityType::Training => self.execute_training_activity(activity),
                AssuranceActivityType::Reviewing => self.execute_reviewing_activity(activity),
                AssuranceActivityType::Testing => self.execute_testing_activity(activity),
                _ => AssuranceResult::new_failed("Activity type not implemented".to_string()),
            }
        } else {
            AssuranceResult::new_failed("Activity not found".to_string())
        }
    }
    
    fn execute_planning_activity(&self, activity: &AssuranceActivity) -> AssuranceResult {
        // 执行质量规划活动
        let mut result = AssuranceResult::new();
        
        // 检查成功标准
        for criterion in &activity.success_criteria {
            if self.check_success_criterion(criterion) {
                result.successful_criteria.push(criterion.clone());
            } else {
                result.failed_criteria.push(criterion.clone());
            }
        }
        
        if result.failed_criteria.is_empty() {
            result.status = AssuranceStatus::Successful;
        } else {
            result.status = AssuranceStatus::Partial;
        }
        
        result
    }
    
    fn execute_training_activity(&self, activity: &AssuranceActivity) -> AssuranceResult {
        // 执行培训活动
        let mut result = AssuranceResult::new();
        
        // 模拟培训执行
        result.successful_criteria.push("Team trained on quality standards".to_string());
        result.successful_criteria.push("Quality awareness increased".to_string());
        result.status = AssuranceStatus::Successful;
        
        result
    }
    
    fn execute_reviewing_activity(&self, activity: &AssuranceActivity) -> AssuranceResult {
        // 执行代码审查活动
        let mut result = AssuranceResult::new();
        
        // 模拟代码审查
        if self.perform_code_review() {
            result.successful_criteria.push("All code reviewed".to_string());
            result.successful_criteria.push("Code quality standards met".to_string());
            result.status = AssuranceStatus::Successful;
        } else {
            result.failed_criteria.push("Code quality standards met".to_string());
            result.status = AssuranceStatus::Failed;
        }
        
        result
    }
    
    fn execute_testing_activity(&self, activity: &AssuranceActivity) -> AssuranceResult {
        // 执行测试活动
        let mut result = AssuranceResult::new();
        
        // 模拟测试执行
        if self.run_tests() {
            result.successful_criteria.push("All tests passed".to_string());
            result.successful_criteria.push("Coverage targets met".to_string());
            result.status = AssuranceStatus::Successful;
        } else {
            result.failed_criteria.push("All tests passed".to_string());
            result.status = AssuranceStatus::Failed;
        }
        
        result
    }
    
    fn check_success_criterion(&self, criterion: &str) -> bool {
        // 简化实现：根据标准名称检查
        match criterion {
            "Quality plan approved" => true,
            "Quality objectives defined" => true,
            "Quality metrics established" => true,
            _ => false,
        }
    }
    
    fn perform_code_review(&self) -> bool {
        // 简化实现：模拟代码审查
        true
    }
    
    fn run_tests(&self) -> bool {
        // 简化实现：模拟测试运行
        true
    }
}

#[derive(Debug, Clone)]
pub struct AssuranceResult {
    pub status: AssuranceStatus,
    pub successful_criteria: Vec<String>,
    pub failed_criteria: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum AssuranceStatus {
    Successful,
    Partial,
    Failed,
}

impl AssuranceResult {
    pub fn new() -> Self {
        AssuranceResult {
            status: AssuranceStatus::Successful,
            successful_criteria: Vec::new(),
            failed_criteria: Vec::new(),
            recommendations: Vec::new(),
        }
    }
    
    pub fn new_failed(message: String) -> Self {
        AssuranceResult {
            status: AssuranceStatus::Failed,
            successful_criteria: Vec::new(),
            failed_criteria: vec![message],
            recommendations: Vec::new(),
        }
    }
}
```

## 2.4.4 质量控制模型

### 质量控制函数

**定义 2.4.13** 质量控制函数 $QC: \mathcal{P} \times \mathcal{M} \rightarrow \mathcal{R}$，其中：

- $\mathcal{P}$ 是项目集合
- $\mathcal{M}$ 是度量指标集合
- $\mathcal{R}$ 是控制结果集合

### 统计过程控制

**定义 2.4.14** 控制图函数：
$$UCL = \mu + 3\sigma$$
$$LCL = \mu - 3\sigma$$

其中：

- $UCL$ 是上控制限
- $LCL$ 是下控制限
- $\mu$ 是过程均值
- $\sigma$ 是过程标准差

### 质量控制实现

```rust
pub struct QualityControl {
    pub control_charts: HashMap<String, ControlChart>,
    pub quality_metrics: Vec<QualityMetric>,
    pub control_limits: HashMap<String, ControlLimits>,
    pub process_capability: HashMap<String, ProcessCapability>,
}

#[derive(Debug, Clone)]
pub struct ControlChart {
    pub metric_id: String,
    pub data_points: Vec<DataPoint>,
    pub control_limits: ControlLimits,
    pub chart_type: ChartType,
    pub status: ChartStatus,
}

#[derive(Debug, Clone)]
pub struct DataPoint {
    pub timestamp: u64,
    pub value: f64,
    pub sample_size: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ControlLimits {
    pub upper_control_limit: f64,
    pub lower_control_limit: f64,
    pub center_line: f64,
    pub warning_limits: Option<(f64, f64)>,
}

#[derive(Debug, Clone)]
pub enum ChartType {
    XBar,      // 均值图
    R,         // 极差图
    S,         // 标准差图
    P,         // 不合格品率图
    C,         // 缺陷数图
    U,         // 单位缺陷数图
}

#[derive(Debug, Clone)]
pub enum ChartStatus {
    InControl,
    OutOfControl,
    Warning,
}

#[derive(Debug, Clone)]
pub struct ProcessCapability {
    pub cp: f64,    // 过程能力指数
    pub cpk: f64,   // 过程能力指数（考虑偏移）
    pub pp: f64,    // 过程性能指数
    pub ppk: f64,   // 过程性能指数（考虑偏移）
}

impl QualityControl {
    pub fn new() -> Self {
        QualityControl {
            control_charts: HashMap::new(),
            quality_metrics: Vec::new(),
            control_limits: HashMap::new(),
            process_capability: HashMap::new(),
        }
    }
    
    pub fn add_control_chart(&mut self, metric_id: String, chart_type: ChartType) {
        let control_limits = self.calculate_control_limits(&metric_id, &chart_type);
        
        let chart = ControlChart {
            metric_id: metric_id.clone(),
            data_points: Vec::new(),
            control_limits,
            chart_type,
            status: ChartStatus::InControl,
        };
        
        self.control_charts.insert(metric_id, chart);
    }
    
    pub fn add_data_point(&mut self, metric_id: &str, value: f64, sample_size: Option<u32>) {
        if let Some(chart) = self.control_charts.get_mut(metric_id) {
            let data_point = DataPoint {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                value,
                sample_size,
            };
            
            chart.data_points.push(data_point);
            
            // 更新控制图状态
            chart.status = self.evaluate_control_chart(chart);
        }
    }
    
    fn calculate_control_limits(&self, metric_id: &str, chart_type: &ChartType) -> ControlLimits {
        // 简化实现：使用默认控制限
        match chart_type {
            ChartType::XBar => ControlLimits {
                upper_control_limit: 100.0,
                lower_control_limit: 80.0,
                center_line: 90.0,
                warning_limits: Some((85.0, 95.0)),
            },
            ChartType::P => ControlLimits {
                upper_control_limit: 0.1,
                lower_control_limit: 0.0,
                center_line: 0.05,
                warning_limits: Some((0.02, 0.08)),
            },
            _ => ControlLimits {
                upper_control_limit: 100.0,
                lower_control_limit: 0.0,
                center_line: 50.0,
                warning_limits: None,
            },
        }
    }
    
    fn evaluate_control_chart(&self, chart: &ControlChart) -> ChartStatus {
        if chart.data_points.is_empty() {
            return ChartStatus::InControl;
        }
        
        let latest_value = chart.data_points.last().unwrap().value;
        let limits = &chart.control_limits;
        
        if latest_value > limits.upper_control_limit || latest_value < limits.lower_control_limit {
            ChartStatus::OutOfControl
        } else if let Some((lower_warning, upper_warning)) = limits.warning_limits {
            if latest_value > upper_warning || latest_value < lower_warning {
                ChartStatus::Warning
            } else {
                ChartStatus::InControl
            }
        } else {
            ChartStatus::InControl
        }
    }
    
    pub fn calculate_process_capability(&mut self, metric_id: &str, specification_limits: (f64, f64)) {
        if let Some(chart) = self.control_charts.get(metric_id) {
            let (usl, lsl) = specification_limits;
            let values: Vec<f64> = chart.data_points.iter().map(|dp| dp.value).collect();
            
            if values.len() >= 2 {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / (values.len() - 1) as f64;
                let std_dev = variance.sqrt();
                
                let cp = (usl - lsl) / (6.0 * std_dev);
                let cpu = (usl - mean) / (3.0 * std_dev);
                let cpl = (mean - lsl) / (3.0 * std_dev);
                let cpk = cpu.min(cpl);
                
                let pp = (usl - lsl) / (6.0 * std_dev);
                let ppu = (usl - mean) / (3.0 * std_dev);
                let ppl = (mean - lsl) / (3.0 * std_dev);
                let ppk = ppu.min(ppl);
                
                let capability = ProcessCapability {
                    cp,
                    cpk,
                    pp,
                    ppk,
                };
                
                self.process_capability.insert(metric_id.to_string(), capability);
            }
        }
    }
    
    pub fn get_control_report(&self, metric_id: &str) -> Option<ControlReport> {
        if let Some(chart) = self.control_charts.get(metric_id) {
            let capability = self.process_capability.get(metric_id).cloned();
            
            Some(ControlReport {
                metric_id: metric_id.to_string(),
                chart: chart.clone(),
                capability,
                recommendations: self.generate_recommendations(chart, capability.as_ref()),
            })
        } else {
            None
        }
    }
    
    fn generate_recommendations(&self, chart: &ControlChart, capability: Option<&ProcessCapability>) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match chart.status {
            ChartStatus::OutOfControl => {
                recommendations.push("Process is out of control. Investigate root cause.".to_string());
                recommendations.push("Implement immediate corrective actions.".to_string());
            },
            ChartStatus::Warning => {
                recommendations.push("Process showing warning signs. Monitor closely.".to_string());
                recommendations.push("Consider preventive actions.".to_string());
            },
            ChartStatus::InControl => {
                recommendations.push("Process is in control. Continue monitoring.".to_string());
            },
        }
        
        if let Some(cap) = capability {
            if cap.cpk < 1.0 {
                recommendations.push("Process capability is low. Consider process improvement.".to_string());
            } else if cap.cpk >= 1.33 {
                recommendations.push("Process capability is good. Consider cost reduction.".to_string());
            }
        }
        
        recommendations
    }
}

#[derive(Debug, Clone)]
pub struct ControlReport {
    pub metric_id: String,
    pub chart: ControlChart,
    pub capability: Option<ProcessCapability>,
    pub recommendations: Vec<String>,
}
```

## 2.4.5 相关链接

- [1.1 形式化基础理论](../01-foundations/README.md)
- [1.2 数学模型基础](../01-foundations/mathematical-models.md)
- [2.1 项目生命周期模型](./lifecycle-models.md)
- [2.2 资源管理模型](./resource-models.md)
- [2.3 风险管理模型](./risk-models.md)

## 参考文献

1. Juran, J. M., & Godfrey, A. B. (1999). Juran's quality handbook. McGraw-Hill.
2. Deming, W. E. (1986). Out of the crisis. MIT press.
3. Crosby, P. B. (1979). Quality is free: The art of making quality certain. McGraw-Hill.
4. Feigenbaum, A. V. (1991). Total quality control. McGraw-Hill.
