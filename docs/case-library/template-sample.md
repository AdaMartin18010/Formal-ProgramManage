# 案例：敏捷产品迭代交付（样板）

## 1. 背景 (Context)

| 项目 | 内容 |
|------|------|
| **组织** | 某互联网公司产品线，约 80 人，敏捷文化 |
| **项目目标** | 在 6 个月内交付新移动端产品 V1，支持核心业务流程 |
| **初始约束** | 预算固定、核心团队 12 人、需与现有后端 API 兼容 |
| **关键干系人** | 产品负责人、技术负责人、业务方、运维 |

## 2. 过程 (Process)

- **Month 1–2**：启动与规划；采用 SAFe 式 PI 规划，确定 3 个迭代为一 PI；关键决策：优先 MVP 范围、引入自动化测试与 CI。
- **Month 3–4**：执行；遇到后端 API 延期，通过缓冲故事与临时 Mock 应对；每周评审与风险同步。
- **Month 5–6**：收尾与发布；灰度发布、监控与回滚预案；正式发布后做回顾与经验教训归档。

## 3. 理论映射 (Theory Mapping)

| 本项目文档/标准 | 应用点 |
|------------------|--------|
| [2.1 项目生命周期模型](../02-project-management/lifecycle-models.md) | 敏捷迭代生命周期、阶段转换（Sprint 边界）、PMBOK 8 绩效域 Scope/Schedule |
| [4.1 敏捷开发模型](../04-industry-applications/software-development/agile-models.md) | Sprint、Backlog、评审与回顾 |
| [2.3 风险管理模型](../02-project-management/risk-models.md) | API 延期风险识别、应对（缓冲与 Mock）、监控 |
| [2.4 质量管理模型](../02-project-management/quality-models.md) | 自动化测试、CI、发布质量标准 |
| PMBOK 8th / ISO 21502:2020 | 适应性生命周期、增量交付；见 [PMBOK_8_ALIGNMENT_PLAN.md](../PMBOK_8_ALIGNMENT_PLAN.md) |

## 4. 关键学习点 (Key Learnings)

- **成功因素**：PI 规划统一目标；自动化与 CI 降低回归风险；每周风险同步及时暴露依赖问题。
- **教训**：对外部 API 依赖应更早识别并设缓冲或契约测试。
- **可复用模式**：MVP 范围锁定 + 迭代交付 + 灰度发布 + 回顾。

## 5. 练习 (Exercises)

1. **分析**：若你是项目经理，在 Month 3 得知 API 延期 2 周，除 Mock 外还会采取哪些应对？如何映射到 [2.3 风险应对模型](../02-project-management/risk-models.md)？
2. **检索**：本案例涉及哪些 [03-retrieval-practice-questions.md](../12-learning-support/03-retrieval-practice-questions.md) 中与生命周期、风险相关的题目？试作答并对照。

## 6. 资源 (Resources)

- [LEARNING_PATHS.md](../LEARNING_PATHS.md) 轨道 A/B
- [02-project-management/README.md](../02-project-management/README.md)
- [04-industry-applications/software-development/agile-models.md](../04-industry-applications/software-development/agile-models.md)

---

**Last Updated**: 2026-02-04
**Status**: 样板案例，供复制扩展
