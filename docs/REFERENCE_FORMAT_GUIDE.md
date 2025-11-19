# 参考文献格式规范指南

## 概述

本文档提供Formal-ProgramManage项目的参考文献格式规范，确保所有文档的参考文献格式统一、规范，符合学术标准。

## 参考文献格式规范

### 1. 学术论文格式

**格式：**

```
作者姓氏, 名字首字母. (年份). 论文标题. 期刊名称, 卷号(期号), 页码.
```

**示例：**

```
Clarke, E. M., Grumberg, O., & Peled, D. A. (1999). Model checking. MIT press.
Hoare, C. A. R. (1969). An axiomatic basis for computer programming. Communications of the ACM, 12(10), 576-580.
```

### 2. 书籍格式

**格式：**

```
作者姓氏, 名字首字母. (年份). 书名. 出版社.
```

**示例：**

```
Baier, C., & Katoen, J. P. (2008). Principles of model checking. MIT press.
Leffingwell, D. (2020). SAFe 6.0 Distilled: Achieving Business Agility with the Scaled Agile Framework. Addison-Wesley.
```

### 3. 标准文档格式

**格式：**

```
标准编号:年份. 标准名称. 发布组织.
```

**示例：**

```
ISO 21500:2012. Guidance on project management. International Organization for Standardization.
ISO/IEC 25010:2011. Systems and software engineering - Systems and software Quality Requirements and Evaluation (SQuaRE) - System and software quality models.
IEEE Std 830-1998. IEEE recommended practice for software requirements specifications.
```

### 4. 会议论文格式

**格式：**

```
作者姓氏, 名字首字母. (年份). 论文标题. In 会议名称 (pp. 页码). 出版社.
```

**示例：**

```
Cousot, P., & Cousot, R. (1977). Abstract interpretation: a unified lattice model for static analysis of programs by construction or approximation of fixpoints. In Proceedings of the 4th ACM SIGACT-SIGPLAN symposium on Principles of programming languages (pp. 238-252).
```

### 5. 在线资源格式

**格式：**

```
作者或组织. (年份). 资源标题. 网站名称. URL (访问日期).
```

**示例：**

```
Schwaber, K., & Sutherland, J. (2020). The Scrum Guide. Scrum.org. https://scrumguides.org/scrum-guide.html (访问日期: 2025-01-XX).
```

## 参考文献列表格式

### 标准格式

所有文档应在末尾包含"参考文献"部分，格式如下：

```markdown
## 参考文献

1. Author, A. (Year). Title. Journal, Volume(Issue), Pages.
2. ISO 21500:2012. Title. Organization.
3. Author, A. (Year). Title. Publisher.
```

### 编号规则

- 使用阿拉伯数字编号：1, 2, 3, ...
- 每个参考文献占一行
- 编号后加句点，然后是内容
- 多个作者用逗号分隔，最后一个作者前用 "&"

### 排序规则

1. **按引用顺序排序**（推荐）：按照在文档中出现的顺序编号
2. **按字母顺序排序**：按第一作者姓氏字母顺序排列
3. **按类别排序**：先标准文档，再学术论文，再书籍

## 常见格式问题

### 问题1：作者姓名格式不一致

**错误：**

```
Clarke, E.M., Grumberg, O., & Peled, D.A. (1999).
```

**正确：**

```
Clarke, E. M., Grumberg, O., & Peled, D. A. (1999).
```

### 问题2：年份位置错误

**错误：**

```
Clarke, E. M., Grumberg, O., & Peled, D. A. Model checking. (1999). MIT press.
```

**正确：**

```
Clarke, E. M., Grumberg, O., & Peled, D. A. (1999). Model checking. MIT press.
```

### 问题3：标点符号不一致

**错误：**

```
ISO 21500:2012 Guidance on project management International Organization for Standardization
```

**正确：**

```
ISO 21500:2012. Guidance on project management. International Organization for Standardization.
```

## 参考文献检查清单

### 格式检查

- [ ] 作者姓名格式正确（姓氏在前，名字首字母在后）
- [ ] 年份位置正确（紧跟在作者后）
- [ ] 标点符号正确（句点、逗号、冒号）
- [ ] 期刊信息完整（卷号、期号、页码）
- [ ] 出版社信息完整

### 内容检查

- [ ] 所有引用都有对应的参考文献
- [ ] 所有参考文献都在文档中被引用
- [ ] 参考文献信息准确（作者、年份、标题等）
- [ ] 标准文档编号正确
- [ ] URL链接有效（如果适用）

### 一致性检查

- [ ] 所有文档使用相同的格式规范
- [ ] 编号方式一致
- [ ] 排序方式一致
- [ ] 标点符号使用一致

## 参考文献模板

### 学术论文模板

```markdown
## 参考文献

1. Author1, A. A., & Author2, B. B. (Year). Title of the paper. Journal Name, Volume(Issue), Pages.
2. Author, A. (Year). Title. Conference Name, Pages.
```

### 书籍模板

```markdown
## 参考文献

1. Author, A. (Year). Book Title. Publisher.
2. Author, A., & Author, B. (Year). Book Title (Edition). Publisher.
```

### 标准文档模板

```markdown
## 参考文献

1. ISO 21500:2012. Guidance on project management. International Organization for Standardization.
2. ISO/IEC 25010:2011. Systems and software engineering - Systems and software Quality Requirements and Evaluation (SQuaRE) - System and software quality models.
3. IEEE Std 830-1998. IEEE recommended practice for software requirements specifications.
```

## 自动化检查建议

### 检查脚本功能

1. **格式验证**
   - 检查作者姓名格式
   - 检查年份位置
   - 检查标点符号

2. **完整性验证**
   - 检查是否有未引用的参考文献
   - 检查是否有未列出的引用

3. **一致性验证**
   - 检查格式一致性
   - 检查编号连续性

## 参考文献示例（完整版）

```markdown
## 参考文献

1. Clarke, E. M., Grumberg, O., & Peled, D. A. (1999). Model checking. MIT press.
2. Baier, C., & Katoen, J. P. (2008). Principles of model checking. MIT press.
3. Hoare, C. A. R. (1969). An axiomatic basis for computer programming. Communications of the ACM, 12(10), 576-580.
4. Cousot, P., & Cousot, R. (1977). Abstract interpretation: a unified lattice model for static analysis of programs by construction or approximation of fixpoints. In Proceedings of the 4th ACM SIGACT-SIGPLAN symposium on Principles of programming languages (pp. 238-252).
5. ISO 21500:2012. Guidance on project management. International Organization for Standardization.
6. Project Management Institute. (2021). A guide to the project management body of knowledge (PMBOK guide) (7th ed.).
7. ISO/IEC 25010:2011. Systems and software engineering - Systems and software Quality Requirements and Evaluation (SQuaRE) - System and software quality models.
8. IEEE Std 830-1998. IEEE recommended practice for software requirements specifications.
9. Schwaber, K., & Sutherland, J. (2020). The Scrum Guide. Scrum.org.
10. Leffingwell, D. (2020). SAFe 6.0 Distilled: Achieving Business Agility with the Scaled Agile Framework. Addison-Wesley Professional.
```

## 更新日志

- **2025-01-XX**: 创建参考文献格式规范指南
- **2025-01-XX**: 添加格式示例和检查清单

---

**维护者**: Formal-ProgramManage团队
**最后更新**: 2025-01-XX
