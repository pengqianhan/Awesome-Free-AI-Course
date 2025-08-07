# 比例风险模型

## 生存分析概述

**作者**: 韩朋谦
**日期**: 2025年3月25日
**来源**: [Cox比例风险模型](https://en.wikipedia.org/wiki/Proportional_hazards_model)

---

## 幻灯片 1: 生存模型基础

- **目标**: 将事件发生时间（例如死亡、故障）与协变量关联起来。
- **风险率**: 在时间 $t$ 存活的情况下，事件在 $[t, t+dt)$ 内发生的概率。
- **组成部分**:
  - 基准风险: $\lambda_0(t)$（随时间变化的风险）。
  - 协变量效应: 风险的乘性缩放。

**风险公式**:
$\lambda(t|X) = \lambda_0(t) \cdot \text{协变量效应}$

---

## 幻灯片 2: 比例风险核心理念

- **假设**: 协变量对风险的缩放呈乘性作用。
- 假设我们只有一个协变量 $x$，因此只有一个系数 $\beta_1$。模型如下:

\[
\lambda(t|x) = \lambda_0(t) \exp(\beta_1 x)
\]

- 考虑 $x$ 增加 1 的效应:

\[
\begin{aligned}
\lambda(t|x+1) &= \lambda_0(t) \exp(\beta_1 (x+1)) \\
&= \lambda_0(t) \exp(\beta_1 x + \beta_1) \\
&= (\lambda_0(t) \exp(\beta_1 x)) \exp(\beta_1) \\
&= \lambda(t|x) \exp(\beta_1)
\end{aligned}
\]

- **比例**:
  $\frac{\lambda(t|x+1)}{\lambda(t|x)} = \exp(\beta_1)$（随时间恒定）

---

## 幻灯片 3: Cox比例风险模型

- **提出者**: 大卫·考克斯爵士 (1972)。
- **形式**: 对于协变量 $X_i = (X_{i1}, \dots, X_{ip})$:$\lambda(t|X_i) = \lambda_0(t) \exp(\beta_1 X_{i1} + \cdots + \beta_p X_{ip}) = \lambda_0(t) \exp(X_i \cdot \beta)$
- **关键特性**: 无需指定 $\lambda_0(t)$ 即可估计 $\beta$。
- **风险比**:
  $\frac{\lambda(t|X_i)}{\lambda(t|X_j)} = \exp((X_i - X_j) \cdot \beta)$。

---

## 幻灯片 4: 示例：二元协变量（医院）

- **数据**: 手术后生存情况（医院 A 与 B）。
- 模型: $\lambda(t|X_i) = \lambda_0(t) \exp(\beta_1 X_i)$, $X_i = 1$ (A), $0$ (B)。
- 估计: $\beta_1 = 2.12$。
- 风险比: $\exp(2.12) = 8.32$。
- **解释**: 医院 A 的风险是医院 B 的 8.3 倍。

---

## 幻灯片 5: 无截距项

- 典型回归: 包含截距 $\beta_0$。
- Cox模型: $\lambda_0(t)$ 吸收截距。
- 如果加入 $\beta_0$:$\lambda(t|X_i) = \lambda_0(t) \exp(X_i \cdot \beta + \beta_0) = [\exp(\beta_0) \lambda_0(t)] \exp(X_i \cdot \beta)$
- 重定义 $\lambda_0^*(t) = \exp(\beta_0) \lambda_0(t)$ $\Rightarrow$ 不可识别。

---

## 幻灯片 6: 扩展与变体

- **时变协变量**: $\lambda(t|X_i(t))$。
- **时变系数**: $\beta(t)$。
- **参数化模型**: 指定 $\lambda_0(t)$（例如，Weibull分布）。
- **LASSO**: 高维协变量选择。

---

## 幻灯片 7: 软件实现

- R: `coxph()`（survival包）。
- Python: `CoxPHFitter`（lifelines包）。
- SAS: `phreg`。
- Stata: `stcox`。
- MATLAB: `fitcox`。

---

## 总结

- **Cox模型**: 灵活的半参数化生存分析方法。
- **优势**: 通过风险比聚焦于协变量效应。
- **局限性**: 假设比例风险。
- **应用领域**: 医学、经济学、工程学。

---
