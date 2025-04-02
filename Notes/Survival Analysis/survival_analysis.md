# 基础概念

### 生存分析 survival\_analysis

The probability density function (PDF) 概率密度函数
$PDF:f(t)$
cumulative distribution function (CDF) 累计分布函数

*   含义: 从负无穷到t时刻的死亡概率 [来自](https://zhuanlan.zhihu.com/p/110764631)
    $CDF: F(t) = P(T ≤ t)  ,{ F(0) = P(T = 0) } $
    Survivor function (or [Failure rate](https://en.wikipedia.org/wiki/Failure_rate)) 生存函数

    举例：F(5) 意思是到第5年，这个人得病的概率；

*   含义: 生存超过某个时间的概率，即关注事件在时间或以内都未发生的概率
    $S(t) ≝ 1 - F(t) = P(T > t) for t > 0$
    风险函数(Hazard function)：

*   含义：风险函数关注事件在时刻发生的概率, 也被称为在t时刻 (瞬间) 发生的风险

*   从定义里面可以看出“风险率”描述的是已经存活了t时间的事物在当前时间点t的瞬时死亡概率密度，并不是概率，所以它可以大于1。风险率可以看成是一个关于时间的函数，所以风险率也叫风险函数。如果选取一小段区间，则该区间的累积死亡概率近似等于 风险率\*区间，意味着风险率越大死亡越多，风险率越小死亡越少，所以通过风险率就可以预估在时间t物体将要死亡的可能性，风险率越大死亡的可能性越大，风险率越小死亡的可能性越小。这就是风险率函数的一个实际意义。来自[知乎](https://www.zhihu.com/question/297553384/answer/1016449898)

$$
P(t,s) = \Pr(t < T < s | T \geq t) \\
h(t) = \lim_{s \to t} \frac{P(t,s)}{s-t} 
$$

结果为：

$$
h(t) = \lim_{\Delta t \to 0} \frac{\Pr[t \leq T < t + \Delta t | T \geq t]}{\Delta t} \\
= \lim_{s \to t} \frac{\Pr(t,s)}{s-t} \\
= \frac{f(t)}{S(t)}
$$

Cumulative hazard function 累积风险函数

*   含义: 关注事件到t时刻为止发生的概率，相比较风险函数更容易被精确估计

$$
H(t) = \int_0^t h(s)ds \\
= -ln[1-F(t)] = - \ln S(t)
$$

Note that

$$
S(t) = e^{-H(t)}
$$

$$
f(t) = h(t)e^{-H(t)} 
$$

Note 1: Note that h(t)dt \= f(t)dt/S(t) \~ Pr\[fail in  \[t,t+dt) |survive until t]. Thus, the hazard function might be of more intrinsic interest than the PDF to a patient who had survived a certain time period and wanted to know something about their prognosis. cite (BIO 244: Unit 1
Survival Distributions, Hazard Functions, Cumulative
Hazards)

For example:Exponential Distribution: denoted $T \sim Exp(\lambda)$. For $t > 0$,

$f(t) = \lambda e^{-\lambda t}$ for $\lambda > 0$ (scale parameter)

$F(t) = 1 - e^{-\lambda t}$ $S(t) = e^{-\lambda t}$

$h(t) = \lambda$ $\implies$ constant hazard function

$H(t) = \lambda t$

可以看出指数分布中hazard rate是个常数浪哒，即某事件在单位时间内发生的平均次数，也可以大于1。说明在指数分布中，物体死亡概率不变，旧的物体和新的物体一样容易死亡，可以refer to memoryless property, 但这在现实世界是不合理的.来自[知乎](https://www.zhihu.com/question/297553384/answer/1634759210)

## censored data 删失数据

删失数据介绍 from [知乎](https://zhuanlan.zhihu.com/p/497968260)

## [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution) 威布尔分布

$$
\begin{cases}
    \frac{k}{\lambda} \left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}, & x \geq 0, \\
    0, & x < 0,
\end{cases}$$
where k > 0 is the shape parameter and λ > 0 is the scale parameter of the distribution.

死亡函数:


$$
CDF(x; k, \lambda) = F(x,k, \lambda) =
\begin{cases}
1 - e^{-(x/\lambda)^k}, & x \geq 0, \\
0, & x < 0.
\end{cases}
$$


生存函数:
$$S(x) ≝ 1 - F(x) = e^{-(x/\lambda)^k} $$

风险函数(Hazard function or failure rate):


$$
h(x; k, \lambda) = \frac{f(x)}{S(x)} = \frac{k}{\lambda} \left( \frac{x}{\lambda} \right)^{k-1}
$$


[If the quantity X is a "time-to-failure", the Weibull distribution gives a distribution for which the failure rate is proportional to a power of time. The shape parameter, k, is that power plus one, and so this parameter can be interpreted directly as follows](https://en.wikipedia.org/wiki/Weibull_distribution)
- A value of k<1 indicates that the failure rate decreases over time
- A value of k=1 indicates that the failure rate is constant over time. The Weibull distribution reduces to an exponential distribution
- A value of k>1 indicates that the failure rate increases with time. This happens if there is an "aging" process, or parts that are more likely to fail as time goes on. In the context of the diffusion of innovations, this means positive word of mouth: the hazard function is a monotonically increasing function of the proportion of adopters.

## Mixture of Weibull distribution
The mixture distribution is a weighted summation of K distributions $\{g_1(x;\Theta_1),\ldots,g_K(x;\Theta_K)\}$ where the weights $\{w_1,\ldots,w_K\}$ sum to one. As is obvious, every distribution in the mixture has its own parameter $\Theta_k$. The mixture distribution is formulated as:

$$f(x;\Theta_1,\ldots,\Theta_K) = \sum_{k=1}^K w_k g_k(x;\Theta_k),$$
$$F(x;\Theta_1,\ldots,\Theta_K) = \sum_{k=1}^K w_k G_k(x;\Theta_k),$$
$$\text{subject to }\sum_{k=1}^K w_k = 1.$$
from [mixture distribution](https://blog.csdn.net/tanghonghanhaoli/article/details/90543917) and 
Fitting A Mixture Distribution to Data: Tutorial

## Maximum Likelihood Estimation (MLE) 极大似然估计
[ChatGPT 解释](https://chatgpt.com/share/6715cf41-c500-8003-ad31-998e39b33ee8)
The MLE aims to find parameter $\Theta$ which maximizes the likelihood:

$$\hat{\Theta} = \arg\max_{\Theta} L(\Theta).$$


According to the definition, the likelihood can be written as:

$$L(\Theta|x_1,\ldots,x_n) := f(x_1,\ldots,x_n;\Theta) \\ 
                            =\prod_{i=1}^n f(x_i;\Theta)
$$$

where the $x_1,\ldots,x_n$ are  identically distributed. Note that in literature, the $L(\Theta|x_1,\ldots,x_n)$ is also denoted by $L(\Theta)$ for simplicity.

Usually, for more convenience, we use log-likelihood rather than likelihood:

$\ell(\Theta) := \log L(\Theta)$
$= \log\prod_{i=1}^n f(x_i,\Theta) = \sum_{i=1}^n \log f(x_i,\Theta).$

from Fitting A Mixture Distribution to Data: Tutorial
