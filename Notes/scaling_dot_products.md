<!-- $softmax(\frac{Q*K^T}{\sqrt{d}})$ 中，当d越小，结果会越趋近one-hot。

## 证明：

$\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \rightarrow softmax = \frac{e^{x_i}}{e^{x_1}+e^{x_2}} \Rightarrow \begin{bmatrix} \frac{e^{x_1}}{e^{x_1}+e^{x_2}} \\ \frac{e^{x_2}}{e^{x_1}+e^{x_2}} \end{bmatrix}$

$\frac{e^{x_1}}{e^{x_2}} = a \Rightarrow \ln a = \ln(\frac{e^{x_1}}{e^{x_2}}) = x_1-x_2$

$\Rightarrow a = e^{x_1-x_2}$

如果 $\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \cdot k = \begin{bmatrix} kx_1 \\ kx_2 \end{bmatrix} , k>0$

那么 $a' = e^{k(x_1-x_2)} = a^k$

当 $a>0$ 时，如果 $k>1$：

$\begin{cases} 
a' > a, & k>1 \\
a' < a, & 0<k<1
\end{cases}$

when k>1，$a'= a^k$, $k$ 越大， a'越大，softmax 的输出就越接近one-hot 向量。
when 0<k<1, $a'= a^k \Rightarrow 1<a'<a $, $k$ 越大，$softmax$ 的输出就越接近one-hot 向量,也就是 d 越小，$softmax$ 的输出就越接近one-hot 向量.

进一步推理 $softmax(\frac{Q*K^T}{\sqrt{d}})$

\[
  \begin{equation}
\begin{bmatrix} x_1 \\ x_2 \\ ...\\x_d \end{bmatrix} \rightarrow softmax \rightarrow \begin{bmatrix} \frac{e^{x_1}}{e^{x_1}+e^{x_2}+...+e^{x_d}} \\ \frac{e^{x_2}}{e^{x_1}+e^{x_2}+...+e^{x_d}} \\...\\ \frac{e^{x_d}}{e^{x_1}+e^{x_2}+...+e^{x_d}} \end{bmatrix}\end{equation}
  \]

相当于上面的公式左侧每一项都乘以$\frac{1}{\sqrt{d}}$, 此时$a'= a^k=a^{\frac{1}{\sqrt{d}}}$ -->



# Proof1: Smaller d in $softmax(\frac{Q*K^T}{\sqrt{d}})$ leads to more one-hot-like outputs

## 1. Basic Softmax Properties

Let's start with a simple 2D case. For a vector $\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$, the softmax function gives:

$softmax(\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}) = \begin{bmatrix} \frac{e^{x_1}}{e^{x_1}+e^{x_2}} \\ \frac{e^{x_2}}{e^{x_1}+e^{x_2}} \end{bmatrix}$

## 2. Ratio Analysis

Let's define $a$ as the ratio of the exponentials:
- $a = \frac{e^{x_1}}{e^{x_2}} = e^{x_1-x_2}$

## 3. Scaling Effect

When we scale the input vector by a positive constant $k$:
- $\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \cdot k = \begin{bmatrix} kx_1 \\ kx_2 \end{bmatrix}$
- The new ratio becomes: $a' = e^{k(x_1-x_2)} = a^k$

## 4. Key Properties

For $a > 0$:
1. When $k > 1$: $a' = a^k > a$
2. When $0 < k < 1$: $a' = a^k < a$

The larger $k$ is, the more extreme the ratio becomes, pushing the softmax output closer to a one-hot vector.

## 5. Application to Attention Mechanism

In the attention mechanism formula $softmax(\frac{Q*K^T}{\sqrt{d}})$:

The general form for d-dimensional case is:
$softmax(\vec{x}) = \begin{bmatrix} \frac{e^{x_1}}{\sum_{i=1}^d e^{x_i}} \\ \frac{e^{x_2}}{\sum_{i=1}^d e^{x_i}} \\ \vdots \\ \frac{e^{x_d}}{\sum_{i=1}^d e^{x_i}} \end{bmatrix}$

When we scale by $\frac{1}{\sqrt{d}}$, it's equivalent to setting $k = \frac{1}{\sqrt{d}}$

## 6. Conclusion

As $d$ decreases:
1. $\frac{1}{\sqrt{d}}$ increases
2. This makes the ratios more extreme ($a' = a^{\frac{1}{\sqrt{d}}}$ becomes larger)
3. Therefore, the softmax output becomes closer to a one-hot vector



------------------------------
# Proof2 当 $S$ 趋近于 one-hot 时 $\frac{dS}{dZ} $ 趋近于 0， 会引起梯度消失
## Softmax 函数求导
已知 softmax 的输入和输出分别是：$Z=(z1,z2,z3,z4)$,$S=(s1,s2,s3,s4)$

两者的关系是：$S = \text{softmax}(Z)$， 其中  $s_i = \frac{e^{z_i}}{\sum_{k=1}^n e^{z_k}}, \quad \forall i = 1, \ldots, n$
先给出 $S$ 对 $Z$ 偏导数的结论：

\[
  \begin{equation}
\frac{dS}{dZ} =J_{softmax} = 
\begin{pmatrix}
s_1 \cdot (1 - s_1) & -s_1 \cdot s_2 & -s_1 \cdot s_3 & -s_1 \cdot s_4 \\
-s_2 \cdot s_1 & s_2 \cdot (1 - s_2) & -s_2 \cdot s_3 & -s_2 \cdot s_4 \\
-s_3 \cdot s_1 & -s_3 \cdot s_2 & s_3 \cdot (1 - s_3) & -s_3 \cdot s_4 \\
-s_4 \cdot s_1 & -s_4 \cdot s_2 & -s_4 \cdot s_3 & s_4 \cdot (1 - s_4)
\end{pmatrix}
\end{equation}
  \]



## 接下来是推导过程：
因为
$\frac{\partial}{\partial z_j} \log(s_i) = \frac{1}{s_i} \cdot \frac{\partial s_i}{\partial z_j}$
所以：
\[\frac{\partial s_i}{\partial z_j} = s_i \cdot \frac{\partial}{\partial z_j} \log(s_i)\] 

计算 $log s_i$
$
\log s_i = \log \left(\frac{e^{z_i}}{\sum_{l=1}^n e^{z_l}}\right) = z_i - \log \left(\sum_{l=1}^n e^{z_l}\right)$

那么 $log s_i$ 对 $z_j$ 的偏微分是：

\[
\frac{\partial}{\partial z_j} \log s_i = \frac{\partial z_i}{\partial z_j} - \frac{\partial}{\partial z_j} \log \left(\sum_{l=1}^n e^{z_l}\right) \]

上式只看左半边我们可以得到以下结论：

$\frac{\partial z_i}{\partial z_j} = 
\begin{cases} 
1, & \text{if } i = j \\
0, & \text{otherwise}
\end{cases}$

接下来计算 $\frac{\partial}{\partial z_j} \log \left(\sum_{l=1}^n e^{z_l}\right)$ 

根据符合函数的求导法则可得： $\frac{\partial}{\partial z_j} \log \left(\sum_{l=1}^n e^{z_l}\right) =
\frac{1}{\sum_{l=1}^n e^{z_l}} \cdot \left(\frac{\partial}{\partial z_j} \sum_{l=1}^n e^{z_l}\right) = s_l$

计算依据如下：$
\frac{\partial}{\partial z_j} \sum_{l=1}^n e^{z_l} = \frac{\partial}{\partial z_j} [e^{z_1} + e^{z_2} + \cdots + e^{z_j} + \cdots + e^{z_n}] = \frac{\partial}{\partial z_j} [e^{z_j}] = e^{z_j}
$

所以 
\[
\frac{\partial s_i}{\partial z_j} = s_i \cdot \frac{\partial}{\partial z_j} \log(s_i) = s_i \cdot (1\{i = j\} - s_j)\]

那么可以得到 
\[\frac{dS}{dZ} = J_{softmax} = 
\begin{pmatrix}
\frac{\partial s_1}{\partial z_1} & \frac{\partial s_1}{\partial z_2} & \cdots & \frac{\partial s_1}{\partial z_n} \\
\frac{\partial s_2}{\partial z_1} & \frac{\partial s_2}{\partial z_2} & \cdots & \frac{\partial s_2}{\partial z_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial s_n}{\partial z_1} & \frac{\partial s_n}{\partial z_2} & \cdots & \frac{\partial s_n}{\partial z_n}
\end{pmatrix}\]

\[
  \begin{equation}
J_{softmax} = 
\begin{pmatrix}
s_1 \cdot (1 - s_1) & -s_1 \cdot s_2 & -s_1 \cdot s_3 & -s_1 \cdot s_4 \\
-s_2 \cdot s_1 & s_2 \cdot (1 - s_2) & -s_2 \cdot s_3 & -s_2 \cdot s_4 \\
-s_3 \cdot s_1 & -s_3 \cdot s_2 & s_3 \cdot (1 - s_3) & -s_3 \cdot s_4 \\
-s_4 \cdot s_1 & -s_4 \cdot s_2 & -s_4 \cdot s_3 & s_4 \cdot (1 - s_4)
\end{pmatrix}
\end{equation}
  \]
