
当 $S$ 趋近于 one-hot 时 $\frac{dS}{dZ} $ 趋近于 0， 会引起梯度消失
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
