### Adaptive Batch Normalization (AdaBN)

**Algorithm 1: Adaptive Batch Normalization (AdaBN)**

**Input:** Target domain data $t$

**Output:** Normalized output $y_j(m)$ for each neuron $j$ and testing image $m$ in the target domain

**Procedure:**

1. **For each neuron $j$ in the Deep Neural Network (DNN):**
    *   Concatenate neuron responses on all images of the target domain $t$: $x_j = [..., x_j(m), ...]$
    *   Compute the mean and variance of the target domain:
        *   $\mu_j^t = \mathbb{E}(x_j^t)$
        *   $\sigma_j^t = \sqrt{\text{Var}(x_j^t)}$

2. **For each neuron $j$ in the DNN and each testing image $m$ in the target domain:**
    *   Compute the Batch Normalization (BN) output:
        *   $y_j(m) := \gamma_j \left( \frac{x_j(m) - \mu_j^t}{\sigma_j} \right) + \beta_j$

**Note:** ¹In practice, we adopt an online algorithm (Donald, 1999) to efficiently estimate the mean and variance.

#### Intuition and Method Description

The core idea behind AdaBN is straightforward: standardizing each layer by domain ensures that each layer receives data from a similar distribution, regardless of whether it originates from the source or the target domain.

For $K$-domain adaptation where $K > 2$, we standardize each sample using the statistics of its own domain. During training, these statistics are calculated for every mini-batch, with the only constraint being that all samples within a mini-batch must belong to the same domain.

In (semi-)supervised domain adaptation, labeled data can be used to fine-tune the weights. Consequently, our method can be adapted to various domain adaptation settings with minimal effort.

#### 3.3 Further Thoughts About AdaBN

The simplicity of AdaBN stands in stark contrast to the complexity of the domain shift problem. A natural question arises: can such simple translation and scaling operations effectively approximate the inherently non-linear domain transfer function?

Consider a simple neural network with input $x \in \mathbb{R}^{p_1 \times 1}$. It comprises one BN layer with mean $\mu_i$ and variance $\sigma_i^2$ for each feature ($i \in \{1...p_2\}$), a fully connected layer with weight matrix $W \in \mathbb{R}^{p_2 \times p_1}$ and bias $b \in \mathbb{R}^{p_2 \times 1}$, and a non-linear transformation layer $f(\cdot)$, where $p_1$ and $p_2$ represent the input and output feature sizes, respectively. The output of this network is $f(W_a x + b_a)$, where:

$W_a = W^T \Sigma^{-1}$, $b_a = -W^T \Sigma^{-1} \mu + b$,
$\Sigma = \text{diag}(\sigma_1, ..., \sigma_{p_1})$, $\mu = (\mu_1, ..., \mu_{p_1})$.

Without BN, the output would simply be $f(W^T x + b)$. This demonstrates that the transformation is highly non-linear even for a simple network with a single computation layer. As the CNN architecture deepens, its capacity to represent more complex transformations increases.

Another question is why we transform neuron responses independently instead of decorrelating and then re-correlating them, as suggested by Sun et al. (2016). While decorrelation can enhance performance under certain conditions, in CNNs, the mini-batch size is often smaller than the feature dimension, leading to singular covariance matrices that are difficult to invert. Consequently, the covariance matrix is frequently singular. Moreover, decorrelation necessitates computing the inverse of the covariance matrix, which is computationally intensive, especially when applying AdaBN to all layers of the network.