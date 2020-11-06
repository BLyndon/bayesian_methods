---
layout: page
title: 2. Latent Variable Models
permalink: /lvmod/
---
* ToC
{:toc}

---

The concept of latent or hidden variables plays a central role in generative models, i.e. models based on a full probability distribution over all variables. The observable features in these models may have sophisticated correlations between each other resulting in complex probability distributions. By introducing latent variables to the model, the complexity can be mapped to simpler and more tractable joint distributions of the expanded space. The latent variables can be introduced by marginalization

$$
    p(x) = \sum_z p(x, z)
$$

The complex distribution over the observables $$p(x)$$ then is constructed by simpler additive components $$p(x, z)$$. In case of continuous latent variables, the sum is replaced by an integral. Continuous latent variables are closely related to manifold lerning, where the all the data lies close to a manifold of much lower dimensionality. For example a translation of pixels of an image can be described by a latent variable.

### Examples
{: .no_toc}
+ clustering
+ topic modeling
+ blind source separation
+ dimensional reduction

The first two examples are discrete latent variable models. The remaining examples are continuous latent variable models.

### Interpretation
{: .no_toc}
In many cases, the latent variables can be interpreted. In case of clustering, the value of the latent variable corresponds to the cluster component. In case of blind source separation, where mixed signals from several sources are measured, each latent variable corresponds to a single source.

### Training
{: .no_toc}
The latent variables are completely unobservable and must be inferred from the observed data. An important training method for latent variable models is the expectation maximation algorithm.

---

## General Expectation Maximation Algorithm

During the training of a model, the parameter $$\theta$$ is tuned to maximize the likelihood $$p(X\|\theta)$$ of the observed dataset.

For i.i.d. data points the loglikelihood factorizes. Using the chain rule, we introduce a latent variables $$t_i$$

$$
    \log p(X|\theta) = \sum_i log p(x_i|\theta) = \sum_{i} \log \sum_{c}p(x_i, t_i=c| \theta)
$$

The advantage of the EM algorithm is to maximize a lower bound instead of the complicated loglikelihood $$p(X\|\theta)$$. To find a lower bound we make use of the [Jenson inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality). But first, we need to transform the argument of the logarithm by inserting $$1 = \frac{q(t_i=c)}{q(t_i=c)}$$

$$
    \log p(X|\theta) = \sum_{i} \log \sum_{c}q(t_i=c) \frac{p(x_i, t_i=c| \theta)}{q(t_i=c)} = \sum_i \log \left \langle \frac{p(X,t_i|\theta)}{q(t_i)} \right\rangle_{q(t)}
$$

By this trick, Jenson inequality is applicable and we find a lower bound $$\mathcal L(\theta, q)$$

$$
    \log p(X|\theta) = \sum_i \log \left \langle \frac{p(X,t_i|\theta)}{q(t_i)} \right\rangle_{q(t)} \geq \sum_i \left \langle \log \left( \frac{p(X,t_i|\theta)}{q(t_i)}\right) \right \rangle_{q(t)}
$$

### The EM-algorithm

The lower bound $$\mathcal L (\theta, q)$$ now, is maximized in two steps. In the first step, called the **expectation step**,  we vary $$q(t_i)$$ while $$\theta$$ is kept fix. 

It can be shown, that the gap $$\Delta$$ between the loglikelihood $$p(X\|\theta)$$ and the lower bound $$\mathcal L$$ is given by the Kullback-Leibler divergence

$$
    \Delta = \log p(X|\theta) - \mathcal L(\theta, q) = \mathcal{KL}\left(q(t_i) || p(t_i| x_i, \theta)\right)
$$

which is minimized by $$q(t_i) = p(t_i\| x_i, \theta)$$.

In the second step, called the **maximization step**, the parameter $$\theta$$ is tuned to maximize the lower bound for the particular choice of $$q$$

$$
    \mathcal L(\theta, q) = \sum_i \langle \log \left(p(X,t_i|\theta)\right)\rangle_{q(t_i)} + const.
$$

The second term is constant w.r.t. $$\theta$$, the first term is usually concave and thus easily maximized by gradient ascent.

### Update Formulas
{: .no_toc}

**E-step:**

$$
    q^{k+1}(t_i) = p(t_i| x_i, \theta^k) = \frac{p(x_i|t_i, \theta^k) q^k(t_i)}{\sum_c p(x_i|t_i=c, \theta^k) q^k(t_i=c)}
$$

**M-step:**

$$
    \theta^{k+1} = \text{argmax} \sum_i \langle \log \left(p(x_i,t_i|\theta^k)\right)\rangle_{q^{k+1}(t_i)}
$$

## Application: Gaussian Mixture Models

An implementation of the EM method for gaussian mixture models can be found [here](https://github.com/BLyndon/bayesian_methods/blob/master/notebooks/GMM-EM.ipynb).

As a special case, a known training method for gaussian mixture models (GMM) is derived from the general EM principle. For a GMM the likelihood is given by a weighted sum of Gaussians

$$
    p(X|\theta) = \sum_c \pi_c \mathcal N (X; \mu_c, \sigma_c)
$$

with normalized weight parameters $$\sum_c \pi_c = 1$$.

### Expectation Step

By comparison to the latent variable approach above, we establish a correspondence for each quantity from the latent variable model to the GMM. Interestingly, the latent variable $$t_i$$ has a natural interpretation as the cluster component.

$$
\begin{aligned}
    p(t_i = c) = \pi_c\\
    p(x_i | t_i = c, \theta) = \mathcal N (x; \mu_c, \Sigma_c)\\
    q^{k+1}(t_i = c) = \gamma_{ic}
\end{aligned}
$$

In the expectation step we want to minimize the Kullback-Leibler divergence by setting the prior $$q^{k1}(t_i)$$ to the posterior $$p(t_i \|x_i, \theta^k)$$. After applying the Bayes formular to the likelihood $$p(x_i \| t_i = c, \theta)$$ we have

$$
    \gamma_{ic} = \frac{\pi_c \mathcal N (x_i; \mu_c, \Sigma_c)}{\sum_{c=1} \pi_c \mathcal N (x_i; \mu_c, \Sigma_c)}
$$

For numerical reasons, we rewrite the expression in the following way

$$
    \gamma_{ic} = \frac{\exp(y_{ic})}{\sum_{c=1} \exp(y_{ic})} = \frac{\exp(y_{ic} - \max(y))}{\sum_{c=1} \exp(y_{ic} - \max(y))}
$$

where $$y_{ic}$$ is given by

$$
    y_{ic} = \log \pi_c -\frac 1 2 \left(({x_i}-{\mu_c})^\mathrm{T}{\Sigma_c}^{-1}({x_i}-{\mu_c}) + d \log 2 \pi + \log \det \Sigma_c \right)
$$

### Maximation Step

In the maximization step the prior given by the expectation step is maximized w.r.t. $$\theta$$.

$$
\begin{aligned}
    \theta^{k+1} = \text{argmax} \sum_{ic} q(t_i = c) \log \left(p(x_i,t_i=c|\theta^k)\right) \\= \text{argmax} \sum_{ic} \gamma_{ic}\left(\log \pi_{c} + \log \mathcal N (x_i; \mu_c, \Sigma_c)\right)
\end{aligned}
$$

In case of a  GMM, this can be done analytically by solving the following equations

$$
\begin{aligned}
    \nabla_{\mu_{c}} \sum_{ik} \gamma_{ik} \log \left(\mathcal N (x; \mu_k, \Sigma_k)\right) = 0
    \\ \nabla_{\Sigma_{c}} \sum_{ik} \gamma_{ik} \log \left(\mathcal N (x; \mu_k, \Sigma_k)\right) = 0
\end{aligned}
$$

Additionally the priors $$p(t_i = c) = \pi_c$$ need to be updated by solving

$$
    \nabla_{\nu} \left( \sum_{ic}  \gamma_{ic} \log \pi_c - \lambda \left(\sum_c \pi_c -1 \right)\right) = 0, \quad \nu = \pi_1, \pi_2, \pi_3, \lambda
$$

where the Lagrange multiplier ensures normalization of the weights $$\pi_c$$.

Finally, this leads to the following update formulas

$$
\begin{aligned}
    \pi_c = \frac{\sum_i \gamma_{ic}}{\sum_{ic} \gamma_{ic}} = \frac{1}{N}\sum_{i=1}^N \gamma_{ic} \\
    \mathbf \mu_c = \frac{\sum_{i=1}^N \gamma_{c,i} \mathbf{x}_i}{\sum_{i=1}^N \gamma_{c,i}} \\
    \Sigma_c = \frac{\sum_{i=1}^N \gamma_{c,i} (\mathbf{x}_i - \mathbf\mu_c) (\mathbf{x}_i - \mathbf{\mu}_1)^\top }{\sum_{i=1}^N \gamma_{c,i}}
\end{aligned}    
$$


**Multivariate Gaussian PDF**

$$
\begin{aligned}
    \mathcal N (x; \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^d \det(\Sigma)}} \exp\left(-\frac 1 2 ({x}-{\mu})^\mathrm{T}{\Sigma}^{-1}({x}-{\mu})\right),\\
    x, \mu \in \mathbb R^d, \Sigma \in \mathbb R^{d\times d}
\end{aligned}
$$