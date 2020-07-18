## General Expectation Maximation Algorithm

During the training of a model, the likelihood $p(X|\theta)$ of the observed dataset is maximized by tuning the parameters $\theta$.

Assuming i.i.d. samples and by introducing a latent variable, the loglikelihood can be written as

$$\log p(X|\theta) = \sum_i log p(x_i|\theta) = \sum_{i} \log \sum_{c}p(x_i, t_i=c| \theta)$$

The idea of the algorithm is to maximize a lower bound instead of the complicated loglikelihood $p(X|\theta)$. By inserting $1 = \frac{q(t_i=c)}{q(t_i=c)}$ we have

$$\log p(X|\theta) = \sum_{i} \log \sum_{c}q(t_i=c) \frac{p(x_i, t_i=c| \theta)}{q(t_i=c)} = \sum_i \log \left \langle \frac{p(X,t_i|\theta)}{q(t_i)} \right\rangle_{q(t)}$$

By the [Jenson inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality), we find a lower bound $\mathcal L(\theta, q)$

$$\log p(X|\theta) = \sum_i \log \left \langle \frac{p(X,t_i|\theta)}{q(t_i)} \right\rangle_{q(t)} \geq \sum_i \left \langle \log \left( \frac{p(X,t_i|\theta)}{q(t_i)}\right) \right \rangle_{q(t)}$$

### The EM-algorithm

The lower bound $\mathcal L (\theta, q)$ now is maximized in two steps. In the first step, called the **expectation step**,  we vary $q(t_i)$ while $\theta$ is kept fix. 

It can be shown, that the gap $\Delta$ between the loglikelihood $p(X|\theta)$ and the lower bound $\mathcal L$ is given by the Kullback-Leibler divergence

$$\Delta = \log p(X|\theta) - \mathcal L(\theta, q) = \mathcal{KL}\left(q(t_i) || p(t_i| x_i, \theta)\right)$$

which is minimized by setting $q(t_i) = p(t_i| x_i, \theta)$.

In the second step, called the **maximization step**, the lower bound is maximized with fixed q:

$$\mathcal L(\theta, q) = \sum_i \mathbb E_{q(t_i)} \log \left(p(X,t_i|\theta)\right) + const$$

where the second term is constant w.r.t. $\theta$. The first term is usually concave and is easily maximized by gradient ascent.

## Summary

**E-step:**

$q^{k+1}(t_i) = p(t_i| x_i, \theta^k) = \frac{p(x_i|t_i, \theta^k) q^k(t_i)}{\sum_c p(x_i|t_i=c, \theta^k) q^k(t_i=c)}$

**M-step:**

$\theta^{k+1} = \text{argmax} \sum_i \mathbb E_{q^{k+1}(t_i)} \log \left(p(x_i,t_i|\theta^k)\right)$


## Gaussian Mixture Model
In Gaussian Mixture Models, the likelihood is given by a weighted sum of Gaussians

$$p(X|\theta) = \sum_c \pi_c \mathcal N (X; \mu_c, \sigma_c)$$

By comparison to the latent variable approach above, we can interprete the values of the latent variable $t_i$ as the cluster components. 

#### *Multivariate Gaussian PDF*

$$\mathcal N (x; \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^d \det(\Sigma)}} \exp\left(-\frac 1 2 ({x}-{\mu})^\mathrm{T}{\Sigma}^{-1}({x}-{\mu})\right),\quad x, \mu \in \mathbb R^d, \Sigma \in \mathbb R^{d\times d}$$

## Expectation Step

+ $p(t_i = c) = \pi_c$
+ $p(x_i | t_i = c, \theta) = \mathcal N (x; \mu_c, \Sigma_c)$
+ $q^{k+1}(t_i = c) = \gamma_{ic}$

This leads to the following update formula

+ $\gamma_{ic} = \frac{\pi_c \mathcal N (x_i; \mu_c, \Sigma_c)}{\sum_{c=1} \pi_c \mathcal N (x_i; \mu_c, \Sigma_c)}$

For numerical reasons, we rewrite the expression in the following way

+ $\gamma_{ic} = \frac{\exp(y_{ic})}{\sum_{c=1} \exp(y_{ic})} = \frac{\exp(y_{ic} - \max(y))}{\sum_{c=1} \exp(y_{ic} - \max(y))}$

where $y_{ic}$ is given by

$y_{ic} = \log \pi_c -\frac 1 2 \left(({x_i}-{\mu_c})^\mathrm{T}{\Sigma_c}^{-1}({x_i}-{\mu_c}) + d \log 2 \pi + \log \det \Sigma_c \right)$

## Maximation Step

The maximization step

+ $\theta^{k+1} = \text{argmax} \sum_{ic} q(t_i = c) \log \left(p(x_i,t_i=c|\theta^k)\right) = \text{argmax} \sum_{ic} \gamma_{ic}\left(\log \pi_{c} + \log \mathcal N (x_i; \mu_c, \Sigma_c)\right)$

can be performed analytically by solving the following equations

+ $\nabla_{\mu_{c}} \sum_{ik} \gamma_{ik} \log \left(\mathcal N (x; \mu_k, \Sigma_k)\right) = 0$
+ $\nabla_{\Sigma_{c}} \sum_{ik} \gamma_{ik} \log \left(\mathcal N (x; \mu_k, \Sigma_k)\right) = 0$

Additionally the priors $p(t_i = c) = \pi_c$ need to be updated by solving

+ $\nabla_{\nu} \left( \sum_{ic}  \gamma_{ic} \log \pi_c - \lambda \left(\sum_c \pi_c -1 \right)\right) = 0, \quad \nu = \pi_1, \pi_2, \pi_3, \lambda$

where the Lagrange multiplier is needed to normalize the weights $\pi_c$.

Finally, this leads to the following update formulas

+ $\pi_c = \frac{\sum_i \gamma_{ic}}{\sum_{ic} \gamma_{ic}} = \frac{1}{N}\sum_{i=1}^N \gamma_{ic}$

+ $\mathbf \mu_c = \frac{\sum_{i=1}^N \gamma_{c,i} \mathbf{x}_i}{\sum_{i=1}^N \gamma_{c,i}}$

+ $\Sigma_c = \frac{\sum_{i=1}^N \gamma_{c,i} (\mathbf{x}_i - \mathbf\mu_c) (\mathbf{x}_i - \mathbf{\mu}_1)^\top }{\sum_{i=1}^N \gamma_{c,i}}$