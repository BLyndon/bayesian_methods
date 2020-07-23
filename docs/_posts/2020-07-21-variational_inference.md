---
layout: page
title: 3. Variational Inference
permalink: /bayesian_methods/vi/
---

* ToC
{:toc}
For many practical models evaluating $$p(Z\|X)$$ is infeasible and approximation schemes are required. There are mainly two types of approximation schemes. The first major group consists of **stochastic approximation schemes** such as Markov Chain Monte Carlo, and the second major group is formed by **deterministic approximation schemes**. 

In this section we will introduce a determinisitic method called **variational inference**. 

## Variational Lower Bound Decomposition

In variational inference, the probability distribution $$p(X)$$ is approximated by a simpler distribution $$q(X)$$ in two steps. First, the functional class of $$q(X)$$ is reduced and afterwards we want to find the best model function $$q^*(X)$$ within this class.

We start from a fully Bayesian model, where all parameters are stochastic with given priors. We absorbe the stochastic parameters into the latent variables $$Z$$, the no longer appear explicitly in the notation.

The full probability can be rewritten as the expectation value of the conditional probability $$p(X\|Z)$$

$$
    \log p(X) = \log \left( \langle p(X|Z) \rangle_{q(Z)}\right) \ge \langle \log p(X|Z) \rangle_{q(Z)}
$$

where we used Jensen's inequality. By subtraction, the inequality gap turns out to be the Kullback-Leibler divergence, so we end up with the following decomposition 

$$
    \log p(X) = \mathcal L (q) + \mathcal{KL}(q||p)
$$

where

$$
\begin{aligned}
    \mathcal L (q) = \langle \log p(X|Z) \rangle_{q(Z)}\\
    \mathcal{KL}(q||p) = - \langle \log(\frac{p(Z|X)}{q(Z)})\rangle_{q(Z)}
\end{aligned}
$$

Maximizing the lower bound $$\mathcal L(q)$$ w.r.t. $$q$$ is equivalent to minimizing the gap, i.e. the Kullback-Leibler divergence. This is achieved by setting the prior $$q(Z)$$ equal to the posterior $$p(Z\|X)$$.

The posterior $$p(Z\| X)$$ is expected to be intractable now, so we need to start the approximation here. As mentioned above, we restrict the family of distributions $$q(Z)$$. The goal will be a restriction to a class of tractable distributions.

But before we present possible restrictons, we derive the variational inference from a physical perspective in the next section.