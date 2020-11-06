---
layout: page
title: 3. Variational Inference
permalink: /vi/
---
* ToC
{:toc}

---

For many practical models evaluating $$p(Z\|X)$$ is infeasible and approximation schemes are required. There are mainly two types of approximation schemes. The first major group consists of **stochastic approximation schemes** such as Markov Chain Monte Carlo, the second major group is formed by **deterministic approximation schemes**. 

In this section we will introduce a determinisitic method called **variational inference**.

---

## Variational Lower Bound Decomposition

In variational inference, the probability distribution $$p(X)$$ is approximated by a simpler distribution $$q(X)$$ in two steps. First, the functional class of $$q(X)$$ is reduced and afterwards we want to find the best model function $$q^*(X)$$ within this class.

We start from a fully Bayesian model, where all parameters are stochastic, i.e. a prior is given for each parameter. Then we can absorb the stochastic parameters into the latent variables $$Z$$, sucht that they no longer explicitly appear in the notation.

The full probability can be rewritten as the expectation value of the conditional probability $$p(X\|Z)$$

$$
    \log p(X) = \log \left( \langle p(X|Z) \rangle_{q(Z)}\right) \ge \langle \log p(X|Z) \rangle_{q(Z)}
$$

where we used **Jensen's inequality**. By subtraction, the inequality gap turns out to be the **Kullback-Leibler divergence**, so we end up with the following decomposition

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

The posterior $$p(Z\| X)$$ is expected to be intractable, so we need to apply an approximation to proceed. As mentioned above, we restrict the family of distributions $$q(Z)$$. The goal will be a restriction to a class of tractable distributions.

But before we present possible restrictons, we derive the variational inference from a physical perspective in the next section.

---

## Variational Free Energy

In isolated many particle systems the energy $$E(x, J)$$ of the system is determined by the state $$x$$ and the interactions $$J$$ between the degrees of freedom. Given the inverse temperature $$\beta$$, the probability of finding the system in state $$x$$ is described by

$$
    p(x|\beta, J) = \frac{\exp(-\beta E(x, J))}{Z_p(\beta, J)}
$$

The normalization constant, called the partition function, is given by

$$
    Z_p(\beta, J) = Tr \exp(- \beta E(x,J))
$$

Further, the free energy $$F$$ of the system is defined as

$$
    F = \beta^{-1} \log Z_p(\beta, J) = \langle E(x, J) \rangle_p - \beta^{-1} H_p
$$

where $$H_p = \langle -\log p \rangle_p$$ ist the entropy.

Approximating the true distribution $$p$$ by any other distribution $$q$$, we can define the *variational free energy* $$F_q$$

$$
\begin{aligned}
    F_q = \langle E(x, J) \rangle_q - \beta^{-1} H_q \\
    F_q = - \beta^{-1} \langle \log p(X| \beta, J) \rangle_q - \beta^{-1} \log Z_p \\ = - \beta^{-1} \langle -\log q \rangle_q
\end{aligned}
$$

which reduces to the more expressive form

$$
    \beta F_q - \beta F = \mathcal {KL} (q || p) \ge 0
$$

As we made no assumptions about $$q$$ so far, the true free energy $$F$$ is always a lower bound for the variational free energy $$F_q$$.

With help of the prior knowledge from physics we circumvented the less obvious transformation of $$p(x)$$ to an expecation value followed by the application of Jenson's inequality. In addition the tedious calculation for the gap was also skipped.