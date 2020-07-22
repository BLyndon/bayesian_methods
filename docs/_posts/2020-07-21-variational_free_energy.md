---
layout: page
title: 3.1 Variational Free Energy
permalink: /vfe/
---
* ToC
{:toc}
In isolated many particle systems the energy $$E(x, J)$$ of the system is determined by the state $$x$$ and the interactions $$J$$ between the degrees of freedom. Given an inverse temperature $$\beta$$, the probability finding the system in state $x$ is described by

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

Approximating the true distribution $$p$$ by any other distribution $q$, we can define the *variational free energy* $$F_q$$

$$
\begin{aligned}
    F_q = \langle E(x, J) \rangle_q - \beta^{-1} H_q \\
    F_q = - \beta^{-1} \langle \log p(X| \beta, J) \rangle_q - \beta^{-1} \log \\ 
    Z_p  - \beta^{-1} \langle -\log q \rangle_q
\begin{aligned}
$$

which reduces to the more expressive form

$$
    \beta F_q - \beta F = \mathcal {KL} (q || p) \ge 0
$$

As we made no assumptions about $$q$$ so far, the true free energy $$F$$ is always a lower bound for the variational free energy $$F_q$$.

With help of the prior knowledge from physics we circumvented the less obvious transformation of $$p(x)$$ to an expecation value followed by the application of Jenson's inequality. In addition the tedious calculation for the gap was also skipped above.