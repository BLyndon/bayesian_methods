# Variational Inference

For many practical models evaluating P(Z|X) is infeasible and approximation schemes are required. There are mainly two types of approximation schemes. On the one hand there are stochastic approximation schemes such as Markov Chain Monte Carlo, but here we introduce a deterministic approximation scheme called *variational inference*. 

In variational inference, the probability distribution $P(X)$ is modeled by another distribution Q(X) in two steps. First, the functional class of $Q(X)$ is reduced to simple functional forms and afterwards we want to find the best model function $Q^*(X)$ within this class.



#### Variational Free Energy 

In isolated many particle systems the energy of the system is defined by the state **x** the interaction **J** between the degrees of freedom E(**x**, **J**). Given an inverse temperature, the probability of finding the system in state **x** is described by

$P(x|\beta, J) = \frac{\exp(-\beta E(x, J))}{Z_P(\beta, J)}$

The normalization constant, also called the partition function is given by

$Z_P(\beta, J) = Tr \exp(- \beta E(x,J))$.

Further, the free energy $F$ of the system is defined as

$F = \beta^{-1} \log Z_p(\beta, J) = \langle E(x, J) \rangle_P - \beta^{-1} H_P$,

where $H_P = \langle -\log P \rangle_P$ ist the entropy.

Approximating the true distribution $P$ by a wrong distribution $Q$, we can define the *variational free energy* $F_Q$

$F_Q = \langle E(x, J) \rangle_Q - \beta^{-1} H_Q = - \beta^{-1} \langle \log P(X| \beta, J) \rangle_Q - \beta^{-1} \log Z_P  - \beta^{-1} \langle -\log Q \rangle_Q$

which reduces to

$\beta F_Q - \beta F = \mathcal {KL} (Q || P) > 0$.

Since we made no assumptions about Q so far, the true free energy $F$ is always a lower bound for the variational free energy $F_Q$.