---
layout: page
title: 6. Information Theory
permalink: /it/
---
* ToC
{:toc}

---

## Self Information
We start our discussion on information theory by defining a completely general term of information $$I(x)$$ by some key properties we know from experience.

Let's consider a single circumstance described by a random variable $$x$$. If a circumstance is completely certain, i.e. $$p(x)=1$$, we know in advance, that this circumstance will happen. Since we were certain in advance, no information was gained by observing it. On the other hand, for low probabilities, the circumstance is very uncertain and by observing this circumstance, we gain a lot of information.

For two completely independent events, by observing the first event we gain no information about the second event. So by observing the second event we gain the full information from its observation, without any influence from the first event. This implies, that the information gained by observing both events sums up.

The second property might turn our focus to the logarithm function. From the first property, we know that the gained information depends on the probability.

The expression $$-\log p(x)$$ fulfills the property of a vanishing gain of information in case of certainty $$p(x)=1$$ and a high information gain for $$p \to 0$$. The second property is fulfilled as well

$$
    I(x, x') = - \log p(x, x') = - \log p(x)p(x') = -\log p(x) - \log p(x') = I(x) + I(x')
$$

## Information Entropy

As a second quantity, we introduce the information entropy, or Shannon-entropy, as the expectation value of the information

$$
    H_S = \langle I(x) \rangle = - k_S \sum_x p(x) \log p(x)
$$

with the constant $$k_S = 1/\log(2)$$ instead of the Boltzmann constant.

The entropy $$H_S$$ can be interpreted as the amount of uncertainty. This interpretation is motivated by three key properties.

Assuming $$\Omega$$ possible states with probabilities $$p(x_k) = p_k$$. Then, the highest uncertainty is given, if each state has equal probability. Consider the case of $$p_k' = 1$$ while the probabilities for the remaining states vanish. Then we are absolutely certain about the state of the system, since state k is the only possible state. In contrast, for equal probabilities the result can be any of the $$\Omega$$ states.

For uniformly distributed states, we have $$p_k = \frac{1}{\Omega}$$, $$k=1, ..., \Omega$$

$$
\begin{aligned}
    H_S(p_1, ... , p_\Omega) & = -\sum_k q_k \log q_k = \sum_k f(q_k) \\ & \le \Omega f\left(\frac{1}{\Omega} \sum q_k\right) = \Omega f\left(\frac{1}{\Omega}\right) \\ & = - \Omega \frac{1}{\Omega}\log \frac{1}{\Omega} = - \sum_k \frac{1}{\Omega} \log \frac{1}{\Omega} \\ & = H_S(1/\Omega, ... , 1/\Omega)
\end{aligned}
$$

The second key property implies, that taking into account more states with probability $$p(x)=0$$ must not change the uncertainty of the system. By $$-p \log p \to 0$$ for $$p \to 0$$, these particular terms in the entropy vanish and the finite value stays the same

$$
    H_S(p_1, ..., p_\Omega, 0, ...) = H_S(p_1, ..., p_\Omega)
$$

From the last property follows, that the entropy changes for conditional probabilities. The conditional probability expects certainty for some circumstances, and we gain this certainty by observing them. By observing states, we gain certainty about parts of the system, therefore the uncertainty of the system necessarily changes.

<!---
Starting with a joint distribution p(A,B), it can be shown, that the 

We start from the joint distribution p(A, B) with

$$
    p(A_j, B_k) = r_{jk}
$$

and

$$
    p(B_l) = q_l
$$

Then, using Bayes' theorem the conditional probability is given by

$$
    p(A_k|B_l) = c_{kl} = \frac{r_{kl}}{q_l}
$$

with

$$
    \sum_k p(A_k|B_l) = \sum_k c_{kl} = 1
$$

Before the measurement of $$B$$, the uncertainty of the system is described by $$H_p(A)$$ and $$H_p(B)$$. The joint probability then is given by

$$
    H(AB) = p(r_{11}, ... , r_{\Omega M}) = p(c_{11} q_1, ... , c_{\Omega M}q_M)
$$

Measuring $$B$$ decreases the uncertainty by $$H_p(B)$$ and we have

$$
    p(A|B) = p(c_{1l}, . . . , c_{Ωl})
$$
--->

## Kullback-Leibler Divergence

The Kullback-Leibler divergence for two probability distributions $$p(x)$$, $$q(x)$$ is defined as

$$
    \mathcal{KL}\left(p || q\right) = \mathbb E_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]
$$

Note, the Kullback-Leibler divergence is neither symmetric nor does it satisfy the triangle inequality. Therefore, it is not a metric and cannot be interpreted as a distance.

The most important property is the non-negativity
$$
    \mathcal{KL}\left(p || q\right) \geq 0
$$
with equality if and only if $$p=q$$.

A quantity related to the Kullback-Leibler divergence is the cross-entropy defined as

$$
    H(p,q) = -\mathbb E_{x \sim p}\log q(x)
$$

From the definition of the Kullback-Leibler divergence, we find

$$
    \mathcal{KL}\left(p || q\right) = \mathbb E_{x \sim p}\log p(x) - \mathbb E_{x \sim p}\log q(x)
$$

or

$$
    H(p,q) = H(p) + \mathcal{KL}\left(p || q\right)
$$

In particular this means, minimizing the cross-entropy w.r.t. $$q$$ is equivalent to minimizing the Kullback-Leibler divergence.

In a machine learning setting, we have

$$
    \mathcal{KL}\left(p_{data} || p_\theta\right) = -H[p_{data}] - \langle \log p_\theta\rangle_{data}
$$

or

$$
    \langle \log p_\theta(x)\rangle_{data} = - H[p_{data}] - \mathcal{KL}\left(p_{data} || p_\theta\right)
$$

This shows the equivalence between the maximization of the loglikelihood and the minimization of the Kullback-Leibler divergence.

## Sources

+ [*MacKay (2003)* Information Theory, Inference, Learning](http://www.inference.org.uk/mackay/itprnn/book.html)
+ [*Sethna (2005)* Statistical Mechanics: Entropy, Order Parameters and Complexity](http://sethna.lassp.cornell.edu/statistical_mechanics_entropy_order_parameters_and_complexity)
+ [*Mehta et al.* A high-bias, low-variance introduction to Machine Learning for physicists ](https://www.sciencedirect.com/science/article/pii/S0370157319300766)