---
layout: page
title: Information Theory
---
## Self Information
We start by defining a completely general term of information $$I(x)$$ by some key properties we know from experience.

Lets consider a single cirumstande described by a random variable $$x$$. If a certain circumstance is completely certain, i.e. $$p(x)=1$$, we know in advance, that this circumstance will happen. Since we were certain in advance, no information was gained by observing it. On the other hand, for low probabilities, the circumstance was very unexpected and by observing this circumstance, we gain a lot of information.

For two completely independent events, we learn from the first event and the gained information nothing about the second event. So by observing the second event we have a gain of the full information from observing it. That means, the informations from observing both events sum up.

The second property might turn our focus to the logarithm function. From the first property, we know that the probabilty plays an explicit role for the definition of the information.

The expression $$-\log p(x)$$ fulfills the property of vanishing information gain form $$p(x)$$ and high information gain for $$p \to 0$$, as shown in the plot. Also the second property is fullfiled

$$
    I(x, x') = - \log p(x, x') = - \log p(x)p(x') = -\log p(x) - \log p(x') = I(x) + I(x')
$$

## Entropy

As a second quanitity, we introduce the information entropy, or Shannon-entropy, as the expectation value of the information

$$
    H_p = \langle I(x) \rangle = - \sum_x p(x) \log p(x)
$$

The entropy $$H_p$$ can be interpreted as the amount of uncertainty. This interpretation is motivated by three key properties.

Assuming $$\Omega$$ states with probabilities $$p(x_k) = p_k$$. The highest uncertainty is given, if each state has equal probability. Consider the case of $$p_k' = 1$$ while the probabilities for the remaining states vanish. Then we are completely certain about the state of the system, while for equal probabilites the result could by any of the $$\Omega$$ states.

For uniformly distributed states, we have $$p_k = \frac{1}{\Omega}$$, $$k=1, ..., \Omega$$

$$
\begin{aligned}
    H_q & = -\sum_k q_k \log q_k = \sum_k f(q_k) \\ & \le \Omega f\left(\frac{1}{\Omega} \sum q_k\right) = \Omega f\left(\frac{1}{\Omega}\right) = - \Omega \frac{1}{\Omega}\log \frac{1}{\Omega} = - \sum_k \frac{1}{\Omega} \log \frac{1}{\Omega} \\ & = H_p
\end{aligned}
$$

The second key property statea, that taking into account more states with probability $$p(x)=0$$ must not change the uncertainty of the system. By $-p \log p \to 0$$ for $$p \to 0$$, these particular terms in the entropy vanish and the finite value stays the same

$$
    H_p(p_1, ..., p_\Omega, 0, ...) = H_p(p_1, ..., p_\Omega)
$$

The last property states, that the entropy changes for conditional probabilities. The conditional probability expects, that we know that some circumstamces are certain and we gain this certainty by observing them. Performing measurements, intuitively changes the uncertainty of the system.

We start from a joint distribution p(A, B) with

$$
    p(A_j, B_k) = r_{jk}
$$

and

$$
    p(B_l) = q_l
$$

Then, the conditional probability is given by

$$
    p(A_k|B_l) = c_{kl} = \frac{r_{kl}}{q_l}
$$

and obviously

$$
    \sum_k p(A_k|B_l) = \sum_k c_{kl} = 1
$$

Before the measurement of $$B$$, the uncertainty of the sytem is given by $$H_p(A)$$ and $$H_p(B)$$. The joint probability then is given by

$$
    p(A, B) = p(c_{11} q_1, ... , c_{\Omega M}q_M)
$$

After we measure $$B$$, the uncertainty decreases by $$H_p(B)$$ and we have

$$
    p(A|B) = p(c1l, . . . , cΩl)
$$


## Kullback-Leibler Divergence

Sources
+ High Bias
+ Mackkay
+ Sethna, stat. physics
+ Goodfellow
+ Bishop