---
layout: page
title: 7. Independent Component Analysis
permalink: /ica/
---
* ToC
{:toc}

---
Here we cast the problem of source separation as a latent variable model and derive the ideas leading to a solution for an environment without noise and reverberation.

## Blind Source Separation

Consider a fixed setting of $$I$$ receivers and $$J$$ sources, denoted by $$x_j$$, $$s_i$$ respectively.

Assuming the mixed signal $$x_i$$ at receiver $$i$$ is a linear superposition of $$J$$ unknown sources {$$s_j$$}

$$
    x_i(t) = \sum_j A_{ij} s_j(t),\quad \text{for all }t
$$

or written as matrix equation

$$
    x(t) = A s(t),\quad \text{for all }t
$$

where the instantaneous mixing matrix $$A$$ is fixed for all t by the spatial arrangement of the sources and the signal propagation.

For $$I=J$$ the separation of the blind sources can be achieved by estimating and inverting the mixing matrix $$A$$. Denoting the inverted mixing matrix by W, the separated signals are given by

$$
    s(t) = Wx(t),\quad \text{for all }t
$$

Before we get into the solution of the problem, we would first like to point out two problems.

### Scaling Ambiguity
By simultaneously scaling the signals $$s_n$$ by a factor $$c$$ and the mixing matrix $$A_{mn}$$ by a factor $$1/c$$, these factors cancel out and superposition does not change. But the magnitude of the separated signals will change.

(e.g. signal damping)

### Permutation Ambiguity
The order of the sources is interchangeable, since the order of the terms in the superposition is commutative. 

(e.g. interchange speaker)

## Blind Source Separation as a Latent Variable Model
By representing the problem as a latent variable model, we have the ability to learn the matrix $$W$$ from a data set $$D=\{x^{(n)}\}_{n=1}^N$$, where $$x^{(n)}=x(t_n)$$ for $$N$$ discrete data points.
To do this, we set up the likelihood function p(D|W), where the matrix $$W$$ is fixed by the system described above and maximize this function with respect to the matrix $$W$$. Assuming an iid. data set, we have

$$
    p(D|W) = \prod_n p(x^{(n)}|W)
$$

and further

$$
    p(x^{(n)}|W)  = \int d^I s^{(n)} p(x^{(n)}, s^{(n)}|W)
$$

Using the superposition of signals, we have 

$$
    p(x^{(n)}\|s^{(n)},W)=\prod_j \delta(x_j^{(n)} - \sum_i W_{ij}s_j^{(n)})
$$ 

and thus

$$
\begin{aligned}
    p(x^{(n)}, s^{(n)}|W) & = p(x^{(n)}|s^{(n)},W)p(s^{(n)}) \\ & = \prod_j \delta(x_j^{(n)} - \sum_i W_{ji}s_i^{(n)}) \prod_j p(s_j^{(n)})
\end{aligned}
$$

Finally, we have

$$
    p(x^{(n)}|W) = \det W \prod_i p_i(\sum_jW_{ij}x_j)
$$

Now after specifying $$p_i(s_i)$$ we have a explicit expression for the loglikelihood, that depends only on the data set and the inverse mixing matrix.
The true inverse mixing matrix $$W$$ maximizes the likelihood function, therefore we can learn the values by applying gradient ascent to the loglikelihood function

$$
    \log p(D|W) = N \log \det W + \sum_{n,i} p_i(\sum_jW_{ij}x_j^{(n)})
$$

## Outlook

### Instantaneous Mixture Models
So far we discussed an instantaneous mixing of $$J$$ signals measured by $$I$$ receivers

$$
    x_i(t) = \sum_j A_{ij} s_j(t),\quad \text{for all }t
$$

In a more realistic model, we need to add noise $$v_i(t)$$ at receiver $$i$$.

$$
    x_i(t) = \sum_j A_{ij} s_j(t) + v_i(t),\quad \text{for all }t
$$

Thus, the problem cannot be solved by inverting the mixing matrix as we have done it before.

### Convolutive Mixture Models
In the case of a reverberant environment, the signals reach the receivers via different paths. The different propagation times along the paths lead to a temporal mixing of the signals.

This results in a convolutive mixture of the original sources

$$
    x_i(t) = \sum_{l=-\infty}^\infty\sum_j A_{ijl} s_j(t-l) + v_i(t),\quad \text{for all }t
$$

## Sources

+ [*MacKay (2003)* Information Theory, Inference, Learning](http://www.inference.org.uk/mackay/itprnn/book.html)