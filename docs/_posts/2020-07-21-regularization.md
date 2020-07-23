---
layout: page
title: 1.2 Bayesian Regularization
permalink: /bayesian_methods/reg/
---
* ToC
{:toc}
The prior distribution introduces regularization in a natural way. Adding regularization to the cost function constrains the magnitude of the parameters. The same can be achieved by a the prior $$p(\theta)$$ in a bayesian model, forcing smaller magnitudes by a higher probability around the origin.

As an example, we consider a prior distribution $$p(\theta) = \mathcal N(\theta; 0, I/\lambda)$$, leading to the following logposterior

$$
    \log p(\theta|X) = \log \frac{p(X|\theta)p(\theta)}{p(X)} = \log p(X|\theta) - \frac{\lambda}{2} \sum_i \theta_i^2 + const.
$$

Thus, this particular choice of the prior leads to a $$L^2$$ regularization.