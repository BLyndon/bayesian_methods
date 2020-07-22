---
layout: page
title: 4. Gaussian Processes
permalink: /gp/
---
* ToC
{:toc}

When hiking in the mountains not only the distance covered is interesting but also the altitude. The altitude along the track then is a 1-dimensional continuous function of the distance.

To create a height profile of the track the altitude is measured at certain points along the path. The amount of data points of course is restricted to a finite number, although the altitude function is defined everywhere.

After collecting the data, we want to construct the true function from the finite set of data points. Instead of fixing a parametric model function, we want to follow a different approach: To construct the function we want to derive a probability distribution over functions.

## Gaussian Processes

To derive a probability distribution over functions, we need to introduce a probability distribution for each outcome along the path. The finite set of measurements $$A(x_i)$$ can be used to infer the probability of the functional outcome for every possible value $$x$$ in continuous set of positions along the countinuous path.

$$
p(A(x)|A(x_1), \dots, A(x_k))
$$

But first, let's think about the joint probability for the measurements. The values $$A(x_i)$$ for each $$x_i$$ are a scalars, so we could try to model each these values by a one dimensional gaussian random variable resulting in a fully factorized joint distribution. 

Compared to climbing, hiking trails vary smoothly with the position, so the values of the random variables will depend on each other. This imposes a correlation between the points at different positions. While close points are strongly correlated, points far from each other are almost uncorrelated.

The factorized joint distribution, which is essentially a diagonal multivariate gaussian, is not able to model these correlations. Correlations between different measured values $$A(x_i)$$ are given by finite off-diagonal entries in the covariance. 

By replacing the diagonal by a non-diagonal covariance matrix, the multivariate gaussian distribution captures both observed qualitative features of our system.

**Gaussian Process** The probability distribution of a function $$y(x)$$ is a Gaussian process if for any finite selection of points $$x_1,\dots,x_k$$, the density $$p(y(x_1),\dots,y(x_k))$$ is Gaussian.

Being a multivariate gaussian, the joint distribution over the k variables $$A(x_1), \dots, A(x_k)$$ is fully specified by the mean and the covariance. The mean and the covariance depend neccesarily on the finite selection of points $$x_1,\dots,x_k$$. Otherwise the outcome would be sampled from the same distribution for all positions and the model wouldn't be able to sample non-constant functions.

The mean is set to be zero for symmetry reasons, since we lack prior knowledge. Additionally, the correlations given by real numbers are symmetric. The elements of the covariance matrix for all possible positional pairings will be modeled by a suitable kernel function $$k$$ decaying with the distance between two points

$$
cov(A_i, A_j) = k(x_i, x_j) = k(||x_i - x_j||) = \Sigma_{ij}
$$

A sample drawn from this Gaussian is a vector of $$k$$ elements corresponding to the vector of $$k$$ positions $$x_i$$. The ordering of these elements are fixed by the ordering of the covariance matrix elements, which in turn are determined by the ordering of the positional measurements. While the positional values follow no specific ordering, the ith element of the sample is paired with the ith element of the positional vector.

Remember, we aimed for a probability distribution over functions in the first place. Lets check with few lines of code wether the sample resembles a function already.