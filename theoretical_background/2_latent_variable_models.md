# Latent Variable Models

The concept of latent or hidden variables plays a central role in generative models, i.e. models based on a full probability distribution over all variables. The observable features in these models may have sophisticated correlations between each other resulting in complex probability distributions. By introducing latent variables to the model, this complexity can be mapped to simpler and more tractable joint distributions of the expanded space. The latent variables can be introduced, using marginalisation

$$
p(x) = \sum_z p(x, z)
$$

The complex distribution over the observables $p(x)$ then can be constructed by simpler components $p(x, z)$. In case of continuous latent variables, the sum is replaced by an integral. **(Motivation for cont. latent var. models see Bishop Ch.12 Intro.)**

### Examples
+ clustering
+ topic modeling
+ blind source separation
+ dimensional reduction

The first two examples are discrete latent variable models. The remaining examples are continuous latent variable models.

### Interpretation
In many cases, the latent variables can be interpreted. In case of clustering, the value of the latent variable corresponds to the cluster component. In case of blind source separation, each latent variable corresponds to a source.

### Training
The latent variables are completely unobservable and must be inferred from the observed data. An important training method for latent variable models is the expectation maximation.