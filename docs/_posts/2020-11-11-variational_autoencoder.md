---
layout: page
title: 5. Variational Autoencoder
permalink: /vae/
---
* ToC
{:toc}

---

## Project Structure

Main notebook [vae_main.ipynb](https://github.com/BLyndon/bayesian_methods/blob/master/notebooks/vae_main.ipynb)  

**Modules**
+ (Conditional) variational autoencoder & encoder/decoder architectures: [variational_autoencoder.py](https://github.com/BLyndon/bayesian_methods/blob/master/notebooks/modules/variational_autoencoder.py)
+ Plotting toolkit: [vae_plotter.py](https://github.com/BLyndon/bayesian_methods/blob/master/notebooks/modules/vae_plotter.py)
+ Load and preprocess datasets: [data_loader.py](https://github.com/BLyndon/bayesian_methods/blob/master/notebooks/modules/data_loader.py)
+ Loss function: [loss_function.py](https://github.com/BLyndon/bayesian_methods/blob/master/notebooks/modules/loss_function.py)

## Sources

+ [*Kingma, Welling (2014)* Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
+ [*Sohn et. al (2015)* Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)