# csci596: Quantifying Uncertainty using Deep Generative Modeling.
A repo for the final project of [Scientific Computing and Visualization (CS596)](http://cacs.usc.edu/education/cs596.html) course.

## Generative Adversarial Networks
![](https://github.com/dhruvpatel108/csci596/blob/main/images/gan.png)


## Objective
Bayesian inference is a powerful method used extensively to quantify the uncertainty in an inferred field given the measurement of a related field
when the two are linked by a mathematical model. Despite its many applications, Bayesian inference faces two major 
challenges: 
1. sampling from high dimensional posterior distributions and 
2. representing complex prior distributions that are difficult to characterize mathematically. 

It has been shown that use of GAN as a prior in Bayesian update can be effective in tackling above two challenges.

We demonstrate the efficacy of this approach by inferring and 
quantifying uncertainty in inference problems arising in computer vision and physics-based applications. 
In both instances we highlight the role of computing uncertainty in providing a measure of confidence in the solution,
and in designing successive measurements to improve this confidence. 

Following animation shows how the proposed method can be used in active learning/design of experiments setting in deciding
optimal sensor placement location:

CelebA             |  MNIST
:-------------------------:|:-------------------------:
![](https://github.com/dhruvpatel108/GANPriors/blob/master/images/celeba_oed.gif)  |  ![](https://github.com/dhruvpatel108/GANPriors/blob/master/images/mnist_oed.gif)
