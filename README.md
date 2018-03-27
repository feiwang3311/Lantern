# Lantern

Lantern is the implementation of a machine learning framework prototype in [Scala](http://scala-lang.org/). The design of Lantern is built on two important and well-studied programming language concepts, [delimited continuations](http://web.cecs.pdx.edu/~apt/icfp09_accepted_papers/113.html) and multi-stage programming ([staging](https://scala-lms.github.io/) for short). Delimited continuations provides a very concise view of the reverse mode automated differentiation, which which permits implementing reverse-mode AD purely via operator overloading and without any auxiliary data structures. Multi-stage programming leading to a highly efficient implementation that combines the performance benefits of deep learning frameworks based on explicit reified computation graphs (e.g., [TensorFlow](https://www.tensorflow.org/)) with the expressiveness of pure library approaches (e.g., [PyTorch](http://pytorch.org/)).

This implementation is the companion artifact of our ICFP18 submission [Demystifying Differentiable Programming: Shift/Reset the Penultimate Backpropagator](https://arxiv.org). 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

[JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html)
[sbt](https://www.scala-sbt.org/1.0/docs/)

### How to run

Fork and clone this repo to your local machine.

In a terminal:

```
$ cd to src
$ sbt 
```

In sbt terminal:

```
sbt:ad> run

Multiple main classes detected, select one to run:

 [1] Hessian_JacobianOfGradient
 [2] Hessian_MuWang_1
 [3] Hessian_MuWang_LMS
 [4] Hessian_MuWang_LMS_IFLOOP
 [5] Hessian_vector_product
 [6] Jacobian
 [7] Jacobian_Forward
 [8] Jacobian_Map
 [9] LMS
 [10] LMS_Greg
 [11] LMS_vector
 [12] ShiftReset

Enter number: [info] Packaging /home/fei/bitbucket/Lantern/src/target/scala-2.12/ad_2.12-1.0.jar ...
[info] Done packaging.
11
```


## Running the tests

There are numerous tests in the main functions in each files. Using ``sbt run`` should trigger those test cases

## Running the evaluations

```
cd ICFP18evaluation/
./run_exp.sh
```
The resulting plots are generated in the save_fig/ dir.
