# Lantern

Lantern is the implementation of a machine learning framework prototype in [Scala](http://scala-lang.org/). The design of Lantern is built on two important and well-studied programming language concepts, [delimited continuations](http://web.cecs.pdx.edu/~apt/icfp09_accepted_papers/113.html) and multi-stage programming ([staging](https://scala-lms.github.io/) for short). Delimited continuations provides a very concise view of the reverse mode automated differentiation, which which permits implementing reverse-mode AD purely via operator overloading and without any auxiliary data structures. Multi-stage programming leading to a highly efficient implementation that combines the performance benefits of deep learning frameworks based on explicit reified computation graphs (e.g., [TensorFlow](https://www.tensorflow.org/)) with the expressiveness of pure library approaches (e.g., [PyTorch](http://pytorch.org/)).

This implementation is the companion artifact of our ICFP18 submission [Demystifying Differentiable Programming: Shift/Reset the Penultimate Backpropagator](https://arxiv.org). 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

[JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html)
[sbt](https://www.scala-sbt.org/1.0/docs/)

### Directory Organization
* [root directory for Lantern](./src)
  * [Lantern code directory](./src/main/scala/lantern)
    * [code for AD on scalar variable](./src/main/scala/lantern/ad_lms.scala)
    * [code for AD on vector and tensor](./src/main/scala/lantern/ad_lms_vector.scala)
    * [LMS framework code](./src/main/scala/lantern/dslapi.scala)
    * [data loader](./src/main/scala/lantern/scanner.scala)
  * [Lantern test directory](./src/test/scala/lantern)
      * [ScalaTest instance of RNN evaluation](./src/test/scala/lantern/vanillaRNN.scala)
      * [ScalaTest instance of LSTM evaluation](./src/test/scala/lantern/LSTM.scala)
      * [ScalaTest instance of TreeLSTM evaluation](./src/test/scala/lantern/sentimentTreeLSTM.scala)
      * [ScalaTest instance of CNN evaluation](./src/test/scala/lantern/mnistCNN.scala)
  * [directory for evaluation code](./src/out/ICFP18evaluation)
      * [directory for RNN evaluation code](./src/out/ICFP18evaluation/evaluationRNN)
      * [directory for LSTM evaluation code](./src/out/ICFP18evaluation/evaluationLSTM)
      * [directory for TreeLSTM evaluation code](./src/out/ICFP18evaluation/evaluationTreeLSTM)
      * [directory for CNN evaluation code](./src/out/ICFP18evaluation/evaluationCNN)
      * [script for evaluation](./src/out/ICFP18evaluation/run_exp.sh)
      * [directory for evaluation results](./src/out/ICFP18evaluation/save_fig/)
      
### Compile deep learning models to C++ programs

Once you have cloned this repo, enter into the root directory of Lantern ($PATH_REPO/src).

If you want to :

```
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
