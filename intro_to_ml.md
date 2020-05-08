# Introduction
This is an introduction to image classification using pytorch.
It is directed to software engineers willing to transition to machine learning and what to know roughly the inner workings of neural networks, including modeling, training and inference.
For that, this project aims to develop a minimal code to train a classifier for images.

# Problem Definition
First and foremost, we are going to define formally the problem we want to solve. 
Given an image we want to predict if it belongs to one of a fixed and predefined number of classes.
For sake of simplicity, say from now on, in our problem we have images of cows and chickens (this can be generalized to any number of classes)
To make such predictions we have available a set of images of chickens as well as cows.
Before picking the right tool to solve the problem we have to come to agreement as to a mechanism to measure the quality of our model,
regardless of the one we pick.
Usually, for classification, a good method is the F1-Score. This metric takes into account to types of errors:
 - Precision: given the predictions of a given class (say cow), which is the proportion of correctly classified images/cows.
 - Recall: given a set of images and a class (say again cow), which is the propotion of the cows that have been predicted.

The formula is the following: 2 * (precision * recall ) / (precision + recall)

```
For that, we develop a class to compute the average F1-score accross classes that can be found in image_classifier/F1Score.py
```

Once we establish the F1-Score as metric, we should establish the data we will use to apply the F1-Score.
There are many ways to partition data (like cross-validation) but for sake of simplicity we are going to make just one partition with two types of data: train and test.
The test data will be the data we are going to use to compute the F1-Score and won't be available to train the model.
On the other side, the train data will be the only one used to train the model.

```
To split the dataset, we develop a script to make a random split in DatasetBuilder.py
```


# Model

Once we know the problem we are facing and a precise way to compute how good our solution is (once we have one!), we 
are now aiming to pick an algorithm to solve our problem. We are going towards and easy path reusing other's hard work
for use by reusing an already trained neural network.

Neural networks are a graph of blocks named 'layers', each taking one or more inputs and producing one or more outputs.
The way the imputs connect to the outputs is up to the neural network designer. 

There are two main types of layers: the ones that have no weights and the ones with weights. The ones without weights are deterministic
functions, namely they will always produce the same outputs given the same inputs. The ones with weights are the stochastic ones,
whos outputs will not only depend on the inputs but also on the weights. 

We are going to use ResNet trained on Imagenet, which is a large image classification problem. We expect the weights (also called 'parameters') to be if not optimal, close to optimal. The only 
caveat is that the last layer that produces the probabilities 
for the classes in the ImageNet problem shall be substited by
our problem (e.g. cows/chicken).

```
You can see in image_classifier/PretrainedResnet.py how 
a resnet is taken and last layer is swapped by a randomized one.
```

# Training

At this point we need to fine tune the weights of our adapted
resnet model so that they are optimal for our problem.
For that we will use Stochastic Gradient Descend which is a very
basic algorithm that updates the weights so that the error goes downhill up until it finds a valley.

The first problem we have is that SGD and backpropagation not only
requires a differentiable network but also a differentiable 'error'
or 'loss' function. It turns out that F1-Score is not directly differentiable and this means we have to come up with a loss function that is differentiable and whos optimal is close to or precisely.

(Note, there are ways to use F1-Score directly by approximation: https://datascience.stackexchange.com/questions/66581/is-it-possible-to-make-f1-score-differentiable-and-use-it-directly-as-a-loss-fun )

One of the solutions is using the SoftMax, which assumes that 
classes are exclusive (there is a cow or a chicken but not both together). Another option is to use cross-entropy, that is suitable
for two classes as well as multiclass. For this project, we allow multiclass and thus we have cross-entropy.

```
The training can be found in image_classifier/ClassifierTrain.py
```

# Inference
 
Once the model is trained, it's often times suitable to have a library 
or a script to use the network to extract its predictions.


```
For that we have the code ClassifierInferenceCli.py to directly do predictions with images.
```







