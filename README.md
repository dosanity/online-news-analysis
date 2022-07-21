# Online News Popularity

We will use classification tools to predict the popularity of online news based on attributes such as the length of the article, the number of images, the day of the week that the article was published, and some variables related to the content of the article. You can learn details about the dataset at the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity). 
This dataset was first used in the following conference paper: 

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. *Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence* (2015).

The dataset contains variables describing 39,644 articles published between January 7, 2013 and January 7, 2015 on the news website, [Mashable](http://mashable.com/). 
There are 61 variables associated with each article. Of these, 58 are *predictor* variables, 2 are variables that we will not use (url and timedelta), and finally the number of shares of each article. The number of shares is what we will use to define whether or not the article was *popular*, which is what we will try to predict. You should read about the predictor variables in the file *OnlineNewsPopularity.names*. Further details about the collection and processing of the articles can be found in the conference paper. 

## Classification Tools

### Classification using K-NN
The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems. It categories unknown variables into different clusters. In our study, we will try out different values to choose the value of K. Low values of K can be noisy and subject the effects of outliers. Large values of K smooth over things, but you don't want K to be so large that a category with only a few samples in it will always be out voted by other categories. We will be using training data, data used for initial clustering (data where we know the categories in advance) to calculate the accuracy of the data.

### Classification using SVM
In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. It uses the support-vector classifier which splits the data with a soft margin (the distance between the observations and the threshold) and some misclassifications. 


### Classification using Decision Trees
A classification tree is a structural mapping of binary decisions that lead to a decision about the class (interpretation) of an object. The top of the tree is the Root Node (The Root) which leads to Internal Nodes (Branches). These have arrows pointing to them and arrows pointing away from them. Finally, the bottom of the decision tree are called Leaf Nodes (Leaves). It is a non-parametric supervised learning method used for classification and regression.


