# Online News Popularity

We will use classification tools to predict the popularity of online news based on attributes such as the length of the article, the number of images, the day of the week that the article was published, and some variables related to the content of the article. You can learn details about the dataset at the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity). 
This dataset was first used in the following conference paper: 

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. *Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence* (2015).

The dataset contains variables describing 39,644 articles published between January 7, 2013 and January 7, 2015 on the news website, [Mashable](http://mashable.com/). 
There are 61 variables associated with each article. Of these, 58 are *predictor* variables, 2 are variables that we will not use (url and timedelta), and finally the number of shares of each article. The number of shares is what we will use to define whether or not the article was *popular*, which is what we will try to predict. You should read about the predictor variables in the file *OnlineNewsPopularity.names*. Further details about the collection and processing of the articles can be found in the conference paper. 

## Classification Tools

### Classification using K-NN
The K-nearest neighbors (K-NN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems. It categories unknown variables into different clusters. In our study, we will try out different values to choose the value of K. Low values of K can be noisy and subject the effects of outliers. Large values of K smooth over things, but you don't want K to be so large that a category with only a few samples in it will always be out voted by other categories. We will be using training data, data used for initial clustering (data where we know the categories in advance) to calculate the accuracy of the data.

We develop a K-NN classification model for the data and use cross validation to choose the best value of K. Using the Train Test Split validation procedure (code below) with a test size of 0.8, we check K values from 1 to 100:

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.8)
scores = []
k_range = range(1,100)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
```

![KNN-Accuracy1](https://user-images.githubusercontent.com/29410712/180119694-7b4d03ba-9429-45a3-918f-468cec392e22.png)

To narrow the search for the best K, we use the same procedure to check K values from 93 to 140:

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.8)
scores = []
k_range = range(93,140)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
```

![KNN-Accuracy2](https://user-images.githubusercontent.com/29410712/180119712-b6f314cf-f1bd-4e39-8355-af2a0180b193.png)

From the two charts, we can see where K has the highest accuracy. This helps us narrow it down to check K values between 90 to 100. Using the code below, we substitute in the value of K to approximate the accuracy.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
clf = KNeighborsClassifier(n_neighbors=93)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
```
From the results, we conclude that the best accuracy one can obtain from the test data is around 0.567 with k = 93.

### Classification using SVM
We now develop a support-vector machine classification model for the data. In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. It uses the support-vector classifier which splits the data with a soft margin (the distance between the observations and the threshold) and some misclassifications. 

Using Train Test Split validation procedure, we can narrow down the C values to see which C value has the highest accuracy.

```
X_train, X_test, y_train, y_test = train_test_split(X_first, y_first, random_state=1, test_size=0.8)
for C in [1, 10, 100, 1000, 8100, 10000]:
    print("C is:", C)
    clf = svm.SVC(kernel='rbf', C=C)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy Train:", metrics.accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
    print("Accuracy Test:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    print()
```

Results are given below:
```
C is: 1
Accuracy Train: 0.562
Accuracy Test: 0.56175

C is: 10
Accuracy Train: 0.563
Accuracy Test: 0.56175

C is: 100
Accuracy Train: 0.563
Accuracy Test: 0.56175

C is: 1000
Accuracy Train: 0.586
Accuracy Test: 0.56625

C is: 8100
Accuracy Train: 0.6
Accuracy Test: 0.5815

C is: 10000
Accuracy Train: 0.602
Accuracy Test: 0.57775
```

From the Train Test Split validation procedure above, we can narrow our search to C values between 8090 to 8110.

```
X_train, X_test, y_train, y_test = train_test_split(X_first, y_first, random_state=1, test_size=0.8)
scores2 = []
C_range = range(8090, 8110)
for C in C_range:
    clf = svm.SVC(kernel='rbf', C=C)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores2.append(metrics.accuracy_score(y_test, y_pred))
```
![SVM-Accuracy](https://user-images.githubusercontent.com/29410712/180119730-fc548d14-00ee-4814-b1e3-eba6e8647eab.png)

From the results, we conclude that the best accuracy one can obtain from the test data is around 0.58175 with C = 8106.

### Classification using Decision Trees
A classification tree is a structural mapping of binary decisions that lead to a decision about the class (interpretation) of an object. The top of the tree is the Root Node (The Root) which leads to Internal Nodes (Branches). These have arrows pointing to them and arrows pointing away from them. Finally, the bottom of the decision tree are called Leaf Nodes (Leaves). It is a non-parametric supervised learning method used for classification and regression.

To check which values will result in the highest accuracy, we will check the maximum depths: `[5, 10, 15, 20]` and the minimum sample split: `[10, 100, 200, 400, 800]`. The results are shown below:

![DecisionTree-Accuracy](https://user-images.githubusercontent.com/29410712/180119785-e7e20abe-595a-43c7-b09e-025e8391820a.png)

From our calculations, we can see that the maximum depth of 10 and the minimum sample split of 400 will give us the highest accuracy. Therefore we will create a decision tree with these values.

```
Accuracy on training data = 0.6657249520734537
Accuracy on test data = 0.6377762082534557
```

![DecisionTree](https://user-images.githubusercontent.com/29410712/180126065-cab71206-fe1d-40e3-8156-a3e1d6256b54.png)

## Results

Based on all the accuracies, the Decision Tree worked best with the highest accuracy achieved for the Test Data of 0.6377. For the K-NN classifier, the parameter K can infuence the accuracy. Choosing smaller values for K can be noisy and will have a higher influence on the result while larger values of K will have smoother decision boundaries which mean lower variance but increased bias. For the SVM classifier, a larger C decreases the final training error, but if you increase C too much you risk losing the generalization properties of the classifier because it will try to fit as best as possible all the training points. If C is small, then the classifier is flat. For Decision Trees, there is a risk of overfitting when the number is higher. All three have a certain "sweet spot" where the accuracy is maximized. The model that is the easiest to interpret is the Decision Tree model because the information can be expressed in a readable form. In the K-NN model, the test is accurate about 0.567 of the time. In the SVM model, the test is accurate about 0.581 of the time. In the Decision Tree, the test is accurate about 0.638 of the time.

