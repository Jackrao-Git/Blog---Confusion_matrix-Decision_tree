# Confusion_matrix-Decision_tree


Based on the content covered on Oct 4th, I would love to talk about the confusion matrix and decision tree.
***

## Confusion Matrix


Confusion matrix is a table that is used in classification problems to assess where errors in the model were made.
Its former name is classification matrix, which later changed to confusion matrix.
The rows represent the actual classes the outcomes should have been. While the columns represent the predictions we have made.
For confusion matrix, there are three measures that we mentioned: precision, recall, and specificity.

Pecision is the ratio of correct positive predictions to the total predicted positives. Precision is calculated as the number of correct positive predictions divided by the total number of positive predictions.

Recall is calculated as the ratio between the number of Positive samples correctly classified as Positive to the total number of Positive samples. 

Specificity is calculated as the number of correct negative predictions divided by the total number of negatives. 

F1 score is the weighted average of precision and recall. F1 Score = 2 * (Precision * Recall) / (Precision + Recall). It is nice as it considers both precision and recall of the model, but this makes it harder to interpret.

Accuracy is calculated as the number of all correct predictions divided by the total number of the dataset. It is one of the most straight-forward metric out of the confusion matrix.

Here are codes in python that can visualize a confusion matrix, here I take the sample size to be 300. 


```
import numpy
from sklearn import metrics
import matplotlib.pyplot as plt

actual = numpy.random.binomial(1, 0.9, size = 300)
predicted = numpy.random.binomial(1, 0.9, size = 300)

confusion_matrix = metrics.confusion_matrix(actual, predicted)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
confusion_matrix_display.plot()
plt.show()
```

Using below lines,
```
Precision = metrics.precision_score(actual, predicted)
Recall = metrics.recall_score(actual, predicted)
f1_score(actual, predicted)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
```
We could calculate these measures. Notice that due to the randomness, these results may change every time we randomize the sample. With confusion matrix, one could tell the performance of the classification model. 

![下载](https://user-images.githubusercontent.com/108682585/194449710-8e20ec90-4108-4ba2-b51e-59a07b8c3e1d.png)


*True Positive(Bottom-Right): Predicted values correctly predicted as actual is positive.  
*False Positive(Top-Right):  Negative values are predicted as positive.  
*False Negative(Bottom-Left): Positive values are predicted as negative.  
*True Negative(Top-Left): Predicted values correctly predicted as actual is negative.  

Notice that there are version of confusion matrix that just return plain text and this one is just for a better look.

***

## Decision Tree

Decision tree is a technique that uses labeled input and output datasets to train models. Decision tree is a supervised learning algorithm that can be used for both classification and regression. Notice that a decision tree is drawn upside down with its root is at the top. 

Decision trees are traditionally trained in a greedy fashion, split after split that often in a binary fashion. 

In fact, decision trees are common in daily life. Think about the time when you buy a car, you would consider a price range first, then is the model, the engine, the color and so on. This sequence of thinking could visualize in a tree-like graph. Now consider the below graph:


<img width="496" alt="fruit-decision-tree" src="https://user-images.githubusercontent.com/108682585/194463218-1d5e1cc5-db83-423a-ab0c-1d1b3253eb57.png">

The root here is the question: Is it yellow? Then it splits out to two different answers, namely yes and no, and then continue this spliting process until reaching the lowest purity. PS: A node is 100% impure when a node is split evenly, namely 50% to 50%, and 100% pure when all of its data belongs to only one class.


From the image above, we could see some advantages and disadvantages about the decision tree.

First, it is easy to understand. Second, non-linear parameter does not affect performance. However, the drawbacks are overfitting and instability as the model could become unstable due to the variation in the data.

Categorical decision tree: Answers that fit in categorical values such as the coin toss heads or tails, 

Continuous decision tree: Also known as the regression tree as the current decisions depend on the farther up the tree. This give an edge to it as it could predict based on multiple variables rather than on a single variable compared to the categorical decision tree.


Real example of a decision tree:

Consider the following data set, 

![123](https://user-images.githubusercontent.com/108682585/194469473-fefd8ad3-6d59-439c-9522-d8bd149cbc18.PNG)

```
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("data123.csv")
# here I choose to assign a numerical value to each categorical value
abc = {'UK': 0, 'USA': 1, 'CN': 2}
df['Nationality'] = df['Nationality'].map(abc)
abc = {'YES': 1, 'NO': 0}
df['Promotion'] = df['Promotion'].map(abc)
features = ['Age', 'Working Years', 'Number of projects', 'Nationality']

x = df[features]
y = df['Promotion']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(x, y)
# this visualizes the tree out below
tree.plot_tree(dtree, feature_names=features)
```

![4](https://user-images.githubusercontent.com/108682585/194469830-695fefb2-9937-4ec4-91f7-9417e61557ed.PNG)

Information gain shows the information that is gained by a certain attribute. It is used to decide which element to split at each step in a decision tree. Entropy is the measure of unpredictability in a dataset. For example, we have a bag of candys and all of them are mixed. In this case the entropy is very high. When a node's entropy is zero(leaf node), no any further splitting need to be done on a leaf node.
Gini Index is a metric to measure how often a randomly chosen element would be incorrectly identified. It means an attribute with lower gini index should be preferred. If all elements are correctly divided into different classes, which is ideal, then it is considered to be pure. 

The Gini method uses this formula:

Gini = 1 - (x/n)^2 - (y/n)^2

x is the number of positive answers("YES"), n is the number of samples, and y is the number of negative answers ("NO"), which gives us the below calculation:

1 - (5 / 10)^2 - (5 / 10)^2 = 0.5, this gini impurity suggests that the elements are distributed equally into some classes and thus is not an optimal split.


### Greedy Algorithm in Decision Tree

A greedy algorithm builds a solution by going one step at a time through the feasible solutions, applying a heuristic to determine the best choice. This means that a greedy algorithm picks the best immediate choice and never reconsiders its choices. One of the abvious advantages for greedy algorithm is that the running time is relatively easy to analyze. However, Sometimes greedy algorithms fail to find the globally optimal solution because they do not consider all the data. 

Let's look at a case when greedy programming would fail:

![567](https://user-images.githubusercontent.com/108682585/194650536-863da207-a7a6-4d82-80c1-c09f5e2aa171.PNG)

Here is an example to find the greatest sum of a path in a tree. As greedy algorithm never looks back and reconsider previous decision, it would continue the execution after picking 12 and would never know there is a node with 99 on the left side of the tree. Therefore, the global result of the problem is wrong, while the local problems are all tackled in a correct way.


### Dynamic Programming

Compared to the greedy algorithm, dynamic programming makes decision at each step considering current problem and solution to the previously sub-problem to reach the optimal solution. It breaks down a big problem into many simpler sub-problems and realizing the fact that the optimal solution to the big problem would be depended on the solution to the sub-problems on the way. If there are overlapping among these sub-problems it divides, then the solutions to these sub-problems can be saved and reused in the future without calculating them again. This technique is called memoization that saves values in the array. Dynamic programming has some similarities to recursion as most of the optimization would need recursion. However, not all recursion can use dynamic programming as some of them do not have overlapping sub-problems, such as the merge sort.

There are four main steps in the dynamic programming.

*Divide problem into smaller problem of same type
*Conquer each one recursively
*Save the answers of each smaller problem for future use
*Combine answers from small problems to get the answer for the original problem

Let's look at an example:

Say the famous example: fibonacci sequence, which is a sequence of numbers in which each number is the sum of the two preceding ones.

If we starts from 0, the first several terms would be: 0, 1, 1, 2, 3, 5, 8 ...

In this case, if we build a A(n)(n stands for the order of each term) function that describes this sequence, it would be something like this:

```
A(0) = 0
A(1) = 1
A(2) = 1 = A(0) + A(1)
A(3) = 2 = A(1) + A(2)
A(4) = 3 = A(2) + A(3)
A(5) = 5 = A(3) + A(4)
A(6) = 8 = A(4) + A(5)
```

This would be a simplified example of how dynamic programming is used.



Picture references:

https://pantelis.github.io/cs634/docs/common/lectures/decision-trees/

https://brilliant.org/wiki/greedy-algorithm/#:~:text=23%2024%2026-,Limitations%20of%20Greedy%20Algorithms,future%20choices%20it%20could%20make.
