# Confusion_matrix-Decision_tree


Based on the content covered on Oct 4th, I would love to talk about the confusion matrix and decision tree.


Confusion matrix is a table that is used in classification problems to assess where errors in the model were made.
The rows represent the actual classes the outcomes should have been. While the columns represent the predictions we have made.
For confusion matrix, there are three measures that we mentioned: precision, recall, and specificity.

Pecision is the ratio of correct positive predictions to the total predicted positives. Precision is calculated as the number of correct positive predictions (TP) divided by the total number of positive predictions (TP + FP).

Recall is calculated as the ratio between the number of Positive samples correctly classified as Positive to the total number of Positive samples. 

Specificity is calculated as the number of correct negative predictions divided by the total number of negatives. 

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
Precision = metrics.precision_score(actual, predicted)
Recall = metrics.recall_score(actual, predicted)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)

We could calculate these three measures. Notice that due to the randomness, these three results may change every time we randomize the sample. With confusion matrix, one could tell the performance of the classification model. 

*True Positive(Bottom-Right): Predicted values correctly predicted as actual is positive.  
*False Positive(Top-Right):  Negative values are predicted as positive.  
*False Negative(Bottom-Left): Positive values are predicted as negative.  
*True Negative(Top-Left): Predicted values correctly predicted as actual is negative.  
