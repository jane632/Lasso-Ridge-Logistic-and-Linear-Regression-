# Lasso-Ridge-Logistic-and-Linear-Regression-
This project is basically for  learning purposes based on algorithms in machine learning
## Machine Learning

**It is a process where application are created without human intervention.**

### Supervised Machine Learning
**Regression Problem Statement**
#### If you have  continuos variable as the output it becomes a regression problem statement.
**Classification Problem Statement**
#### Any output that has fixed number of categories it becomes a classification problem.

**1.Linear Regression**

#### In linear regression, the best fit line helps us make predictions by capturing the relationship between the independent variable(s) and the dependent variable.



#### The best fit line should minimize the difference (error) between the actual data points and the predicted values. This difference is known as the residual error.



#### This is achieved by adjusting θ₀ (intercept) and θ₁ (slope) to minimize the total error using a method called Ordinary Least Squares (OLS), which minimizes the Sum of Squared Errors (SSE).



#### The process of adjusting θ₀ and θ₁ is done through Gradient Descent (for large datasets) or Normal Equation (for smaller datasets).


#### The model's performance is evaluated using metrics like:
* Mean Squared Error (MSE)

* Root Mean Squared Error (RMSE)

* R² Score (Coefficient of Determination)

**2.Ridge and Lasso Regression**

 #### a. Ridge and Lasso come into picture when overfitting happens with the training data where there is   low bias with the training data and high variance with the test data.
 
**Overfitting happens when the model learns too much from the training data, leading to low bias and high variance.**


** Ridge Regression (L2 Regularization)** adds a penalty on large coefficients, preventing them from growing too large and reducing overfitting.



**Lasso Regression (L1 Regularization)** not only penalizes large coefficients but can also eliminate some coefficients (feature selection), making the model simpler.


**Elastic Net combines both L1 and L2 penalties**, balancing feature selection and regularization.


**Assumptions of Linear Regression**

#### 1. Normally Distributed - Data will be trained well on data that is normally distributed.
#### 2. Standardization - Scaling your data by using z-scores. ( mean = 0 and std = 1).
#### 3. Linearity - If the data is too much linear it will get better results.
#### 4. Multicollinearity - This is when two indepedent features are highly correlated and it may cause one of the features to be dropped.
**Variance Inflation Factor helps in check of multcollinearity.**
#### 5.Homoscedasticity - refers to a situation in which the variance of the errors (residuals) in a regression model remains constant across all levels of the independent variable(s). This is to ensure unbiased and efficient coefficients.

**3.Logistic Regression**

#### It works well with binary classification
#### The question is why are we not using linear regression for a classification problem?
#### In logistic regression, a sigmoid function is introduced to squash the lines into a straight line,
**that is our values should be between 0 and 1**
#### The sigmoid function helps when the outlier issues occurs when using linear regression.
#### In Logistic regression,a non-convex function contains a lot of local minima unable to reach the global minima.
#### The problem of local minima , the slope = 0, In order to solve this we have a logistic regression cost function
**The log always helps to get the global minima.**


## Perfomance Metrics

**Confusion Matrix**

#### Accuracy = TP/ TP + FP + TN + FN
#### The model aim is to reduce false negatives and false positives.
#### What if our dataset is imbalanced?
#### An imbalanced dataset often affects algorithms , it always leads to higher accuracy but not a correct model.
#### One thing to check about a dataset is whether its balanced or imbalanced.

#### Precision - Out of the predicted positive values,how many are actual true or positive. Our main aim here is to reduce false positive.
#### Recall - Out of all the actual true positives,how many have been predicted correctly.Our main aim here is to reduce false negative.
**e.g Spam classification - Precision is the main perfomance metric.**
    **Has cancer or not - Recall is the main perfomance metric.**
#### F-score - 
**When Beta = 1 ,Both the false positive and false negative are important - F1-score**

**When Beta is decreased , FP is more important than FN - F.5score**

**When Beta is increased, FN is more important than  FP - F2-score**

**4.KNN Algorithm**
**Classification Problem**
Let's say k value is 5 and its going to take the 5 nearest closest points.Now from Group A we are getting 3 points and from group 2 we are getting 2 points . The maximum number of categories it is coming from thats the the category it will choose with the help of :
                                           **a) Euclidean Distance**
                                           **b) Manhattan Distance**
What is euclidean distance? Its the square root of the sum of differences between the x's and Y's.
What is manhattan distance? Takes the modulous

**Regression Problem**
##### For a regression problem, It will try to calculate the average of all points that will become the new data point.

K Nearest Neighbour works very bad with outliers and an imbalanced dataset.

**Decision Trees is explained well in the notebook file together with its practicals.Kindly this is for educational purpose only and to get familiar with the models and not a project.


