# Scikit-Learn_Elasticnet_with_HyperparameterTuning
Use case of ElasticNet regression, which is a combination of L1(also known as Lasso) and L2(also known as ridge) prior as regularizer.

Lasso uses the L1 penalty to regularize;

  Quick reminder;
  lasso loss function = OLS(ordinary least square) function + alpha * sum(abs(all weights among the x axis))
  
  from sklearn.linear_model import Lasso
  lasso = Lasso(alpha=0.1, normalize=True) the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If alpha=0 we get back to plain OLS model which can lead to over fitting. Very high alpha means that large coefficients are significantly penalized, which can lead to a model too simple and ends up under fitting the data.

  Not: L1 regularizer is very good at important feature selection, which shrinks the coefficients of important features to exactly 0.
  

Ridge uses the L2 penatly to regularize;

  Quick reminder;
  ridge loss fucntion = OLS + alpha * sum((all weights among the x axis)**2)
  
  L2 is very good for multi variate regression, such as n_samples and n_targets. L2 adds extra squared value on the loss function.
  
Lasso ised the L1 penalty to regularize, while ridge used the L2 penalty to regularize, together in a linear combination of the L1 penalty and L2 penalty known as ElasticNet. 

  a*L1 + b*L2 In sklearn 'l1_ratio' of 1 corresponds to an L1 penalty and anything lower is a combination of L1 & L2.

It is very common to use ElasticNet with linear regression model or logistic regression model(LR) for the purpose of regression problems (usually continious data is required as output).

We can use SVM in some cases that we can use LR, but the case is to use ElasticNet which we cannot use with SVM, SVM can only use L2 regularization in some cases.
