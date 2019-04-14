# Scikit-Learn_Elasticnet_with_HyperparameterTuning
Use case of ElasticNet regression, which is a combination of L1 and L2 prior as regularizer.

Lasso uses the L1 penalty to regularize;

  Quick reminder;
  lasso loss function = OLS(ordinary least square) function + alpha * sum(abs(all values on the x axis))
  
  from sklearn.linear_model import Lasso
  lasso = Lasso(alpha=0.1, normalize=True) the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
