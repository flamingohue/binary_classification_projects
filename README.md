# binary_classification_projects

### This is a project to do a binary classification to predict which customer is likely to churn. 

## Features: 
       - CustomerId, 
       - CreditScore, 
       - City, 
       - Gender, 
       - Age, 
       - BranchId,
       - Tenure, 
       - Balance, 
       - CurrencyCode, 
       - PrefLanguage, 
       - NumOfProducts,
       - PrimaryAcHolder, 
       - HasOnlineService, 
       - HasCrCard, 
       - PrefContact,
       - IsActiveMember, 
       - EstimatedSalary
      
# Model development process
<ol>
<li>Train-validation 7:3 split from training dataset</li>
<li>Metrics: Accuracy, Recall, Precision, AUC-ROC score</li>
<li>Candidate Models: Logistic Regression, Random Forest, XGBoost, SVM</li>
<li>Normalization: MinMax scale  each feature to (0, 1)</li>
<li>Train the initial models with all features</li>
<li>Feature selection: Rank feature importances based on the best model and take top features</li>
<li>Hyperparameter Search:  Use validation AUC-ROC  as the performance metrics to select the best hyperparameters for each model.</li>
<li>Test Performance on the test datasets using the same metrics</li> 
</ol>
