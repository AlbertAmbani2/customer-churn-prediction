Week 1 project: Predicting customer churn.

1.	What customer churn event means.
Customer churn means the customer is no longer a subscriber of the company. This means the customer had subscribed earlier to the services of the company until it reached a point the customer terminates the subscription. For instance, there are Sprint customers who have stopped using their services due to various reasons. 

2.	Data Collection
After identifying what the problem is about, related Sprint historical data is gathered about when customers have left before, information about these customers, the type of services they used, type of billing and other related things.

```python 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
```
```python
#Load the dataset
df = pd.read_csv('/kaggle/input/churn-prediction/tele.csv')

print(df.head())
```

3.	Data Preprocessing
This involves cleaning and transforming data. Here data is transformed by handling missing values, dealing with outliers and carrying out normalization on the data.

How I preprocessed data

```python
#check for missing values
df.isna().sum()

#drop customer ID
df = df.drop(['CustomerID'], axis=1)
df.head()
```


4.	Feature Engineering and Selection
Here, new features can be added to existing ones in order to extract valuable information. For instance, features such as period the customer has been with Sprint and overall usage statistics of the customer with Sprint can be added.

Specific features will include
MonthlyCharges of customers
TotalSpent by each customer
ContractType of a customer


5.	Model Selection 
An appropriate model is selected that is appropriate for the problem. Churn prediction is a problem that requires binary solution. Thus, likely choice for churn prediction is logistic regression.


6.	Model Training
In this part, training of the selected model is carried out. Historical data collected is fed into the model allowing it to learn from the patterns associated with churn.

```python
#Split data into train and test sets¶ 
def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

    X = df.drop(columns = ['Churn'])
    y = df['Churn'].values

    #train using train set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40, stratify=y)
```
7.	Model Evaluation
Use of validation set to determine model performance is carried out here. Evaluation metrics for churn prediction include Accuracy, precision, recall. F1 score and ROC-AUC. Evaluate the model on test to determine model performance. This process ensures that the model generalizes well with new data.

```python
lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
accuracy_lr = lr_model.score(X_test,y_test)
print("Logistic Regression accuracy is :",accuracy_lr)
```

```python
lr_pred= lr_model.predict(X_test)
report = classification_report(y_test,lr_pred)
print(report)
```

8.	Model deployment and integration
Once satisfied with the model, deploy it into Sprint Company operations in order to make real time predictions on new data. This could involve integrating the model into customer management system.


