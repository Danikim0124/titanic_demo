import pandas as pd
import csv as csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

with open('/Users/danikim/desktop/kaggle_titanic_demo/Data/train_cleaned_file.csv','r') as f:
    reader=csv.reader(f)
    data_train=np.array(list(reader))

X=data_train[1::,2::1]
y=data_train[1::,1]
# print(X)
# print(y)
model = DecisionTreeClassifier(max_depth=8, random_state=0)
model.fit(X,y)

with open('/Users/danikim/desktop/kaggle_titanic_demo/Data/test_cleaned_file.csv','r') as f1:
    reader=csv.reader(f1)
    data_test=np.array(list(reader))
X_test = data_test[1::,1::1]
predictions = model.predict(X_test)
# print(predictions)

with open('/Users/danikim/desktop/kaggle_titanic_demo/Data/gender_submission.csv','r') as f2:
    reader=csv.reader(f2)
    data_submission=np.array(list(reader))
    data_submission[1::,1]=predictions
    print(data_submission)

with open('/Users/danikim/desktop/kaggle_titanic_demo/Data/result.csv','w') as writef2:
    writer=csv.writer(writef2)
    writer.writerows(data_submission)
