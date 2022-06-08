import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('Train_Dataset_190604N.csv')
scale_mapper = {"yes":1, "no":0}
scale_mapper1 = {"Yes":1, "No":0}
scale_mapper2 = {445:1, 452:2, 547:3}
df["intertiol_plan"] = df["intertiol_plan"].replace(scale_mapper)
df["voice_mail_plan"] = df["voice_mail_plan"].replace(scale_mapper)
df["Churn"] = df["Churn"].replace(scale_mapper1)
df["location_code"] = df["location_code"].replace(scale_mapper2)

X = df.drop(['customer_id','Churn'],axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xgbmodel = XGBClassifier()
xgbmodel.fit(X_train, y_train)


# svmclf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
xgbmodel.fit(X_train, y_train)
joblib.dump(xgbmodel,'xgbmodel.pkl')
