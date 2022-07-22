from typing import overload
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib
from imblearn.over_sampling import SMOTE

df = pd.read_csv('Train_Dataset_190604N.csv')
scale_mapper = {"yes":1, "no":0}
scale_mapper1 = {"Yes":1, "No":0}
scale_mapper2 = {445:1, 452:2, 547:3}
df["intertiol_plan"] = df["intertiol_plan"].replace(scale_mapper)
df["voice_mail_plan"] = df["voice_mail_plan"].replace(scale_mapper)
df["Churn"] = df["Churn"].replace(scale_mapper1)
df["location_code"] = df["location_code"].replace(scale_mapper2)

df['total_charge'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] 
df['total_calls'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls']
df['total_min'] = df['total_day_min'] + df['total_eve_min'] + df['total_night_minutes']

df.drop(['customer_id','location_code','total_day_min', 'total_day_calls', 'total_day_charge', 'total_eve_min',
       'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
       'total_night_calls', 'total_night_charge'],axis=1,inplace=True)
X = df.drop(['Churn'],axis=1)
y = df['Churn']
oversample = SMOTE(sampling_strategy=1.0, k_neighbors=5)
x_over, y_over = oversample.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xgbmodel = XGBClassifier(learning_rate=0.1, max_depth=12, n_estimators=100)
xgbmodel.fit(X_train, y_train)


# svmclf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
xgbmodel.fit(X_train, y_train)

joblib.dump(xgbmodel,'xgbmodel3.pkl')

# testdf = pd.read_csv('Test_Dataset_190604N_2.csv')
# testdf1 = testdf.copy(deep=True)

# #encoding the nominal categorical variables
# scale_mapper = {"yes":1, "no":0}
# testdf1["intertiol_plan"] = testdf1["intertiol_plan"].replace(scale_mapper)
# testdf1["voice_mail_plan"] = testdf1["voice_mail_plan"].replace(scale_mapper)

# #aggregating the features
# testx = testdf1.copy(deep=True)
# testx['total_charge'] = testx['total_day_charge'] + testx['total_eve_charge'] + testx['total_night_charge'] 
# testx['total_calls'] = testx['total_day_calls'] + testx['total_eve_calls'] + testx['total_night_calls']
# testx['total_min'] = testx['total_day_min'] + testx['total_eve_min'] + testx['total_night_minutes']

# #dropping the unnecessary columns
# testx.drop(['total_day_min', 'location_code', 'customer_id',
#        'total_day_calls', 'total_day_charge', 'total_eve_min',
#        'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
#        'total_night_calls', 'total_night_charge'],axis=1,inplace=True)
# testx.to_csv('predtest.csv', index=False) 
# #normalization
# # scaler = MinMaxScaler()
# # cols = ['account_length', 'number_vm_messages', 'total_min', 'total_calls',
# #        'total_charge', 'total_intl_minutes', 'total_intl_calls',
# #        'total_intl_charge', 'customer_service_calls',]
# # testx[cols] = scaler.fit_transform(testx[cols])


# #making the predictions
# predtest = xgbmodel.predict(testx)

# #preparing the output file for submission
# testdf1['Churn'] = predtest
# scale_mapper1 = {1:'Yes', 0:'No'}
# testdf1["Churn"] = testdf1["Churn"].replace(scale_mapper1)
# testdf1.drop(['account_length','location_code','intertiol_plan','voice_mail_plan', 'number_vm_messages', 'total_day_min',
#        'total_day_calls', 'total_day_charge', 'total_eve_min',
#        'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
#        'total_night_calls', 'total_night_charge', 'total_intl_minutes',
#        'total_intl_calls', 'total_intl_charge', 'customer_service_calls',],axis=1,inplace=True)
# testdf1.to_csv('pred42.csv', index=False) 
