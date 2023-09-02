import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from sklearn.metrics import accuracy_score



# warnings.filterwarnings("ignore")

data=pd.read_csv("Crop_recommendation.csv")
data=np.array(data)

x=data[ : , :-1]
y=data[ : , -1]
# x=x.astype('int')


#Random forest can accept catagorical data it seems :) but XGboost does :(
le=LabelEncoder()
y=le.fit_transform(y)
print(y[ :])
#checking vro
res = []
for i in y:
    if i not in res:
        res.append(i)
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

# print(res)


#splitting

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)


# rf_classifier = RandomForestClassifier(random_state=0)
# rf_classifier.fit(x_train, y_train)
# rf_accuracy = rf_classifier.score(x_test, y_test)

#Initialize the XGBoost model
xgb_classifier = xgb.XGBClassifier()

#Train the model
xgb_classifier.fit(x_train, y_train)
# print("Random Forest Accuracy:", rf_accuracy)
# y_pred = rf_classifier.predict(x_test)

# accuracy = np.mean(y_pred == y_test)
# print("Random Forest Accuracy:", accuracy)
y_pred = xgb_classifier.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

pickle.dump(xgb_classifier,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

