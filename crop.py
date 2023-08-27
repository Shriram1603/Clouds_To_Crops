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



# warnings.filterwarnings("ignore")

data=pd.read_csv("Crop_recommendation.csv")
data=np.array(data)

x=data[ : , :-1]
y=data[ : , -1]
x=x.astype('int')


#Random forest can accept catagorical data it seems :)
# le=LabelEncoder()
# y=le.fit_transform(y)
# print(y[ :])
#checking vro
res = []
for i in y:
    if i not in res:
        res.append(i)

# print(res)


#splitting

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# print(x_train)
# print(y_train)

rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(x_train, y_train)
rf_accuracy = rf_classifier.score(x_test, y_test)
# print("Random Forest Accuracy:", rf_accuracy)
y_pred = rf_classifier.predict(x_test)

accuracy = np.mean(y_pred == y_test)
print("Random Forest Accuracy:", accuracy)

pickle.dump(rf_classifier,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

