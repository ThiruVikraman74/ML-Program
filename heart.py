import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv('C:/Users/thiru/Downloads/heart (1).csv')

s=data['sex'] = data['sex'].replace({'male':0,'female':1})


X=data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]#input varaible

y=data['target']#out put varaible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
   
KNN= KNeighborsClassifier(n_neighbors=3)
predictions=KNN.fit(X_train, y_train)
predictions = KNN.predict(X_test)

print(accuracy_score(y_test, predictions))
cm=(confusion_matrix(y_test, predictions, labels=model_obj.classes_))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['0','1','2'])
disp.plot()
plt.show()
print(disp)
print(classification_report(y_test, predictions))
    
#New Input Varaible

age=int(input("enter age"))
sex=int(input("enter sex"))
cp=int(input("enter cp"))
trestbps=int(input("enter trestbps"))
chol=int(input("enter chol"))
fbs=int(input("enter fbd"))
restecg=int(input("enter restech"))
thalach=int(input("enter thalach"))
exang=int(input("enter exang"))
oldpeak=int(input("enter oldpeak"))
slope=int(input("enter slope"))
ca=int(input("enter ca"))
thal=int(input("enter thal"))
testPrediction=KNN.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

print(testPrediction)


if testPrediction==0:
    print('heart disease')

else:
    print('no heart desease')
