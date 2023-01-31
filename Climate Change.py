import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
data = pd.read_csv("D:\hackathon\climate change report.csv")
#Print First Five Values In Data Set
print(data.head())


X=data[['CO2 content ppm','Temprature in C']]#input varaible

y=data['Predictionvalues']#out put varaible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
   
KNN= KNeighborsClassifier(n_neighbors=3)
predictions=KNN.fit(X_train, y_train)
predictions = KNN.predict(X_test)
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

#New Input Varaible

CO2content=float(input("enter CO2 content in PPM     :"))
Temperature=float(input("enter temperature in celcius :"))
testPrediction=KNN.predict([[CO2content,Temperature]])
print(testPrediction ,'\n')


if testPrediction==0:
    print(' PH WILL NOT AFFECT REPRODUCTION OF MARINE ORGANISMS\n')

elif testPrediction==1:
    print('MILD PH CHANGE WILL AFFECT REPRODUCTION OF MARINE ORGANISMS\n')

else:
    print('HAZARDOUS CHANGE IN PH WILL AFFECT REPRODUCTION OF MARINE ORGANISMS \n')

#Confusion Matrix

CM=(confusion_matrix(y_test, predictions))
print(CM)
sns.heatmap(CM,cmap='Reds',annot=True)
plt.show()
