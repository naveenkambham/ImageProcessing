import numpy as np 
import cv2
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection



iris_data = datasets.load_iris()

#to understand the data
print(iris_data.data.shape) 
print(iris_data.target.shape)
print(iris_data.feature_names)


#using only 2 classes hence removing the class 2
indices = iris_data.target !=2

X= iris_data.data[indices].astype(np.float)
Y= iris_data.target[indices].astype(np.float)

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=7)

model = cv2.ml.LogisticRegression_create()
model.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
model.setMiniBatchSize(1)

model.setIterations(50)

model.train(X_train,cv2.ml.ROW_SAMPLE,Y_train)

predictions = model.predict(X_test)

print("accuracy is:", metrics.accuracy_score(Y_test,predictions))

