
#author - naveen
#Machine Learning model to detect a person inside an image.
#histogram of oriented gradients technique is used to identify the person inside an image
import cv2
import matplotlib.pyplot as plt 
from os import listdir
import numpy as np 
from sklearn import model_selection as ms 

directory = r"E:/Drive/JobHunts/PythonCode/ImageProcessing/pedestrians128x64/"

files = listdir(directory)


#checking the few image data files
i=0
for file in files[0:5]:
	img= cv2.imread(directory+file)
	plt.subplot(1,5,i+1)
	i=i+1
	plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	plt.axis('off')
# plt.show()
X_people=[]

#hog descriptors
window_size =(48,96)
block_size = (16,16)
block_stride=(8,8)
cell_size=(8,8)
hog = cv2.HOGDescriptor(window_size,block_size,block_stride,cell_size,9)



for file in files:
	filename = directory+file
	img = cv2.imread(filename)

	if img is None:
		continue

	X_people.append(hog.compute(img,(64,64)))
	
X_people=np.array(X_people,dtype=np.float32)
Y_people=np.ones(X_people.shape[0],dtype=np.int32)



#loading the other than people images
file_dir=r"E:/Drive/JobHunts/PythonCode/ImageProcessing/pedestrians_neg/"
X_nonpeople=[]
for file in listdir(file_dir):
	img= cv2.imread(file_dir+file)
	img= cv2.resize(img,(64,128))
	X_nonpeople.append(hog.compute(img,(64,64)))

X_nonpeople = np.array(X_nonpeople,dtype=np.float32)
Y_nonpeople=  np.zeros(X_nonpeople.shape[0],dtype=np.int32)

X= np.concatenate((X_people,X_nonpeople))
Y= np.concatenate((Y_people,Y_nonpeople))


#preparing the test and training data
X_train,X_test,Y_train,Y_test = ms.train_test_split(X,Y,test_size=0.25,random_state=7)

print(len(X_test))

model = cv2.ml.SVM_create()
model.train(X_train,cv2.ml.ROW_SAMPLE,Y_train)

from sklearn import metrics

Y_Predicted = model.predict(X_test)

print(len(Y_test))
print(len(Y_Predicted))
print("accuracy of model: ",metrics.accuracy_score(Y_test,Y_Predicted))