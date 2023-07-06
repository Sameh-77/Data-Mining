## -------------- Sameh Algharabli -- CNG 514 -- Assignment 2 ------------------- ##

import pandas
import matplotlib.pyplot as plot
import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataSet = pandas.read_csv("yourData")

#pandas.set_option('display.max_rows', None)

print(dataSet)


#------------------------ Finding Missing Values -----------------------------------# 

missingData = dataSet['GENDER'].isnull().any()
print("Missing data in 'Gender':                ", end='')
print(missingData)

missingData = dataSet['AGE'].isnull().any()
print("Missing data in 'Age':                   ", end='')
print(missingData)

missingData = dataSet['SMOKING'].isnull().any()
print("Missing data in 'Smoking':               ", end='')
print(missingData)

missingData = dataSet['YELLOW_FINGERS'].isnull().any()
print("Missing data in 'Yellow fingers':        ", end='')
print(missingData)

missingData = dataSet['ANXIETY'].isnull().any()
print("Missing data in 'Anxiety':               ", end='')
print(missingData)

missingData = dataSet['CHRONIC DISEASE'].isnull().any()
print("Missing data in 'Chronic Disease':       ", end='')
print(missingData)

missingData = dataSet['FATIGUE '].isnull().any()
print("Missing data in 'Fatigue':               ", end='')
print(missingData)

missingData = dataSet['ALLERGY '].isnull().any()
print("Missing data in 'Allergy':               ", end='')
print(missingData)

missingData = dataSet['WHEEZING'].isnull().any()
print("Missing data in 'Wheezing':              ", end='')
print(missingData)

missingData = dataSet['ALCOHOL'].isnull().any()
print("Missing data in 'Alcohol':               ", end='')
print(missingData)

missingData = dataSet['COUGHING'].isnull().any()
print("Missing data in 'Coughing':              ", end='')
print(missingData)

missingData = dataSet['SHORTNESS OF BREATH'].isnull().any()
print("Missing data in 'Shortness of Breath':   ", end='')
print(missingData)

missingData = dataSet['SWALLOWING DIFFICULTY'].isnull().any()
print("Missing data in 'Swallowing Difficulty': ", end='')
print(missingData)

missingData = dataSet['CHEST PAIN'].isnull().any()
print("Missing data in 'Chest pain':            ", end='')
print(missingData)


# -- Using where() -- #

nullData = np.where(pandas.isnull(dataSet))
if(len(nullData[0]) > 0):
    print("\nThere are empty fields in the dataset\n")
else:
    print("\nThere are NO empty fields in the dataset\n")


#--------------------------------Noisy Data (Outliers) -------------------------------------------#

print("\n\n******************* Boxplots*******************")
dataSet.boxplot(column = ['AGE'], grid = False)
plot.suptitle("Age boxplot")
plot.show() 

#--------------------------Removing the outliers of the Age using IQR ------------------------#

# Finding the 1st quartile
arr=dataSet['AGE']
q1 = np.quantile(arr, 0.25)

# Finding the 3rd quartile
q3 = np.quantile(arr, 0.75)
med = np.median(arr)

# Finding the IQR
iqr = q3-q1

# Finding upper and lower boundaries
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
print("IQR: " + str(iqr))
print("Upper Bound: " + str(upper_bound))
print("Lower Bound: " + str(lower_bound))

print('\nThe following are the outliers in the boxplot:')
#finding the elements that are less than the lower bound and greater than the upper bound         
for i in range(len(arr)):
    if ((arr[i] <= lower_bound) | (arr[i]>=upper_bound)):  
        print(arr[i])
        dataSet=dataSet.drop(i) # here I drop the rows from the dataset
    
#dataSet1 = dataSet[(dataSet['AGE'] >= lower_bound) & (dataSet['AGE'] <= upper_bound)]

dataSet.reset_index(drop=True,inplace=True) # resetting the index of the dataset after removing age outliers
print(dataSet) # printing data after removing the age outliers 
dataSet.boxplot(column = ['AGE'], grid = False) # drawing boxplot after removing age outliers 
plot.suptitle("Age boxplot2")  
plot.show() 

#----------------------------------------------------------------------------------------------------------------#


#-------------------------Creating the training and testing sets -------------------------#


#Mapping 'YES' and 'NO' in the lung cancer to 1 and 0 respectively
label_encoder = preprocessing.LabelEncoder()
dataSet['LUNG_CANCER'] = label_encoder.fit_transform(dataSet['LUNG_CANCER'])

#Mapping 'M' and 'F' in the GENDER to 1 and 0 respectively
dataSet['GENDER'] = label_encoder.fit_transform(dataSet['GENDER'])

#print(dataSet)

X = dataSet.iloc[:,:-1].values # X= all the columns except the last one 
y = dataSet.iloc[:, 14].values # y= LUNG_CANCER columns 

# 2/3 of the X and y is taken as training data, 1/3 of the X and y is taken testing data # 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#-----------------------------Deciding on the best k --------------------------------------------# 
#error = []
# Calculating error for K values between 1 and 20
# for i in range(1, 20):
        # classifier = KNeighborsClassifier(n_neighbors = i)
        # classifier.fit(X_train, y_train)
        # # Predicting the Test set results
        # y_pred = classifier.predict(X_test)
        
        # error.append(np.mean(y_pred != y_test))

# plot.figure(figsize=(12, 6))
# plot.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
# plot.title('Error Rate K Value')
# plot.xlabel('K Value')
# plot.ylabel('Mean Error')

# plot.show()

# training and fitting the model 
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("\n******------------ Accuracy and Confusion matrix ---------------********\n")

# Making the Confusion Matrix
print("\nConfusion Matrix:\n")
cm = pandas.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True,)
print(cm)

#calculating the accuracy 
score = accuracy_score(y_test, y_pred)
print("\nThe prediction accuracy is: ", score)



# cm = confusion_matrix(y_test, y_pred)
# sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
# print('\nSensitivity : ', sensitivity)

# specificity = cm[1,1]/(cm[1,0]+cm[1,1])
# print('\nSpecificity : ', specificity)

print("\n**************------------ K-Fold cross validation ---------------******************\n")

score = cross_val_score(classifier, X, y, cv=10) # cv = number of folds (k) 
print("Cross Validation Scores are {}".format(score))
print("\nAverage Cross Validation score :{}".format(score.mean()))
