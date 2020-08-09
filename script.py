import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

#Formulate columns that hold strike and ball information to numeric values
aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B':0})
jose_altuve['type'] = jose_altuve['type'].map({'S':1, 'B':0})
david_ortiz['type'] = david_ortiz['type'].map({'S':1, 'B':0})

#Drop Nan values
aaron_judge = aaron_judge.dropna(subset=['plate_x', 'plate_z', 'type'])
jose_altuve = jose_altuve.dropna(subset=['plate_x', 'plate_z', 'type'])
david_ortiz = david_ortiz.dropna(subset=['plate_x', 'plate_z', 'type'])


#Set and create scatter plot with data
fig, ax = plt.subplots()
plt.scatter(aaron_judge.plate_x, aaron_judge.plate_z, c=aaron_judge.type, cmap=plt.cm.coolwarm, alpha=0.25)
plt.scatter(jose_altuve.plate_x, jose_altuve.plate_z, c=jose_altuve.type, cmap=plt.cm.coolwarm, alpha=0.25)
plt.scatter(david_ortiz.plate_x, david_ortiz.plate_z, c=david_ortiz.type, cmap=plt.cm.coolwarm, alpha=0.25)

#Create a train_test_split to separate training and validation set data with split equal to solution code
training_set, validation_set = train_test_split(aaron_judge, random_state=1)
training_set2, validation_set2 = train_test_split(jose_altuve, random_state=1)
training_set3, validation_set3 = train_test_split(david_ortiz, random_state=1)

#Create and fit classifier SVM to training set columns
classifier = SVC(kernel='rbf', gamma=3, C=1)
classifier2 = SVC(kernel='rbf', gamma=3, C=1)
classifier3 = SVC(kernel='rbf', gamma=3, C=1)
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
classifier2.fit(training_set2[['plate_x', 'plate_z']], training_set2['type'])
classifier3.fit(training_set3[['plate_x', 'plate_z']], training_set3['type'])

#Call CodeCademy function to draw in area of strikezone
draw_boundary(ax, classifier)
draw_boundary(ax, classifier2)
draw_boundary(ax, classifier3)


ax.set_ylim(-2, 6)
ax.set_xlim(-3, 3)
plt.show()


#Scoring models. All above an 83.9% accuracy
print(classifier.score(training_set[['plate_x', 'plate_z']], training_set['type']))
print(classifier2.score(training_set2[['plate_x', 'plate_z']], training_set2['type']))
print(classifier3.score(training_set3[['plate_x', 'plate_z']], training_set3['type']))

