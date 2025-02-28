import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from matplotlib import pyplot as plt
path = r"path"
data = pd.read_csv(path, encoding='latin1')
#print(data.describe())
#print(data.columns)

newData = np.array([[8, 7, 6, 9, 8, 7, 9, 8, 7, 30, 25, 1],
                     [5, 6, 4, 7, 6, 5, 8, 7, 6, 28, 24, 0]])


features = ['prob', 'attr', 'sinc', 'intel', 'fun', 'amb', 'shar','age', 'age_o', 'samerace', 'goal', 'go_out']
target = 'match'

dataDropNA = data[features + [target]].dropna()
dataArray = dataDropNA.to_numpy()
X = dataArray[:,:-1]
y = dataArray[:,-1] 

scaler = StandardScaler()
X = scaler.fit_transform(X)
newData = scaler.transform(newData)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=93) #187/2=93

#Classification Tree
classifierTree = DecisionTreeClassifier(random_state = 1234)
modelTree = classifierTree.fit(X_train, y_train)
#Tree plot
plt.figure(figsize=(15, 15))
tree.plot_tree(modelTree, filled=True, feature_names=features, class_names=['No Match', 'Match'])
plt.title('Classification Tree')
plt.show()

#Feature importance plot Tree
importance = modelTree.feature_importances_
feature_importance = pd.Series(importance, index = features)
feature_importance.plot(kind='bar', color='pink')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Classification Tree')
plt.show()
#Post Pruning
print(f"Score Tree Model Train - {modelTree.score(X_train, y_train):.5f}") #100
print(f"Score Tree Model Test - {modelTree.score(X_test, y_test):.5f}") #75
grid = {"max_depth": (5, 10, 15, 20),
        "min_samples_split": (5, 10, 15),
        "min_samples_leaf":(5, 10, 15)}

gcv = GridSearchCV(estimator = classifierTree, param_grid = grid)
gcv.fit(X_train, y_train)

modelTreeGCV = gcv.best_estimator_
print(f"Decision Tree Classifier - {modelTreeGCV}")
modelTreeGCV.fit(X_train, y_train)
print(f"Score Tree Model Train after GCV - {modelTreeGCV.score(X_train, y_train):.5f}") #83
print(f"Score Tree Model Test after GCV- {modelTreeGCV.score(X_test, y_test):.5f}") #82

plt.figure(figsize=(15, 15)) #lower figsize?
tree.plot_tree(modelTreeGCV, filled=True, feature_names=features, class_names=['No Match', 'Match'], max_depth = 1)
plt.title('Classification Tree after GCV')
plt.show()

predTree = modelTreeGCV.predict(newData)
print(f"Prediction for new data with the use of Classification Tree - {predTree}") 

