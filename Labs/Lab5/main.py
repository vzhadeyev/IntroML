from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import utils

#-------------------------------------------------------------------------------
# Part 1 - Data Loading
#-------------------------------------------------------------------------------

print("Loading data")
data = load_breast_cancer()
X = data['data']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print()

#-------------------------------------------------------------------------------
# Part 2 - Decision Tree Classifier
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 2.1 - Building and Training a Decision Tree Classifier
#-------------------------------------------------------------------------------

print("Training the Decision Tree Classifier")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
print()

#-------------------------------------------------------------------------------
# Part 2.2 - Evaluating the Decision Tree Classifier
#-------------------------------------------------------------------------------
print("Evaluating the Decision Tree Classifier")
pred_train = clf.predict(X_train)
print(accuracy_score(y_true=y_train, y_pred=pred_train))
pred_test = clf.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=pred_test))
print()

#-------------------------------------------------------------------------------
# Part 2.3 - Interpreting the Decision Tree Classifier
#-------------------------------------------------------------------------------

print("Interpreting the Random Forest Classifier")
print(utils.sort_features(data['feature_names'], clf.feature_importances_))
utils.display_decision_tree(clf, data['feature_names'], data['target_names'])
print()

#-------------------------------------------------------------------------------
# Part 3 - Random Forest Classifier
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Part 3.1 - Building and Training a Random Forest Classifier
#-------------------------------------------------------------------------------

print("Training the Random Forest Classifier")
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print()

#-------------------------------------------------------------------------------
# Part 3.2 - Evaluating the Random Forest Classifier
#-------------------------------------------------------------------------------

print("Evaluating the Random Forest Classifier")
pred_train = clf.predict(X_train)
print(accuracy_score(y_true=y_train, y_pred=pred_train))
pred_test = clf.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=pred_test))
print()

#-------------------------------------------------------------------------------
# Part 3.3 - Interpreting the Random Forest Classifier
#-------------------------------------------------------------------------------

print("Interpreting the Random Forest Classifier")
for tree in clf.estimators_:
    print(utils.sort_features(data['feature_names'], tree.feature_importances_))
    print()
    utils.display_decision_tree(tree, data['feature_names'], data['target_names'])
print()