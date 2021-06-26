# Libraries
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings(action='ignore')
pd.set_option("max_columns", 25)

# Data input
df = pd.read_csv("./heart.csv")
print(df.head())
print(df.info())

df.hist(figsize=(30,20))
plt.show()

sns.countplot(x="target", data=df)
plt.show()

# Remove correlated features
corr_features =[]

for i , r in df.corr().iterrows():
    k=0
    for j in range(len(r)):
        if i!= r.index[k]:
            if r.values[k] >=0.5:
                corr_features.append([i, r.index[k], r.values[k]])
        k += 1

feat =[]
for i in corr_features:
    print(i[2])
    if i[2] >= 0.8:
        feat.append(i[0])
        feat.append(i[1])

df.drop(list(set(feat)), axis=1, inplace=True)

print(df.corr())

# Prep test/train
x = df.drop(['target'], axis=1)
y = df['target']

cat_features_string = ['sex','fbs','restecg','exang','slope','thal']
cat_features = [x.sex,x.fbs,x.restecg,x.exang,x.slope,x.thal]

for i in range(len(cat_features_string)):
    x_temp = pd.get_dummies(cat_features[i], prefix=cat_features_string[i])
    x = x.drop(cat_features_string[i],axis=1)
    x = x.join(x_temp)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

print(x_train.head())

# Normalise data
scaler = StandardScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# Model selection
log_regression = linear_model.LogisticRegression()
n_bayes = naive_bayes.GaussianNB()
sgd = linear_model.SGDClassifier()
knn = neighbors.KNeighborsClassifier()
dec_tree = tree.DecisionTreeClassifier()
rand_forest = ensemble.RandomForestClassifier(n_jobs=-1)
svc = svm.SVC()

models = [log_regression, n_bayes, sgd, knn, dec_tree, rand_forest, svc]
model_names = ['Logistic Regression', 'Naive Bayes', 'Stochastic Gradient Descent', 'K-Nearest-Neighbours', 'Decision Tree', 'Random Forest', 'Support Vector Classifier']

print('Recall:')
for i in range(len(models)):
    kfold = KFold(n_splits=7)
    result = cross_val_score(models[i], x_train, y_train, cv=kfold, scoring='recall')
    print(model_names[i] + ': ' + str(result.mean()) )

print('Precision:')
for i in range(len(models)):
    kfold = KFold(n_splits=7)
    result = cross_val_score(models[i], x_train, y_train, cv=kfold, scoring='precision')
    print(model_names[i] + ': ' + str(result.mean()) )

print('F1:')
for i in range(len(models)):
    kfold = KFold(n_splits=7)
    result = cross_val_score(models[i], x_train, y_train, cv=kfold, scoring='f1')
    print(model_names[i] + ': ' + str(result.mean()) )

print('Accuracy:')
for i in range(len(models)):
    kfold = KFold(n_splits=7)
    result = cross_val_score(models[i], x_train, y_train, cv=kfold, scoring='accuracy')
    print(model_names[i] + ': ' + str(result.mean()) )


# Hyperparameter tuning
parameters = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            }



grid = GridSearchCV(log_regression, param_grid = parameters,n_jobs=-1, scoring='recall', verbose=2)
grid.fit(x_train, y_train)
print(grid.best_score_)
print(grid.best_params_)  


# Final Model
log_regression = linear_model.LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000000, n_jobs=-1)
log_regression.fit(x_train,y_train)
predictions = log_regression.predict(x_test)
accuracy = metrics.accuracy_score(y_test,predictions)
f1 = metrics.f1_score(y_test,predictions)
recall = metrics.recall_score(y_test,predictions)
precision = metrics.precision_score(y_test,predictions)
print('Accuracy: '+str(accuracy))
print('F1 Score: '+str(f1))
print('Recall: '+str(recall))
print('Precision: '+str(precision))


# Model evaluation
c_matrix = metrics.plot_confusion_matrix(log_regression, x_test,y_test)
plt.show()

metrics.plot_roc_curve(log_regression, x_test,y_test)
plt.show()

# Increase recall
log_regression = linear_model.LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000000, n_jobs=-1, C=0.25)
log_regression.fit(x_train,y_train)
predictions = log_regression.predict(x_test)
accuracy = metrics.accuracy_score(y_test,predictions)
f1 = metrics.f1_score(y_test,predictions)
recall = metrics.recall_score(y_test,predictions)
precision = metrics.precision_score(y_test,predictions)
print('Accuracy: '+str(accuracy))
print('F1 Score: '+str(f1))
print('Recall: '+str(recall))
print('Precision: '+str(precision))

c_matrix = metrics.plot_confusion_matrix(log_regression, x_test,y_test)
plt.show()

# Feature analysis
coefs = log_regression.coef_[0]
features = x_train.columns.tolist()

fig = plt.figure(figsize=[30,10])
plt.bar(features, coefs)
plt.show()

