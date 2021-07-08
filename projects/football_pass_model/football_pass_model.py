######## Libraries ########
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings(action='ignore')
pd.set_option("max_columns", 50)






####### Data cleaning #########

# Read in data
df = pd.read_csv("../projects/football_pass_model/Sample_Game_1/Sample_Game_1_RawEventsData.csv")
print(df) 

# Print types and subtypes
print('Types: ',df.Type.unique(),'\n')
print('Subtypes: ',df.Subtype.unique())

# Look at pass and ball lost types
df_pass = df[df['Type'] =='PASS']
df_lost = df[df['Type'] =='BALL LOST']
print('Pass Subtypes: ',df_pass.Subtype.unique(),'\n')
print('Ball Lost Subtypes: ',df_lost.Subtype.unique())

# Trim ball lost to just relevant subtypes
condition = (df_lost.Subtype != '') & (df_lost.Subtype != ' ') & (df_lost.Subtype != 'THEFT') &  (df_lost.Subtype != 'HEAD') & (df_lost.Subtype != 'HEAD-FORCED') & (df_lost.Subtype != 'OFFSIDE') &  (df_lost.Subtype != 'FORCED') & (df_lost.Subtype != 'END HALF') & (df_lost.Subtype != 'WOODWORK') & (df_lost.Subtype != 'REFEREE HIT')
df_lost_trimmed = df_lost[condition]
df_lost_trimmed = df_lost_trimmed.dropna(subset=['Subtype'])
print('Pass Subtypes: ',df_pass.Subtype.unique(),'\n')
print('Ball Lost Subtypes: ',df_lost_trimmed.Subtype.unique())

# Put pass and ball lost data abck together
pass_data = pd.concat([df_pass, df_lost_trimmed])

# Extra cleaning of various columns
pass_data.rename(columns={'Type': 'pass_sucess'}, inplace=True)
pass_data["pass_sucess"].replace({"PASS": 1, "BALL LOST": 0}, inplace=True)
pass_data.dropna(subset = ["End X"], inplace=True)
pass_data["Subtype"].fillna("STANDARD", inplace=True)
pass_data["Subtype"].replace({"INTERCEPTION": "STANDARD"}, inplace=True)
pass_data["Subtype"].replace({"HEAD-INTERCEPTION": "HEAD"}, inplace=True)
pass_data["Subtype"].replace({"THROUGH BALL-DEEP BALL": "DEEP BALL"}, inplace=True)
pass_data["Subtype"].replace({"CROSS-INTERCEPTION": "CROSS"}, inplace=True)
pass_data["Subtype"].replace({"HEAD-CLEARANCE": "CLEARANCE"}, inplace=True)
pass_data["Subtype"].replace({"GOAL KICK-INTERCEPTION": "GOAL KICK"}, inplace=True)
pass_data["Team"].replace({"Away": 0}, inplace=True)
pass_data["Team"].replace({"Home": 1}, inplace=True)
pass_data.rename(columns={'Team': 'home_team'}, inplace=True)
pass_data["Period"].replace({2: 0}, inplace=True)
pass_data.rename(columns={'Period': 'first_half'}, inplace=True)
print(pass_data)







########### Data exploration ############
# Print distributions of features
pass_data.hist(figsize=(30,20))
plt.show()
pass_data['Subtype'].value_counts().plot(kind='bar')
plt.show()
pass_data['From'].value_counts().plot(kind='bar')
plt.suptitle("'From' feature distribution")
plt.show()
pass_data['To'].value_counts().plot(kind='bar')
plt.suptitle("'To' feature distribution")
plt.show()







####### Feature engineering ##########
# Remove redundant feartures
pass_data.drop(['Start Frame', 'End Frame', 'To'], axis=1, inplace=True)

# Create new pass_length feature
new_column = pass_data["End Time [s]"] - pass_data["Start Time [s]"]
pass_data["pass_length"] = new_column
pass_data.drop(['End Time [s]'], axis=1, inplace=True)
print(pass_data)

# Split into feature set and dependent variable
x = pass_data.drop(['pass_sucess'], axis=1)
y = pass_data['pass_sucess']

# Deal with categorical variables
cat_features_string = ['Subtype','From']
cat_features = [x.Subtype, x.From]
for i in range(len(cat_features_string)):
    x_temp = pd.get_dummies(cat_features[i], prefix=cat_features_string[i])
    x = x.drop(cat_features_string[i],axis=1)
    x = x.join(x_temp)
print(x)









########## Model selection ###########
# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Oversample and undersample
rus = RandomUnderSampler()
x_train, y_train = rus.fit_resample(x_train, y_train)
x_train, y_train = SMOTE().fit_resample(x_train, y_train)

# Show new, sampled distribiution
y_train.hist(figsize=(10,10))
plt.suptitle("New distribution of failed/sucessful passes")
plt.show()

# Define the models to be used
log_regression = linear_model.LogisticRegression()
n_bayes = naive_bayes.GaussianNB()
sgd = linear_model.SGDClassifier()
knn = neighbors.KNeighborsClassifier()
rand_forest = ensemble.RandomForestClassifier(n_jobs=-1)
svc = svm.SVC()
models = [log_regression, n_bayes, sgd, knn, rand_forest, svc]
model_names = ['Logistic Regression', 'Naive Bayes', 'Stochastic Gradient Descent','K-Nearest-Neighbours', 'Random Forest', 'Support Vector Classifier']

# Use cross validation to get some preliniary model scores
print('F1:')
for i in range(len(models)):
    kfold = KFold(n_splits=7)
    result = cross_val_score(models[i], x_train, y_train, cv=kfold, scoring='f1')
    print(model_names[i] + ': ' + str(result.mean()) )

print('\nAccuracy:')
for i in range(len(models)):
    kfold = KFold(n_splits=7)
    result = cross_val_score(models[i], x_train, y_train, cv=kfold, scoring='accuracy')
    print(model_names[i] + ': ' + str(result.mean()) )

print('\nBalanced Accuracy:')
for i in range(len(models)):
    kfold = KFold(n_splits=7)
    result = cross_val_score(models[i], x_train, y_train, cv=kfold, scoring='balanced_accuracy')
    print(model_names[i] + ': ' + str(result.mean()) )

# Define parametres to be hypertuned on best model
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
            }

# Hypertune to find best parameteres
grid = GridSearchCV(rand_forest, param_grid = parameters, scoring='balanced_accuracy', verbose=1)
grid.fit(x_train, y_train)
print(grid.best_params_)  

# Use hypertuned model
rand_forest = ensemble.RandomForestClassifier(criterion = 'entropy', max_features='sqrt', n_jobs=-1)
rand_forest.fit(x_train,y_train)
predictions = rand_forest.predict(x_test)
accuracy = metrics.balanced_accuracy_score(y_test,predictions)
f1 = metrics.f1_score(y_test,predictions)
print('Balanced Accuracy: '+str(accuracy))
print('F1 Score: '+str(f1))
probs = rand_forest.predict_proba(x_test)
print(probs[42])









########## Model evaluation ############
# Confusion matrix and ROC curve
c_matrix = metrics.plot_confusion_matrix(rand_forest, x_test,y_test)
metrics.plot_roc_curve(rand_forest, x_test,y_test)
plt.show()

# Fetaure importances
feature_importances = rand_forest.feature_importances_
feature_strings = x_train.columns.tolist()
fig = plt.figure(figsize=[15,10])
plt.xticks(rotation=90)
plt.bar(feature_strings, feature_importances)
plt.show()