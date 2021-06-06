import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings(action='ignore')

# Read data in and clean
df = pd.read_csv('./kc_house_data.csv')
df.head()
df.info()
df.fillna(0, inplace=True)


# Make date just years, and then create new column for age of house
d =[]
for i in df['date'].values:
    d.append(i[:4])
    
df['date'] = d
df['date']=df['date'].astype(float)

df['age'] = df['date'] - df['yr_built']


# Remove redundant columns
df = df.drop(["date", "id", 'yr_built'],  axis=1)


# Remove highly correlated features
# corr_features =[]

# for i , r in df.corr().iterrows():
#     k=0
#     for j in range(len(r)):
#         if i!= r.index[k]:
#             if r.values[k] >=0.5:
#                 corr_features.append([i, r.index[k], r.values[k]])
#         k += 1

# feat =[]
# for i in corr_features:
#     if i[2] >= 0.8:
#         feat.append(i[0])
#         feat.append(i[1])

# df.drop(list(set(feat)), axis=1, inplace=True)


# # Visualise distributions of columns
# df.hist(figsize=(30,20))
# plt.show()

# Removing outliers
zscore = []
outlier =[]
threshold = 3

price_mean = np.mean(df['price'])
price_std = np.std(df['price'])

for i in df['price']:
        z = (i-price_mean)/price_std
        zscore.append(z)
        if np.abs(z) > threshold:
            outlier.append(i)


# plt.figure(figsize = (10,6))
# sns.distplot(zscore, kde=False)
# plt.axvspan(xmin = -3 ,xmax= min(zscore),alpha=0.2, color='blue', label='Lower Outliers')
# plt.axvspan(xmin = 3 ,xmax= max(zscore),alpha=0.2, color='red', label='Upper Outliers')
# plt.show()

dj=[]
for i in df.price:
    if i in set(outlier):
        dj.append(0.0)
    else:
        dj.append(i)
        
df['P'] = dj

df = df.drop(df[df['P'] == 0.0].index) 
X = df.drop(['price','P'], axis=1)
Y = df['price']

# Training/Test Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

# One Hot Encode
train_zip = pd.get_dummies(x_train.zipcode, prefix="zipcode")
x_train = x_train.drop('zipcode',axis=1)
x_train = x_train.join(train_zip)

test_zip = pd.get_dummies(x_test.zipcode, prefix="zipcode")
x_test = x_test.drop('zipcode',axis=1)
x_test = x_test.join(test_zip)


# Scaling data
scaler = StandardScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


# Define Models
lr = linear_model.LinearRegression()
ridge = linear_model.Ridge()
lasso = linear_model.Lasso()
e_net = linear_model.ElasticNet()
r_forest = RandomForestRegressor()
k_neighbours = KNeighborsRegressor()
d_tree = DecisionTreeRegressor()


models = [lr, ridge, lasso, e_net, r_forest, k_neighbours, d_tree]
model_names = ['Linear', 'Ridge', 'Lasso', 'Elastic Net', 'Random Forest', 'KNN', 'Decision Tree']


# # Validation testing to see which model is best
# for i in range(len(models)):
#     kfold = KFold(n_splits=7)
#     result = cross_val_score(models[i], x_train, y_train, cv=kfold, scoring='r2')
#     print(model_names[i] + ': ' + str(result.mean()) )



# Hyperparmeter tuning on best model
parameters = {
    'n_estimators': [50, 100, 200],
    'criterion': ['mse', 'mae'],
    'max_features': ['auto', 'sqrt', 'log2'],
            }


grid = GridSearchCV(r_forest, param_grid = parameters,n_jobs=-1, scoring='r2', verbose=2)
grid.fit(x_train, y_train)
print(grid.best_score_)
print(grid.best_params_)  


# Use optimised model
r_forest = RandomForestRegressor()
r_forest.fit(x_train,y_train)
predictions = r_forest.predict(x_test)
score = r2_score(y_test,predictions)
print(score)


# Plot 2D regressions to visualise feature relationships
for e in enumerate(x_train.columns):
    r_forest.fit(x_train[e].values[:,np.newaxis], y_train.values)
    plt.title("Best fit line")
    plt.xlabel(str(e))
    plt.ylabel('Price')
    plt.scatter(x_train[e].values[:,np.newaxis], y_train)
    plt.plot(x_train[e].values[:,np.newaxis], r_forest.predict(x_train[e].values[:,np.newaxis]),color='r')
    plt.show()