from settings import *
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

# check/create path for saving output
dataset = 'UCSDped1'
save_path = os.path.join('./results', dataset)
create_video_of_plot = False
# path for saving movie frames
movie_save_dir = os.path.join(save_path, 'movie_frames')
if not os.path.exists(movie_save_dir):
    os.makedirs(movie_save_dir)

df = pd.read_pickle(os.path.join(save_path, 'df.pkl.gz'))

# create feature set
# df.model_std = df.model_std.rolling(10).mean() 
# df = df.fillna(0)
X = df.loc[:,['model_std']] #'model_mse', 'model_p_50', 'model_p_75', 'model_p_90', 'model_p_95', 'model_p_99', 'model_std']]
# alternatively, for comparison, you could also use the trivial solution of using the previous frame for predicting the current frame
# X = df.loc[:,['prev_mse', 'prev_p_50', 'prev_p_75', 'prev_p_90', 'prev_p_95', 'prev_p_99', 'prev_std']]

y = df.anom.values.astype('bool')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

clf = LogisticRegression() #C=1e-1, solver='saga', multi_class='auto', fit_intercept=True, tol=1e-10)
#clf = RandomForestClassifier(n_estimators=50)

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))

pipeline = Pipeline([('scaler',scaler), ('classifier', clf)])

pipeline.fit(X_train, y_train)

print("train accuracy: ", pipeline.score(X_train, y_train))

if create_video_of_plot:
    n_frames = 200
    y_prob = pipeline.predict_proba(X)
    for frame in range(n_frames):
        sns.set_context("talk")
        sns.set_style("white")
        fig = plt.figure(figsize=(20,3))
        plt.plot(y[:n_frames], label="Data", linewidth=5)   
        plt.plot(y_prob[:frame,1], label="Prediction", linewidth=5)
        plt.legend(loc=2, frameon=False, fontsize=24)
        plt.ylabel("Probability", fontsize=24)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        sns.despine()   
        plt.savefig(os.path.join(movie_save_dir,  'plot_%03d.png' % (frame)))
        plt.close()


y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("test accuracy: ", acc)


roc = roc_auc_score(y_test, y_pred)
print("test ROC: ", roc)

precision = precision_score(y_test, y_pred)
print("test precision: ", precision)

recall = recall_score(y_test, y_pred)
print("test recall: ", recall)

cm = confusion_matrix(y_test, y_pred)
print("comnfusion Matrix:\n", cm)

fitted_forest = pipeline.steps[1][1]
importances = fitted_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in fitted_forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

ranked_features = []
for f in range(X_train.shape[1]):
    feature = indices[f]
    feature_name = X_train.columns[feature]
    ranked_features.append(feature_name)
    print("%d. feature %s (%d, %f)" % (f + 1, feature_name, indices[f], importances[indices[f]]))
print(X_train.columns)

import matplotlib.pyplot as plt

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="grey", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), labels=ranked_features)
plt.xlim([-1, X_train.shape[1]])
plt.show()

print("Done")
# print(clf.n_iter_)
# print(clf.coef_)
# print(clf.intercept_)


# plt.scatter(df.mse_prev, df.mse_model)
# plt.show()

