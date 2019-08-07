import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import pprint
import itertools
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix

'''
This model is dealing with fraud detection. I have implemented a Random Forest Classifier model to try and predict
the frauds. Due to the fact that the data is very imbalanced, I have tried approaching the problem using the following:
1. Undersample the data so it will be balanced and try to predict the undersampled test set
2. Undersample the data so it will be balanced and try to predict the real test set
3. Using the current distribution of the classes and let the model learn from the imbalanced data
'''

# Random seed to replicate output
seed = 42

#CSV file
csv_file = 'C:/Users/Tal/Desktop/Tal/Credit Card Frauds Detection/creditcard.csv'

#Loading the CSV file using pandas
data = pd.read_csv(csv_file)

#Looking how the data looks like
data.head()

data = data.reset_index()

# Descrption of the dataset
def attribute_description(dataframe):
    for feature in dataframe.columns:
        print(dataframe[feature].describe())


attribute_description(data)

# Columns for drawing distributions
columns = data.columns


# Show distributions of the features based on the classes
def draw_distributions(data, columns):
    plt.figure(figsize=(12,8*4))
    gs = gridspec.GridSpec(7, 4)
    for i, cn in enumerate(data[columns]):
        ax = plt.subplot(gs[i])
        sns.distplot(data[cn][data.Class == 1], bins=50)
        sns.distplot(data[cn][data.Class == 0], bins=50)
        ax.set_xlabel('')
        ax.set_title('feature: ' + str(cn))
    plt.show()
    return


draw_distributions(data, columns)


data.describe()


def values_count(dataframe):
    for feature in dataframe.columns:
        print(dataframe[feature].value_counts())


values_count(data)

'''
After small data exploration:
- seems like data is very imbalanced towards non-fraudelent transactions
- looks like that all features are scaled besides the amount of the transaction
'''

# scaling Amount using StandardScaler
data['newAmount'] = scale(data['Amount']).reshape(-1, 1)

# Drop Amount feature since it is not useful anymore
data = data.drop(['Amount'], axis=1)

# amount of columns
print(len(data.columns))

# Show the heatmap with correlations between the features
def heatmap(num_of_columns, dataframe):
    k = num_of_columns #number of variables for heatmap
    corrmat = dataframe.corr()
    cols = corrmat.nlargest(k, 'Class')['Class'].index
    cm = np.corrcoef(dataframe[cols].values.T)
    sns.set(font_scale=.7)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6.5}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return


heatmap(len(data.columns), data)

# Draw some correlations between attributes noticed on heatmap
sns.scatterplot(x='V21', y='newAmount', data=data, color="m")

sns.scatterplot(x='V7', y='newAmount', data=data, color="m")

sns.scatterplot(x='Time', y='V3', data=data, color="m")

# Check if some columns contain NaN values
data.isnull().any()

# Look how the data looks like now
pprint.pprint(data.head())

# Seperating the Target class from the other features
X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']

X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=seed)

# Count the amount of frauds in dataset
frauds_count = len(data[data.Class == 1])

# All indices of the frauds in data
frauds_indices = np.array(data[data.Class == 1].index)

# Selecting all indices of non-fraud class
non_frauds_indices = data[data.Class == 0].index

# Select random samples in order to balance data
random_normal_indices = np.random.choice(non_frauds_indices, frauds_count, replace=False)
random_normal_indices = np.array(random_normal_indices)

# transform the non-frauds to array
non_frauds_indices_array = np.array(non_frauds_indices)

# Append the two undersample classes together
under_samples = np.concatenate([frauds_indices, random_normal_indices])

# Shuffle data
under_sample_data = data.iloc[under_samples, :]

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number: ", len(under_sample_data))

# Split data to Features (X) and Target (y)
X_under = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_under = under_sample_data.ix[:, under_sample_data.columns == 'Class']

X_under = X_under.reset_index()
y_under = y_under
# Remove rows that were present in both X_test in undersample data
X_test = X_test[~X_test.isin(under_sample_data)].dropna().reset_index()
y_true = y_true[~y_true.isin(under_sample_data)].dropna()

# Split the under dataset, 80% training, 20% testing
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_under, y_under, test_size=0.2, random_state = seed)

print("under train data: ", len(X_train_undersample))
print("under test data: ", len(X_test_undersample))
print("Total: ", len(X_train_undersample)+len(X_test_undersample))

# plotting confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.cool):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return

# report which will be used in the grid search
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return


def grid_search(classifier, X, y, parameters):
    clf = classifier
    grid = RandomizedSearchCV(clf, parameters)
    grid.fit(X, y.values.ravel())
    print("GridSearchCV %d candidate parameter settings."
          % (len(grid.cv_results_['params'])))
    report(grid.cv_results_)
    return report(grid.cv_results_)

rf_grid_parameters = {'n_estimators': [10, 50, 70, 100],
                      'criterion': ['mse', 'mae'],
                      'max_depth': [8, 16, 30, 50],
                      'min_samples_split': [2, 3, 5, 10],
                      'min_samples_leaf': [1, 3, 5],
                      "max_features": ['auto', 'sqrt', 'log2', 0.5]}


# Random Forest Model for undersampled data
rand_forest = RandomForestClassifier(random_state=seed)
rf_best_parameters_search = grid_search(rand_forest, X_train_undersample, y_train_undersample, rf_grid_parameters)

# Mean validation score: 0.946 (std: 0.017)
rf_best_parameters = {'min_samples_leaf': 1,
                      'criterion': 'entropy',
                      'max_depth': 8,
                      'max_features': 0.4,
                      'n_estimators': 70
                     }

rand_forest = RandomForestClassifier(min_samples_leaf=1, criterion='entropy', max_depth=8,max_features=0.4, n_estimators=70, random_state=seed)

np.set_printoptions(precision=2)

# Random Forest
# Predictions on under-sample Set Random Forest
rand_forest.fit(X_train_undersample, y_train_undersample.values.ravel())

rf_y_pred = rand_forest.predict(X_test_undersample.values)

# feature importances
rand_forest.feature_importances_

rf_cnf_matrix = confusion_matrix(y_test_undersample, rf_y_pred)


print("Recall metric for RF model in the testing dataset: ", rf_cnf_matrix[1,1]/(rf_cnf_matrix[1,0]+rf_cnf_matrix[1,1]))
recall = recall_score(y_test_undersample, rf_y_pred)
# Recall 0.909
precision = precision_score(y_test_undersample, rf_y_pred)
# Precision 0.978
print('F1 score', 2 * ((precision * recall) / (precision + recall)))
# F1-score 0.942
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(rf_cnf_matrix, classes=class_names, title='Confusion matrix Random Forest')
plt.show()

# Thresholds check
def thresholds(classifier, X_test, y_test):
    y_pred_undersample_proba = classifier.predict_proba(X_test.values)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    plt.figure(figsize=(10,10))

    j = 1
    for i in thresholds:
        y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i

        plt.subplot(3, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)
        np.set_printoptions(precision=2)

        recall = recall_score(y_test, y_test_predictions_high_recall)
        print('i=',i)
        print('recall', recall)
        precision = precision_score(y_test, y_test_predictions_high_recall)
        print('precision', precision)
        print('F1 score', 2 * ((precision * recall) / (precision + recall)))
        print('')
        # Plot non-normalized confusion matrix
        class_names = [0, 1]
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' % i)
    return

thresholds(rand_forest,X_test_undersample, y_test_undersample)

# Best threshold

# threshold = 0.3
# Recall = 0.969
# Precision = 0.923
# F1-score = 0.945

# Real test-set with undersampled train set

rf_y_pred_all = rand_forest.predict(X_test.values)

rf_cnf_matrix_all = confusion_matrix(y_true, rf_y_pred_all)
np.set_printoptions(precision=2)

print("Recall metric for RF model in the whole testing dataset: ", rf_cnf_matrix_all[1,1]/(rf_cnf_matrix_all[1,0]+rf_cnf_matrix_all[1,1]))
recall = recall_score(y_true, rf_y_pred_all)
# Recall 0.963
print("Precision metric in the testing dataset: ", precision_score(y_true, rf_y_pred_all))
precision = precision_score(y_true, rf_y_pred_all)
# Precision 0.051
print('F1 score', 2 * ((precision * recall) / (precision + recall)))
# F1-score 0.097

class_names = [0, 1]
plt.figure()
plot_confusion_matrix(rf_cnf_matrix_all, classes=class_names, title='Confusion matrix Random Forest')
plt.show()

# Thresholds check
thresholds(rand_forest, X_test, y_true)

# For under-sample training-set, real test-set
# threshold 0.9
# F1-score 0.834
# Recall 0.852
# Precision 0.816


# Real data-set
rand_forest = RandomForestClassifier(random_state=seed, min_samples_leaf=1, n_estimators=74, max_depth=8, max_features=0.5)

rand_forest.fit(X_train, y_train.values.ravel())

rf_y_pred_all = rand_forest.predict(X_test.values)

rf_cnf_matrix_all = confusion_matrix(y_true, rf_y_pred_all)
np.set_printoptions(precision=2)

print("Recall metric for RF model in the whole testing dataset: ", rf_cnf_matrix_all[1,1]/(rf_cnf_matrix_all[1,0]+rf_cnf_matrix_all[1,1]))
recall = recall_score(y_true, rf_y_pred_all)
# Recall 0.816
print("Precision metric in the testing dataset: ", precision_score(y_true, rf_y_pred_all))
precision = precision_score(y_true, rf_y_pred_all)
# Precision 0.948
print('F1 score', 2 * ((precision * recall) / (precision + recall)))
# F1-score 0.877

class_names = [0, 1]
plt.figure()
plot_confusion_matrix(rf_cnf_matrix_all, classes=class_names, title='Confusion matrix Random Forest')
plt.show()

# Thresholds check
thresholds(rand_forest, X_test, y_true)

# For real data-set
# threshold 0.9
# F1-score 0.834
# Recall 0.852
# Precision 0.816