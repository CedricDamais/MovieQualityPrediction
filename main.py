import pandas as pd
import os
import requests
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD

# Data downloading script

########
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if ('dataset.csv' not in os.listdir('../Data')):
    print('Dataset loading.')
    url = "https://www.dropbox.com/s/0sj7tz08sgcbxmh/large_movie_review_dataset.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/dataset.csv', 'wb').write(r.content)
    print('Loaded.')

# The dataset is saved to `Data` directory
########

"""
We will work with the Large Movie Review Dataset  It has a wide range of metadata, including cast, crew, 
plot keywords, budget, posters, release dates, languages, production companies, and countries in movie reviews. We 
have ratings from 0 to 10 for all films in the training data. Our goal is to build a model for predicting which movie 
will have higher ratings after analyzing the reviews with metadata.

By predicting the ratings from 0 to 10, we can solve the regression problem. To do such a task since there is no 
difference between movies with ratings of 2.72 or 3.14 we will tag movies with ratings >7 as good and with movie ratings
< 5 as " bad" other data isn't relevant. So the problem becomes a classification problem.
"""


def delete_row(data, condition: str) -> None:
    """

    """
    data = data.drop(data[(data['ratings'] <= 5 & (data['rating'] <= 7))])


def split_data(data, target_col, X_col):
    """

    """
    y = data[target_col]
    X = data[X_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)
    return X_train, X_test, y_train, y_test


def make_predictions(model: sklearn.linear_model, X_train, y_train, X_test, y_test, model_name: str) -> None:
    model.fit(X=X_train, y=y_train)
    preds = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)
    acc = accuracy_score(y_true=y_test, y_pred=preds)
    AUC = roc_auc_score(y_true=y_test, y_score=pred_probs[:, 1])
    print(f"The accuracy score for {model_name} is : {acc}")
    print(f"The AUC score for {model_name} is : {AUC}")


def get_cons_features(model: sklearn.linear_model) -> int:
    lst = []
    for coef in model.coef_[0]:
        if abs(coef) > 0.0001 and coef is not None:
            lst.append(coef)
    return len(lst)


def main():
    # write your code here
    df = pd.read_csv('../Data/dataset.csv')

    # Dropping all the movies with irrelevant data to us
    df.drop(df[(df['rating'] >= 5) & (df['rating'] <= 7)].index, inplace=True)

    # Adding binary column label
    # 1 if rating > 7 and 0 if rating < 5

    df['label'] = 0
    df.loc[df['rating'] > 7, "label"] = 1

    # We no longer need the rating column since we already have the binary col
    df.drop('rating', inplace=True, axis=1)
    good_movies_col = df['label'].value_counts()[0]

    # print("Proportion of 1 in the dataset : " + str(round(good_movies_col / df['label'].count(), 1)))

    # Transform our words into a feature matrix using the bag-of-words model

    # Split the data into training  and test samples
    X_train, X_test, y_train, y_test = split_data(data=df, target_col='label', X_col='review')
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    training_ft_matrix = vectorizer.fit_transform(X_train)
    tst_ft_matrix = vectorizer.transform(X_test)
    terms = vectorizer.get_feature_names_out()
    print(f"The amount of terms {len(terms)}")

    # It's time to solve the classification problem using a logistic regression model
    # We Will add two metrics to the model Accuracy and AUC ( Area under the curve )
    # AUC is the measure of a classifier's ability to distinguish between classes
    # predict_proba returns probabilities for each class.
    # To calculate the AUC score you would need probabilities for class 1

    model_0 = LogisticRegression(solver='liblinear')
    make_predictions(model=model_0,
                     X_train=training_ft_matrix,
                     y_train=y_train,
                     X_test=tst_ft_matrix,
                     y_test=y_test, model_name='model_0')

    # Attempt to make my model perform better on this task
    # Because You can guess that there are a lot of extra words in our bag
    # and a lot of extra features in our feature matrix do not improve the model's performance
    # To find the extra features we will use the L1-regularization method
    # The L1-regularization sets coefficients of extra features to null, or sometimes near null

    model_1 = LogisticRegression(solver='liblinear', penalty='l1', C=0.15)
    make_predictions(model=model_1,
                     X_train=training_ft_matrix,
                     X_test=tst_ft_matrix,
                     y_test=y_test,
                     y_train=y_train,
                     model_name='model_1')

    considered_feat_mod1 = round(get_cons_features(model_1), -2)
    # considered_feat_mod0 = get_cons_features(model_0)
    # print(f"In model_1 considered features : {considered_feat_mod1}")
    # print(f"In model_0 considered features : {considered_feat_mod0}")

    # From Model_0 to model_1 we dropped in terms of accuracy and AUC of about 6%
    # But the amount of features in model_1 is considerably smaller compared to model_0

    model_2 = LogisticRegression(solver='liblinear')
    SVD = TruncatedSVD(n_components=considered_feat_mod1)
    SVD_training_ft_matrix = SVD.fit_transform(training_ft_matrix)
    SVD_test_ft_matrix = SVD.transform(tst_ft_matrix)

    make_predictions(model=model_2,
                     X_train=SVD_training_ft_matrix,
                     y_train=y_train,
                     X_test=SVD_test_ft_matrix,
                     y_test=y_test,
                     model_name='model_2')
    # We can see that using the SVD method made our model much better in terms of predictions
    # and in terms of computing speed


if __name__ == '__main__':
    main()
