from sklearn.cross_validation import cross_val_score, train_test_split
import pandas as pd

def parse_model_final(X):
    if "Survived" not in X.columns:
        raise ValueError("target column survived should belong to df")
    target = X["Survived"]
    X['Title'] = X['Name'].map(lambda x: x.split(',')[1].split('.')[0])
    X['Surname'] = X['Name'].map(lambda x: '(' in x)
    X['Cabin'] = X['Cabin'].map(lambda x: x[0] if not pd.isnull(x) else -1)
    to_dummy = X[["Pclass", "Sex", "Title", "Embarked", "Cabin"]]
    for dum in to_dummy:
        split_temp = pd.get_dummies(X[dum], prefix=dum)
        X = X.join(split_temp)
        del X[dum]

    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['is_child'] = X['Age'] <= 8
    to_del = ["PassengerId", "Name", "Survived" , "Ticket"]
    for col in to_del:
        del X[col]
    return X, target

def compute_score(clf, X, y, cv=5):
    """compute score in a classification modelisation.
    clf: classifier
    X: features
    y:target
    """
    xval = cross_val_score(clf, X, y, cv=5)
    print("Accurancy: %0.2f (+/- %0.2f)" % (xval.mean(), xval.std() * 2))
    return xval

