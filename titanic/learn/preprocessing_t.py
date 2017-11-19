from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def best_model(X, target):
    clf = RandomForestClassifier(n_estimators=100)
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10,20,None),'max_features':('sqrt','log2',None)}
    grid_search = GridSearchCV(clf, parameters, n_jobs=-1)
    grid_search.fit(X, target)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
