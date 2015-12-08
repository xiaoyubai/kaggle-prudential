import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from eval import quadratic_weighted_kappa


def myscore(y_true, y_pred):
    y_pred = np.array([min(8, max(1, int(round(x)))) for x in y_pred])
    return quadratic_weighted_kappa(y_true, y_pred)


def main():
    train = pd.read_csv('data/train_mapcat.csv')
    test = pd.read_csv('data/test_mapcat.csv')

    X_train = train.drop('Response', axis=1).values
    y_train = train['Response'].values
    X_test = test.values

    # params = {'learning_rate': [0.01, 0.03, 0.1, 0.3],
    #           'n_estimators': [50, 125, 300],
    #           'subsample': [0.1, 0.5, 1.0],
    #           'max_depth': [1, 3, 10]}
    #
    # mdl = xgb.XGBRegressor()
    # scorer = make_scorer(myscore)
    # gs = GridSearchCV(mdl, params, cv=5, scoring=scorer, n_jobs=-1)
    # gs.fit(X_train, y_train)
    # print 'Best Params:', gs.best_params_
    # print 'Best Score:', gs.best_score_
    #
    # mdl = gs.best_estimator_
    # preds = mdl.predict(X_test)
    # preds = [min(max(1, int(round(pred))), 8) for pred in preds]
    #
    # sub = pd.DataFrame({'Id': test['Id'], 'Response': preds})
    # sub.to_csv('submissions/xgb.csv', index=False)

    mdl = xgb.XGBRegressor('n_estimators': 300, 'subsample': 1.0,
                           'learning_rate': 0.03, 'max_depth': 10)
    scorer = make_scorer(myscore)
    scores = cross_val_score(mdl, X_train, y_train, cv=5,
                             scoring=scorer, n_jobs=-1)
    print 'CV5 score =', np.mean(scores)

    mdl.fit(X_train, y_train)

    preds = mdl.predict(X_test)
    preds = [min(max(1, int(round(pred))), 8) for pred in preds]

    sub = pd.DataFrame({'Id': test['Id'], 'Response': preds})
    sub.to_csv('submissions/xgb_mapcat.csv', index=False)


if __name__ == '__main__':
    main()
