import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from eval import quadratic_weighted_kappa, histogram
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.cross_validation import KFold

def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    enc = LabelEncoder()
    joined = pd.concat((train['Product_Info_2'],
                        test['Product_Info_2']), axis=0)
    enc.fit(joined)
    train['Product_Info_2'] = enc.transform(train['Product_Info_2'])
    test['Product_Info_2'] = enc.transform(test['Product_Info_2'])


    X_train = train.drop('Response', axis=1).values
    y_train = train['Response'].values
    X_test = test.values

    mdl = xgb.XGBRegressor(learning_rate=0.05,
                           n_estimators=200,
                           subsample=0.5,
                           max_depth=6,
                           silent=False)
    mdl.fit(X_train, y_train)

    train_histogram = histogram(y_train, min_rating=1, max_rating=8)
    cum_histogram = np.cumsum(train_histogram) / len(y_train)
    bins = np.insert(cum_histogram, 0, 0)

    # cross_val_score
    # kappas = []
    # kf = KFold(y_train.shape[0], n_folds=4, shuffle=True, random_state=1)
    # for train_index, test_index in kf:
    #     mdl.fit(X_train[train_index],y_train[train_index])
    #     predictions = mdl.predict(X_train[test_index])
    #     predictions = np.array(predictions)
    #     #preds = [min(max(1, int(round(pred))), 8) for pred in predictions]
    #     indices = np.argsort(predictions)
    #     r = bins * len(test_index)
    #     for i in range(len(r)-1):
    #         low = r[i]
    #         high = r[i+1]
    #         predictions[indices[low:high]] = i + 1
    #     preds = predictions
    #     preds[preds>8] = 8
    #     preds[preds<1] = 1
    #     print len(preds)
    #     actuals = y_train[test_index]
    #     print len(actuals)
    #     kappa = quadratic_weighted_kappa(actuals, preds)
    #     kappas.append(kappa)
    # print np.mean(kappas)

    preds = mdl.predict(X_test)
    preds = np.array(preds)
    indices = np.argsort(preds)
    r = bins * len(X_test)
    for i in range(len(r)-1):
        low = r[i]
        high = r[i+1]
        preds[indices[low:high]] = i + 1
    preds[preds>8] = 8
    preds[preds<1] = 1
    preds = map(int, preds)

    #preds = [min(max(1, int(round(pred))), 8) for pred in preds]

    sub = pd.DataFrame({'Id': test['Id'], 'Response': preds})
    sub.to_csv('xgb.csv', index=False)


if __name__ == '__main__':
    main()
