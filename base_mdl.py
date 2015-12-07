import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


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

    preds = mdl.predict(X_test)
    preds = [min(max(1, int(round(pred))), 8) for pred in preds]

    sub = pd.DataFrame({'Id': test['Id'], 'Response': preds})
    sub.to_csv('submissions/xgb.csv', index=False)


if __name__ == '__main__':
    main()
