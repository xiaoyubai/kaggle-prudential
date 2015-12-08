import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def main():
    # load data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # encode string category
    enc = LabelEncoder()
    joined = pd.concat((train, test), axis=0)
    enc.fit(joined['Product_Info_2'])
    train['Product_Info_2'] = enc.transform(train['Product_Info_2'])
    test['Product_Info_2'] = enc.transform(test['Product_Info_2'])

    categoricals = ['Product_Info_1', 'Product_Info_3', 'Product_Info_5',
                    'Product_Info_6', 'Product_Info_7', 'Employment_Info_2',
                    'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1',
                    'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4',
                    'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
                    'Insurance_History_1', 'Insurance_History_2',
                    'Insurance_History_3', 'Insurance_History_4',
                    'Insurance_History_7', 'Insurance_History_8',
                    'Insurance_History_9', 'Family_Hist_1',
                    'Medical_History_2', 'Medical_History_3',
                    'Medical_History_4', 'Medical_History_5',
                    'Medical_History_6', 'Medical_History_7',
                    'Medical_History_8', 'Medical_History_9',
                    'Medical_History_10', 'Medical_History_11',
                    'Medical_History_12', 'Medical_History_13',
                    'Medical_History_14', 'Medical_History_16',
                    'Medical_History_17', 'Medical_History_18',
                    'Medical_History_19', 'Medical_History_20',
                    'Medical_History_21', 'Medical_History_22',
                    'Medical_History_23', 'Medical_History_25',
                    'Medical_History_26', 'Medical_History_27',
                    'Medical_History_28', 'Medical_History_29',
                    'Medical_History_30', 'Medical_History_31',
                    'Medical_History_33', 'Medical_History_34',
                    'Medical_History_35', 'Medical_History_36',
                    'Medical_History_37', 'Medical_History_38',
                    'Medical_History_39', 'Medical_History_40',
                    'Medical_History_41']

    joined['Medical_History_10'] = joined['Medical_History_10'].fillna(-1)
    train['Medical_History_10'] = train['Medical_History_10'].fillna(-1)
    test['Medical_History_10'] = test['Medical_History_10'].fillna(-1)

    # map categoricals based on average response
    for cat in categoricals:
        unique_vals = joined[cat].unique()
        responses = []
        # get the mean response for each val
        for val in unique_vals:
            response = joined[joined[cat] == val]['Response'].mean(skipna=True)
            # choosing to place categories that don't show up in train as
            # the mean of response
            if response is np.nan:
                response = train['Response'].mean()
            responses.append(response)
        responses = np.array(responses)
        indexes = np.argsort(responses)
        sorted_map = dict(zip(unique_vals[indexes], range(len(unique_vals))))

        train[cat] = train[cat].map(sorted_map).astype('int32')

    train.to_csv('data/train_mapcat.csv', index=False)
    test.to_csv('data/test_mapcat.csv', index=False)


if __name__ == '__main__':
    main()
