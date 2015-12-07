import numpy as np
from sklearn.metrics import confusion_matrix

# def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
#
#     if min_rating==None:
#         min_rating = min(rater_a + rater_b)
#     if max_rating==None:
#         max_rating = max(rater_a + rater_b)
#     number_of_rating = int(max_rating - min_rating) + 1
#     conf_mat = np.zeros((number_of_rating, number_of_rating))
#     for i in range(len(rater_a)):
#         conf_mat[rater_a[i] - 1, rater_b[i] - 1] += 1
#     return conf_mat

def weights(conf_mat):

    length = len(conf_mat)
    w = np.zeros((length, length))
    for i in xrange(length):
        for j in xrange(length):
            w[i, j] = pow(i - j, 2) / float(pow(length - 1, 2))
    return w


def histogram(ratings, min_rating=None, max_rating=None):

    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = np.zeros(num_ratings)
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def expected_rating(rater_a, rater_b, min_rating=None, max_rating=None):
    if min_rating==None:
        min_rating = min(rater_a + rater_b)
    if max_rating==None:
        max_rating = max(rater_a + rater_b)
    number_of_rating = int(max_rating - min_rating) + 1
    exp_rat = np.zeros((number_of_rating, number_of_rating))
    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)
    length_rater = len(rater_a)
    for i in range(number_of_rating):
        for j in range(number_of_rating):
            exp_rat[i, j] = hist_rater_a[i] * hist_rater_b[j] / float(length_rater)
    return exp_rat

def quadratic_weighted_kappa(rater_a, rater_b):

    conf = np.array(confusion_matrix(rater_a, rater_b))
    print conf
    weight = np.array(weights(conf))
    print weight
    exp_rat = np.array(expected_rating(rater_a, rater_b))
    print exp_rat
    return 1 - np.sum(conf * weight) / np.sum(exp_rat * weight)

if __name__ == '__main__':
    rater_a = [1,2,1,1]
    rater_b = [1,1,1,1]
    print quadratic_weighted_kappa(rater_a, rater_b)
