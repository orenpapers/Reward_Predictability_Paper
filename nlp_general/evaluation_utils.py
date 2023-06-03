import scipy as sp

def cosine_similarity(vector1, vector2):
    return 1 - sp.spatial.distance.cosine(vector1, vector2)


def euclidean_similarity(vector1, vector2):
    return 1 - sp.spatial.distance.euclidean(vector1, vector2)


def pearson_correlation(vector1, vector2):
    return sp.pearsonr(vector1, vector2)[0]