import numpy 

"""
    input :
        embeddings: (n_samples, n_features)
        
    returns:
        anomalies           : (n_anomalies,) - indices of anomalous items
        mahalanobis_dist    : (n_samples,)

    - Well-defined mean (central tendency)
    - Well-defined covariance structure
    - Assumes features are in a metric space where Euclidean distance is meaningful
"""
def z_score_anomaly_detection(embeddings, sigma_th = 3):
    # compute mahalanobis distance
    mean = numpy.mean(embeddings, axis=0)
    covariance = numpy.cov(embeddings.T)
    inv_covmat = numpy.linalg.inv(covariance)
    
    mahalanobis_dist = []
    for x in embeddings:
        diff = x - mean
        dist = numpy.sqrt(diff.dot(inv_covmat).dot(diff))
        mahalanobis_dist.append(dist)
    
    # z-score threshold
    threshold = numpy.mean(mahalanobis_dist) + sigma_th * numpy.std(mahalanobis_dist)
    anomalies = numpy.where(mahalanobis_dist > threshold)[0]
    
    return anomalies, mahalanobis_dist