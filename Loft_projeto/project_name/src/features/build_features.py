from numpy import ndarray
from sklearn.base import TransformerMixin, BaseEstimator

class ClippingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_clipped = X.copy()
        X_clipped[X_clipped > self.threshold] = self.threshold
        return X_clipped
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
        