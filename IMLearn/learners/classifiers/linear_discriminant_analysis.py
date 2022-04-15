from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, nks = np.unique(y, return_counts=True)
        self.pi_ = nks / y.size
        
        self.mu_ = np.empty((self.classes_.size, X.shape[1]))
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for cls, idx in enumerate(self.classes_):
            X_group = X[y==cls, :]
            self.mu_[idx, :] = np.sum(X_group, axis=0) / nks[idx]
            centered_X_group = X_group - self.mu_[idx, :]
            self.cov_ = self.cov_ + ((centered_X_group.T @ centered_X_group) / y.size)
        
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        aks = self._cov_inv @ self.mu_.T
        bks = np.log(self.pi_) - 0.5 * np.diag(self.mu_ @ self._cov_inv @ self.mu_.T)
        predictions = X @ aks + bks
        return self.classes_[np.argmax(predictions)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        const = 1 / np.sqrt(np.pow(2*np.pi, X.shape[1]) * np.norm(self.cov_))
        likelihood_mat = np.empty((X.shape[0], self.classes_.size))
        for i in range(X.shape[0]):
            centered = X[i, :] - self.mu_
            exp_arg = -0.5 * np.diag(centered.T @ self._cov_inv @ centered)
            total_prob = const * np.exp(exp_arg)
            likelihood_mat[i] = total_prob
        return likelihood_mat

            

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X), normalize=True)
