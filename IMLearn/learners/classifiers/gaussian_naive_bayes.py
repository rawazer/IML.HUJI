from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = np.zeros((self.classes_.size, X.shape[1], X.shape[1]))
        for cls, idx in enumerate(self.classes_):
            X_group = X[y==cls, :]
            self.mu_[idx, :] = np.sum(X_group, axis=0) / nks[idx]
            centered_X_group = X_group - self.mu_[idx, :]
            self.vars_[idx, :, :] = (centered_X_group.T @ centered_X_group) / nks[idx]

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
        likelihood_mat = self.likelihood(X)
        return self.classes_[np.argmax(likelihood_mat, axis=1)]

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

        likelihood_mat = np.empty((X.shape[0], self.classes_.size))
        for k in range(self.classes_.size):
            const = 1 / np.sqrt(np.power(2*np.pi, X.shape[1]) * np.linalg.norm(self.vars_[k, :, :]))
            centered = X - self.mu_[k, :]
            exp_arg = -0.5 * np.diag(centered @ np.linalg.inv(self.vars_[k, :, :]) @ centered.T)
            likelihood_mat[:, k] = const * np.exp(exp_arg)
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
