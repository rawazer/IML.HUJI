from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        thr_err = 1
        for j in range(X.shape[1]):
            cur_thr_1, cur_thr_err_1 = self._find_threshold(X[:, j], y, 1)
            if cur_thr_err_1 < thr_err:
                thr_err = cur_thr_err_1
                self.threshold_ = cur_thr_1
                self.j_ = j
                self.sign_ = 1
            cur_thr_2, cur_thr_err_2 = self._find_threshold(X[:, j], y, -1)
            if cur_thr_err_2 < thr_err:
                thr_err = cur_thr_err_2
                self.threshold_ = cur_thr_2
                self.j_ = j
                self.sign_ = 1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        responses = np.empty((X.shape[0]))
        responses[X[:, self.j_] < self.threshold_] = -1*self.sign_
        responses[X[:, self.j_] >= self.threshold_] = self.sign_
        return responses

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        thr, amt = 0, 0
        it = np.nditer(values, flags=["c_index"])
        for cur_thr in it:
            cur_amt = 0
            lower_labels = labels[values < cur_thr]
            upper_labels = labels[values >= cur_thr]
            cur_amt += lower_labels[lower_labels == -1*sign].size
            cur_amt += upper_labels[upper_labels == sign].size
            if cur_amt > amt:
                amt = cur_amt
                thr = cur_thr.item()
        N = labels.size
        thr_err = (N - amt) / N
        return thr, thr_err

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
