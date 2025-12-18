"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        distances = pairwise_distances(X, self.X_train_)
        neighbor_idx = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbor_labels = self.y_train_[neighbor_idx]

        y_pred = np.array([
            np.unique(labels, return_counts=True)[0][
                np.argmax(np.unique(labels, return_counts=True)[1])
            ]
            for labels in neighbor_labels
        ])

        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        if self.time_col == "index":
            dates = pd.Series(X.index)
        else:
            dates = X[self.time_col]

        if not np.issubdtype(dates.dtype, np.datetime64):
            raise ValueError("Dates must be datetime")

        months = dates.dt.to_period("M")
        return len(months.unique()) - 1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        if not hasattr(X, "index"):
            raise TypeError("X must be a pandas DataFrame")

        if self.time_col == "index":
            dates = pd.Series(X.index)
        else:
            dates = X[self.time_col]

        if not np.issubdtype(dates.dtype, np.datetime64):
            raise ValueError("Dates must be datetime")

        # Sort by date
        sorted_idx = np.argsort(dates.values)
        sorted_dates = dates.iloc[sorted_idx]

        months = sorted_dates.dt.to_period("M")
        unique_months = months.unique()

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_mask = months == train_month
            test_mask = months == test_month

            idx_train = sorted_idx[train_mask.values]
            idx_test = sorted_idx[test_mask.values]

            yield idx_train, idx_test
