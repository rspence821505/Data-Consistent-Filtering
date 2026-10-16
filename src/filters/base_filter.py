"""
Abstract base class for data assimilation filters.
"""

from abc import ABC, abstractmethod


class BaseFilter(ABC):
    """
    Abstract base class for ensemble-based data assimilation filters.
    """

    def __init__(self, H, R):
        """
        Initialize filter.

        Parameters
        ----------
        H : ndarray, shape (m, n)
            Observation operator.
        R : ndarray, shape (m, m)
            Observation error covariance.
        """
        self.H = H
        self.R = R

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Perform analysis update.

        Must be implemented by subclasses.
        """
        pass
