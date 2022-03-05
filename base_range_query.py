from abc import ABC, abstractmethod


class BaseRangeQuery(ABC):
    """A range query on an array A takes indices i <= j as input and returns
    the result of apply a function f to A[i:j+1]. Using precomputation, these
    queries can be answered more efficiently."""

    @abstractmethod
    def range_query(self, left: int, right: int) -> int:
        """Return the result of applying the range function to [left, right]"""
        pass
