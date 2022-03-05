import math
from typing import Callable, List


class SparseTable:
    """A sparse table implements static range queries in constant time after preprocessing.
    The query-function supplied must be idempotent, such as min or max. The result of applying
    the query-function on all ranges with power-of-2 length is precomputed, requiring
    O(n log n) time and space to initialize."""

    def __init__(self, arr: List[int], func: Callable[[int, int], int]) -> None:
        self.max_n = len(arr)
        self.func = func
        self.num_bits = self._log2_floor(self.max_n)
        self.table = [[0 for _ in range(self.num_bits + 1)] for _ in range(self.max_n)]
        self._fill_table(arr)

    def _fill_table(self, arr: List[int]) -> None:
        """Initialize the table's values in O(n log n) time and space."""
        for i in range(self.max_n):
            self.table[i][0] = arr[i]

        for j in range(1, self.num_bits + 1):
            shift = 1 << (j - 1)
            for i in range(0, self.max_n - (1 << j) + 1):
                self.table[i][j] = self.func(
                    self.table[i][j - 1], self.table[i + shift][j - 1]
                )

    def _log2_floor(self, x: int) -> int:
        """Helper function for log2. Can be cached for performance optimization."""
        return math.floor(math.log2(x))

    def range_query(self, left: int, right: int) -> int:
        """Return the result of applying the table's range function to [left, right]"""
        j = self._log2_floor(right - left + 1)
        return self.func(self.table[left][j], self.table[right - (1 << j) + 1][j])
