import math
from typing import Callable, List


class SparseTable:

    def __init__(self, arr: List[int], func: Callable[[int, int], int]):
        self.max_n = len(arr)
        self.func = func
        self.K = self._log2_floor(self.max_n)
        self.table = [[0 for _ in range(self.K + 1)] for _ in range(self.max_n)]
        self._fill_table(arr)

    def _fill_table(self, arr: List[int]):
        for i in range(self.max_n):
            self.table[i][0] = arr[i]

        for j in range(1, self.K + 1):
            shift = 1 << (j - 1)
            for i in range(0, self.max_n - (1 << j) + 1):
                self.table[i][j] = self.func(
                    self.table[i][j - 1], self.table[i + (1 << shift)][j - 1])

    def _log2_floor(self, x: int) -> int:
        return math.floor(math.log2(x))

    def range_func(self, left: int, right: int) -> int:
        j = self._log2_floor(right - left + 1)
        return self.func(self.table[left][j],
                         self.table[right - (1 << j) + 1][j])
