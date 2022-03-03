import functools
import math
from time import perf_counter
import random
from typing import List, Tuple



class SparseTable:

    def __init__(self, arr: List[int], func):
        self.max_n = len(arr)
        self.func = func
        self.K = self._log2_floor(self.max_n)
        self.table = [[0 for _ in range(self.K + 1)] for _ in range(self.max_n)]
        self._fill_table(arr)

    def _fill_table(self, arr: List[int]):
        for i in range(self.max_n):
            self.table[i][0] = arr[i]

        for j in range(1, self.K + 1):
            for i in range(0, self.max_n - (1 << j) + 1):
                self.table[i][j] = self.func(self.table[i][j - 1],
                                             self.table[i + (1 << (j - 1))][j - 1])

    def _log2_floor(self, x: int) -> int:
        return math.floor(math.log2(x))

    def range_func(self, left: int, right: int):
        j = self._log2_floor(right - left + 1)
        return self.func(self.table[left][j],
                         self.table[right - (1 << j) + 1][j])


class AlternateMinMax:

    def __init__(self, arr: List[int]):
        self.arr = arr[:]
        self.n = len(self.arr)

        self.evens = self.arr[0::2]
        self.odds = self.arr[1::2]

        self.even_upper_bnds = SparseTable(arr=self.evens, func=min)
        self.even_lower_bnds = SparseTable(arr=self.evens, func=max)

        self.odd_upper_bnds = SparseTable(arr=self.odds, func=min)
        self.odd_lower_bnds = SparseTable(arr=self.odds, func=max)

    def query(self, left: int, right: int) -> int:
        # Inclusive [left, right]

        if left > right:
            raise ValueError

        if left == right:
            return self.arr[left]

        left_is_even = (left % 2 == 0)

        if left_is_even:
            up_bnds_table = self.even_upper_bnds
            low_bnds_table = self.odd_lower_bnds

        else:
            up_bnds_table = self.odd_upper_bnds
            low_bnds_table = self.even_lower_bnds

        up_table_effective_len = (
                                             right - left + 2) // 2  # 1 longer than low_table_len iff R-L+1 = num_elements odd
        low_table_effective_len = up_table_effective_len - ((right - left + 1) % 2)

        up_table_start_idx = self._main_idx_to_parity_idx(left)
        low_table_start_idx = self._main_idx_to_parity_idx(left + 1)

        # print(f'{left=}, {right=}, {up_table_effective_len=}, {low_table_effective_len=}, {up_table_start_idx=}, {low_table_start_idx=}')

        up_func = self._index_and_table_to_func(start_idx=up_table_start_idx,
                                                effective_len=up_table_effective_len,
                                                table_to_use=up_bnds_table)

        low_func = self._index_and_table_to_func(start_idx=low_table_start_idx,
                                                 effective_len=low_table_effective_len,
                                                 table_to_use=low_bnds_table)

        spot = self._binary_search_complex(decreasing_func_calc=up_func,
                                           increasing_func_calc=low_func,
                                           my_len=up_table_effective_len)

        # print(
        #     f'{spot=}, {up_func(0)=}, {low_func(0)=}')
        # print(f'{up_func(up_table_effective_len-1)=}, {low_func(up_table_effective_len-1)=}')
        # print()

        if spot == up_table_effective_len:
            low = low_func(spot - 1)
            high = up_func(spot - 1)
            if (right - left + 1) % 2 == 0:
                return min(low, high)
            else:
                return max(low, high)
        else:
            high_val = up_func(spot)
            if spot == 0:
                return high_val
            last_low_val = low_func(spot - 1)

            if last_low_val >= high_val:
                return last_low_val
            else:
                return high_val

    def _index_and_table_to_func(self, start_idx: int, effective_len: int,
                                 table_to_use: SparseTable):
        def special_func(x: int, start, eff_len, table_) -> int:
            if x == eff_len:
                return table_.range_func(left=start, right=start + eff_len - 1)
            else:
                return table_.range_func(left=start, right=start + x)

        return functools.partial(special_func, start=start_idx, eff_len=effective_len,
                                 table_=table_to_use)

    def _range_func_main_idxs(self, left_idx: int, right_idx: int, left_is_even: bool,
                              table_to_use: SparseTable):
        # Inclusive [left, right]
        return self._parity_idx_to_main_idx(
                table_to_use.range_func(self._main_idx_to_parity_idx(left_idx),
                                        self._main_idx_to_parity_idx(right_idx)),
                left_is_even)

    def _binary_search_simple(self, decreasing_arr, increasing_arr):
        """Given two m-length arrays, first non-increasing, second non-decreasing,
         find smallest index i s.t. decreasing_arr[i] <= increasing_arr[i],
          or m if no such i exists"""
        m = len(decreasing_arr)
        assert m == len(increasing_arr)

        if decreasing_arr[m - 1] > increasing_arr[m - 1]:
            return m

        low, high = 0, m - 1
        while low < high:
            mid = (low + high) // 2
            if decreasing_arr[mid] <= increasing_arr[mid]:
                high = mid
            else:
                low = mid + 1
        return low

    def _binary_search_complex(self, decreasing_func_calc, increasing_func_calc, my_len: int):
        """Given two my_len-length arrays, first non-increasing, second non-decreasing,
         find smallest index i s.t. decreasing_arr[i] <= increasing_arr[i],
          or my_len if no such i exists"""

        if decreasing_func_calc(my_len - 1) > increasing_func_calc(my_len - 1):
            return my_len

        low, high = 0, my_len - 1
        while low < high:
            mid = (low + high) // 2
            if decreasing_func_calc(mid) <= increasing_func_calc(mid):
                high = mid
            else:
                low = mid + 1
        return low

    def _parity_idx_to_main_idx(self, parity_idx: int, is_even: bool) -> int:
        return 2 * parity_idx + (0 if is_even else 1)

    def _main_idx_to_parity_idx(self, main_idx) -> int:
        return main_idx // 2


class BruteForceMinMax:

    def __init__(self, arr: List[int]):
        self.arr = arr[:]
        self.n = len(self.arr)

    def _recurse(self, idx, lower, upper):
        if idx >= upper:
            return self.arr[idx]
        if (idx - lower) % 2 == 0:
            return min(self.arr[idx], self._recurse(idx + 1, lower, upper))
        else:
            return max(self.arr[idx], self._recurse(idx + 1, lower, upper))

    def query(self, left: int, right: int) -> int:
        # Inclusive [left, right]

        if left > right:
            raise ValueError

        if left == right:
            return self.arr[left]

        return self.recurse(left, left, right)