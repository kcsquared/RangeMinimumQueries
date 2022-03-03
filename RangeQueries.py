from functools import partial
import math
from typing import List, Tuple

from SparseTable import SparseTable



class AlternateMinMax:

    def __init__(self, arr: List[int]):
        self.arr = arr[:]
        self.n = len(self.arr)

        evens = self.arr[0::2]
        odds = self.arr[1::2]

        self.even_upper_bounds = SparseTable(arr=evens, func=min)
        self.even_lower_bounds = SparseTable(arr=evens, func=max)

        self.odd_upper_bounds = SparseTable(arr=odds, func=min)
        self.odd_lower_bounds = SparseTable(arr=odds, func=max)

    def query(self, left: int, right: int) -> int:
        # Inclusive [left, right]

        if left > right:
            raise ValueError

        if left == right:
            return self.arr[left]

        left_is_even = (left % 2 == 0)

        if left_is_even:
            up_bounds_table = self.even_upper_bounds
            low_bounds_table = self.odd_lower_bounds

        else:
            up_bounds_table = self.odd_upper_bounds
            low_bounds_table = self.even_lower_bounds
            
        # 1 longer than low_table_len iff R-L+1 = num_elements odd
        up_table_effective_len = (right - left + 2) // 2  
        low_table_effective_len = up_table_effective_len - ((right - left + 1) % 2)

        up_table_start_idx = self._main_idx_to_parity_idx(left)
        low_table_start_idx = self._main_idx_to_parity_idx(left + 1)


        up_func = self._index_and_table_to_func(start_idx=up_table_start_idx,
                                                effective_len=up_table_effective_len,
                                                table_to_use=up_bounds_table)

        low_func = self._index_and_table_to_func(start_idx=low_table_start_idx,
                                                 effective_len=low_table_effective_len,
                                                 table_to_use=low_bounds_table)

        spot = self._binary_search(decreasing_func_calc=up_func,
                                           increasing_func_calc=low_func,
                                           my_len=up_table_effective_len)

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


    def _index_and_table_to_func(self, start_idx: int,
     effective_len: int, 
     table_to_use: SparseTable) -> partial[int]:
        def special_func(x: int, start: int, eff_len: int, table_) -> int:
            if x == eff_len:
                return table_.range_func(left=start, right=start + eff_len - 1)
            else:
                return table_.range_func(left=start, right=start + x)

        return partial(special_func, start=start_idx, eff_len=effective_len,
                                 table_=table_to_use)
                                 

    def _binary_search(self, decreasing_func_calc, increasing_func_calc, my_len: int) -> int:
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

    def _main_idx_to_parity_idx(self, main_idx: int) -> int:
        return main_idx // 2


class BruteForceMinMax:

    def __init__(self, arr: List[int]):
        self.arr = arr[:]
        self.n = len(self.arr)

    def _recurse(self, idx: int, lower: int, upper: int) -> int:
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

        return self._recurse(left, left, right)