from functools import partial
from typing import Callable, List
from SparseTable import SparseTable


class AlternateMinMax:
    """Support range queries with alternating and interleaved min-max.
    For example, query(L, R) should return
    min(arr[L], max(arr[L+1], min(arr[L+2], ...arr[R])))...)"""

    def __init__(self, arr: List[int]):
        self.arr: List[int] = arr[:]

        # Break array into even and odd indices
        evens: List[int] = self.arr[0::2]
        odds: List[int] = self.arr[1::2]

        self.even_upper_bounds = SparseTable(arr=evens, func=min)
        self.even_lower_bounds = SparseTable(arr=evens, func=max)

        self.odd_upper_bounds = SparseTable(arr=odds, func=min)
        self.odd_lower_bounds = SparseTable(arr=odds, func=max)

    def query(self, left: int, right: int) -> int:
        """Return the alternating min-max."""

        if left > right:
            raise ValueError(f"Queries must have left <= right but {left=} > {right=}")

        if left == right:
            return self.arr[left]

        left_is_even: bool = left % 2 == 0

        if left_is_even:
            up_bounds_table: SparseTable = self.even_upper_bounds
            low_bounds_table: SparseTable = self.odd_lower_bounds

        else:
            up_bounds_table: SparseTable = self.odd_upper_bounds
            low_bounds_table: SparseTable = self.even_lower_bounds

        # 1 longer than low_table_len iff R-L+1 = num_elements odd
        up_table_effective_len: int = (right - left + 2) // 2
        low_table_effective_len: int = up_table_effective_len - ((right - left + 1) % 2)

        up_table_start_idx: int = self._main_idx_to_parity_idx(left)
        low_table_start_idx: int = self._main_idx_to_parity_idx(left + 1)

        up_func: Callable[[int], int] = self._index_and_table_to_func(
            start_idx=up_table_start_idx,
            effective_len=up_table_effective_len,
            table_to_use=up_bounds_table,
        )

        low_func: Callable[[int], int] = self._index_and_table_to_func(
            start_idx=low_table_start_idx,
            effective_len=low_table_effective_len,
            table_to_use=low_bounds_table,
        )

        crossover_spot: int = self._binary_search(
            decreasing_func=up_func,
            increasing_func=low_func,
            my_len=up_table_effective_len,
        )

        # No crossover case: Return min or max of final values, based on parity
        if crossover_spot == up_table_effective_len:
            low = low_func(crossover_spot - 1)
            high = up_func(crossover_spot - 1)
            if (right - left + 1) % 2 == 0:
                return min(low, high)
            else:
                return max(low, high)
        else:
            high_val = up_func(crossover_spot)
            if crossover_spot == 0:
                return high_val
            last_low_val = low_func(crossover_spot - 1)

            if last_low_val >= high_val:
                return last_low_val
            else:
                return high_val

    def _index_and_table_to_func(
        self, start_idx: int, effective_len: int, table_to_use: SparseTable
    ) -> partial[int]:
        """Wrapper that accepts relative indices and allows passing them to a
        query as absolute indices.

        table_to_use: Class instance supporting range queries on some parity array

        relative indices: [start_idx: start_idx + effective_len]

        Returns:
        partial function f with single integer input.
        f(x) is defined as Query(start_idx, start_idx + x)
        Note that this is a 'prefix query' on the relative array
        parity_arr[start_idx: start_idx + effective_len], not on the main array.

        Note: A call f(start_idx + effective_len) will not give an error,
        as might be expected. This detail is currently required as the 'effective lengths'
        of parity arrays may differ by one, and the range minimum and maximum queries
        are required to be well defined by _binary_search on ranges larger than the array.
        As minimum and maximum are monotonic, this is well defined mathematically but
        allows some out-of-bounds calls to the returned function without error.
        """

        def special_func(x: int, start: int, eff_len: int, table_: SparseTable) -> int:
            if x > eff_len:
                raise ValueError(
                    f"Query for {x} out of bounds for range [{start}, {start+eff_len-1}]"
                )

            if x == eff_len:
                return table_.range_query(left=start, right=start + eff_len - 1)
            else:
                return table_.range_query(left=start, right=start + x)

        return partial(
            special_func, start=start_idx, eff_len=effective_len, table_=table_to_use
        )

    def _binary_search(
        self,
        decreasing_func: Callable[[int], int],
        increasing_func: Callable[[int], int],
        my_len: int,
    ) -> int:
        """Given:
        (weakly) decreasing function decreasing_func,
        (weakly) increasing function increasing_func,
        both taking integers in [0, my_len-1] and returning integers.

        Returns:
        First 'crossover' point:
        The smallest index 0 <= i < my_len
        such that decreasing_func(i) <= increasing_func(i).

        If no such index exists, returns my_len."""

        # No crossover
        if decreasing_func(my_len - 1) > increasing_func(my_len - 1):
            return my_len

        low, high = 0, my_len - 1
        while low < high:
            mid = (low + high) // 2
            if decreasing_func(mid) <= increasing_func(mid):
                high = mid
            else:
                low = mid + 1
        return low

    def _parity_idx_to_main_idx(self, parity_idx: int, is_even: bool) -> int:
        """Converts a relative index from even-only or odd-only array to index of arr."""
        return 2 * parity_idx + (0 if is_even else 1)

    def _main_idx_to_parity_idx(self, main_idx: int) -> int:
        """Converts an index in arr to index in even-only or odd-only array."""
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
