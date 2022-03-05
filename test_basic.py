from time import perf_counter
import random
from typing import List
from RangeQueries import AlternateMinMax, BruteForceMinMax
from testing_utils import get_random_arr, get_random_query, get_separated_random_arr


def check_correctness(
    test_solver,
    baseline_solver,
    test_array_generator=get_random_arr,
    special_array_generator=get_separated_random_arr,
    min_len=2,
    max_len=10**2,
    min_val=1,
    max_val=10**2,
    num_trials=10**6,
    num_queries=10**2,
    special_chance=0.15,
):

    start_time = perf_counter()
    interval_report = max(num_trials // 100, 10)

    assert test_solver.__name__ != baseline_solver.__name__

    assert interval_report > 5

    print(
        f"Starting {num_trials=} with {baseline_solver.__name__=}, {test_solver.__name__=}, {min_len=}, {max_len=}"
    )

    for tri in range(num_trials):
        if tri % interval_report == 5:
            print(f"At trial {tri} after time {perf_counter() - start_time :.4f}")

        special_round: bool = random.random() < special_chance

        if special_round:
            my_arr: List[int] = special_array_generator(
                min_length=min_len,
                max_length=max_len,
                min_value=min_val,
                max_value=max_val,
            )
        else:
            my_arr: List[int] = test_array_generator(
                min_length=min_len,
                max_length=max_len,
                min_value=min_val,
                max_value=max_val,
            )

        my_len = len(my_arr)

        base_sol = baseline_solver(my_arr[:])
        test_sol = test_solver(my_arr[:])

        for _ in range(num_queries):
            my_left, my_right = get_random_query(my_len)

            real_ans = base_sol.query(my_left, my_right)

            test_ans = test_sol.query(my_left, my_right)

            if real_ans != test_ans:
                print(
                    f"DIFF: {real_ans=}, {test_ans=}, {special_round=}, {baseline_solver.__name__=}, {test_solver.__name__=} on:"
                )
                print()
                print(my_arr)
                print(f"query: left={my_left}, right={my_right}")
                assert False


def main():
    base_solver = BruteForceMinMax
    test_solver = AlternateMinMax

    check_correctness(test_solver=test_solver, baseline_solver=base_solver)


if __name__ == "__main__":
    main()
