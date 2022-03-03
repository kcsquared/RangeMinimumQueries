import random
from typing import List, Tuple



def get_random_arr(min_length: int, max_length: int, min_value: int, max_value: int) -> List[
    int]:
    my_len = random.randrange(min_length, max_length)
    return random.choices(range(min_value, max_value), k=my_len)


def get_separated_random_arr(min_length: int, max_length: int, min_value: int, max_value: int) -> \
List[
    int]:
    my_len = random.randrange(min_length, max_length)
    arr = []
    arr.append(random.randrange(min_value + 1, max_value))

    for i in range(1, my_len):
        if i % 2 == 0:
            arr.append(random.randrange(arr[-1] + 1, max_value))
        else:
            arr.append(random.randrange(min_value, arr[-1]))

    return arr


def get_random_query(my_len: int) -> Tuple[int, int]:
    first, second = random.choices(range(my_len), k=2)
    if first > second:
        first, second = second, first
    return first, second