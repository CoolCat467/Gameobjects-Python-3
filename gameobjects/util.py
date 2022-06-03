#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Utils module

"Util module"

__title__ = 'util'

from typing import Iterable, Generator

import math

def format_number(n: int | float, accuracy: int=6) -> str:
    """Formats a number in a friendly manner
    (removes trailing zeros and unneccesary point."""
    
    fs = f'%.{accuracy}f'
    str_n = fs % (float(n) + 0)# convert -0 to 0
    if '.' in str_n:
        str_n = str_n.rstrip('0').rstrip('.')
    return str_n

def lerp(a: int | float, b: int | float, i: int | float) -> int | float:
    "Linear enterpolate from a to b."
    return a+(b-a)*i

def range2d(range_x: Iterable, range_y: Iterable) -> Iterable:
    "Creates a 2D range."
    range_x = list(range_x)
    return [ (x, y) for y in range_y for x in range_x ]

def xrange2d(range_x: Iterable, range_y: Iterable) -> Generator:
    "Iterates over a 2D range."
    range_x = list(range_x)
    for y in range_y:
        for x in range_x:
            yield (x, y)

def saturate(value: int | float, low: int | float, high: int | float) -> int | float:
    "Ensure value is within bounds, min of low, max of high."
    return min(max(value, low), high)

def is_power_of_2(n: int | float) -> bool:
    "Returns True if a value is a power of 2."
    return not math.log(n, 2) % 1

def next_power_of_2(n: int | float) -> int:
    "Returns the next power of 2 that is >= n"
    return 2 ** math.ceil(math.log(n, 2))

if __name__ == '__main__':
    print(list( xrange2d(range(3), range(3)) ))
    print(range2d(range(3), range(3)))
    print(is_power_of_2(7))
    print(is_power_of_2(8))
    print(is_power_of_2(9))
    
    print(next_power_of_2(7))
