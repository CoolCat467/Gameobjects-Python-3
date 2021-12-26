from math import sqrt
from util import format_number

from vector import Vector2

if __name__ == "__main__":

    v1 = Vector2(1, 2)
    print(v1('yx'))
    print(Vector2.from_points((5,5), (10,10)))
