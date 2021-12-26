#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Vector3 module"

__title__ = 'vector3'

from math import sqrt
from util import format_number

from vector import Vector3 as _Vector3

class Vector3(_Vector3):
    def in_sphere(self, sphere):
        """Returns true if this vector (treated as a position) is contained in
        the given sphere.

        """
        
        return distance3d(sphere.position, self) <= sphere.radius

def distance3d_squared(p1, p2):
    x, y, z = p1
    xx, yy, zz = p2
    dx = x - xx
    dy = y - yy
    dz = z - zz

    return dx*dx + dy*dy +dz*dz

def distance3d(p1, p2):
    x, y, z = p1
    xx, yy, zz = p2
    dx = x - xx
    dy = y - yy
    dz = z - zz

    return sqrt(dx*dx + dy*dy +dz*dz)

def centre_point3d(points):
    return sum( Vector3.from_iter(p) for p in points ) / len(points)

def test():
    v1 = Vector3(2.2323, 3.43242, 1.)
    
    print(3*v1)
    print((2, 4, 6)*v1)
    
    print((1, 2, 3)+v1)
    print(v1('xxxyyyzzz'))
    print(v1[2])
    print(v1.z)
    v1[2]=5.
    print(v1)
    v2= Vector3(1.2, 5, 10)
    print(v2)
    v1 += v2
    print(v1.get_length())
    print(repr(v1))
    print(v1[1])
    
    p1 = Vector3(1,2,3)
    print(p1)
    print(repr(p1))
    
    for v in p1:
        print(v)
    
    #print(p1[6])
    
    ptest = Vector3( [1,2,3] )
    print(ptest)
    
    z = Vector3()
    print(z)
    
    open("test.txt", "w").write( "\n".join(str(float(n)) for n in range(20)) )
    f = map(float, open("test.txt").read().splitlines())
    v1 = Vector3.from_iter( f )
    v2 = Vector3.from_iter( f )
    v3 = Vector3.from_iter( f )
    print(v1, v2, v3)
    
    print("--")
    print(v1)
    print(v1 + (10,20,30))
    
    print(v1('xz'))

    print(-v1)

    #print(tuple(ptest))
    #p1.set( (4, 5, 6) )
    #print(p1)
    
    print(Vector3(10,10,30)+v1)
    
    print(Vector3((0,0,0,1)))
    
    print(Vector3(1, 2, 3).scale(3))
    
    print(Vector3(1, 2, 3).scale((2, 4, 6)))
    
    print(bool(v1))

if __name__ == "__main__":
    test()
