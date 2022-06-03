#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sphere Module

"Sphere module"

__title__ = 'sphere'

from typing import Iterable

import vector3

class Sphere:
    "Sphere class"
    __slots__ = ('_position', '_radius')
    def __init__(self, position: Iterable=(0,0,0), radius:int|float=1) -> None:
        self._position = vector3.Vector3(position)
        self._radius = radius
    
    def get_position(self) -> vector3.Vector3:
        "Return position of the center of this sphere"
        return self._position
    
    def set_position(self, position: Iterable) -> None:
        x, y, z = position
        self._position.x = x
        self._position.y = y
        self._position.z = z
    
    position = property(get_position, set_position, None, "Position of sphere centre.")
    
    def get_radius(self) -> int|float:
        "Return radius of this sphere"
        return self._radius
    
    def set_radius(self, radius: int|float) -> None:
        "Set the radius of this sphere"
        self._radius = radius
    
    radius = property(get_radius, set_radius, None, "Radius of the sphere.")
    
    def __str__(self) -> str:
        "Return text about self"
        return f'( position {self.position}, radius {self.radius} )'
    
    def __repr__(self):
        "Return representaton of self"
        return f'Sphere({tuple(self.position)}, {self.radius})'
    
    def __contains__(self, shape) -> bool:
        "Return if shape is in this sphere"
        if not hasattr(shape, 'in_sphere'):
            raise TypeError( "No 'in_sphere' method supplied by %s" % type(shape) )
        return shape.in_sphere(self)
    
    contains = __contains__
    
    def in_sphere(self, sphere: 'Sphere') -> bool:
        "Return if sphere is in this sphere"
        return self.position.get_distance(sphere.position) + self.radius <= sphere.radius
    
    def intersects(self, shape) -> bool:
        "Return if shape intersects this sphere"
        if not hasattr(shape, 'intersects_sphere'):
            raise TypeError(f"No 'intersects_sphere' method supplied by {type(shape)}")
        return shape.intersects_sphere(self)
    
    def intersects_sphere(self, sphere: 'Sphere') -> bool:
        "Return if sphere is in this sphere"
        return self.position.get_distance(sphere.position) < self.radius + sphere.radius

def test():
    "Do test"
    s1 = Sphere()
    s2 = Sphere( (1,1,1) )
    s3 = Sphere( radius=10 )
    s4 = eval(repr(s2))
    
    print(s1)
    print(repr(s2))
    print(s2, s4)
    
    v = vector3.Vector3(0, 1, 0)
    print(v in s1)
    
    big = Sphere(radius=1)
    small = Sphere(position=(.8, 0, 0), radius=.2)
    
    
    print(small, big)
    print(small in big)

if __name__ == '__main__':    
    test()
