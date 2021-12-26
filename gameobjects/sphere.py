#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sphere Module

"Sphere module"

__title__ = 'sphere'

import vector3
from util import format_number

class Sphere:
    __slots__ = '_position', '_radius'
    
    def __init__(self, position=(0,0,0), radius= 1.):
        self._position = vector3.Vector3(position)
        self._radius = float(radius)
    
    def get_position(self):
        return self._position
    def set_position(self, position):
        self._position.set(*position)
    position = property(get_position, set_position, None, "Position of sphere centre.")
    
    def get_radius(self):
        return self._radius
    def set_radius(self, radius):
        self._radius = float(radius)
    radius = property(get_radius, set_radius, None, "Radius of the sphere.")
    
    def __str__(self):
        return 'f( position {self.position}, radius {format_number(self.radius)} )'
    
    def __repr__(self):
        return f'Sphere({tuple(self.position)}, {self.radius})'
    
    def __contains__(self, shape):
        if not hasattr(shape, 'in_sphere'):
            raise TypeError( "No 'in_sphere' method supplied by %s" % type(shape) )
        return shape.in_sphere(self)
    
    def contains(self, shape):
        return shape in self
    
    def in_sphere(self, sphere):
        return self.position.get_distance(sphere.position) + self.radius <= sphere.radius
    
    def intersects(self, shape):
        if not hasattr(shape, 'intersects_sphere'):
            raise TypeError( "No 'intersects_sphere' method supplied by %s" % type(shape) )
        return shape.intersects_sphere(self)
    
    def intersects_sphere(self, sphere):
        return self.position.get_distance(sphere.position) < self.radius + sphere.radius
        
if __name__ == "__main__":
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
