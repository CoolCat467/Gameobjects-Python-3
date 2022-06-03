#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Matrix44 Module

"Matrix44"

# Programmed by CoolCat467

__title__ = 'Matrix44'
__author__ = 'CoolCat467'

from typing import Iterable
import math

from matrix import Matrix
from vector import Vector4

class Matrix44(Matrix):
    "4x4 Matrix with extra 3d math functions"
    __slots__: tuple = tuple()
    def __init__(self, data, shape=(4, 4), dtype=list):
        super().__init__(data, (4, 4), dtype)
    
    @property
    def shape(self) -> tuple[int, int]:
        "Return (4, 4)"
        return 4, 4
    
    def __matmul__(self, rhs):
        "Return matrix multiply with rhs."
        result = super().__matmul__(rhs)
        if result.shape == (4, 4):
            return self.__class__(result[:])
        return result
    
    @classmethod
    def identity(cls, size=None) -> 'Matrix44':
        return super().identity(4)
    
    @classmethod
    def zeros(cls, shape=None) -> 'Matrix44':
        return super().zeros((4, 4))
    
    def minor(self, index: tuple) -> Matrix:
        "Return new matrix without index location in any shape"
        pos_row, pos_col = index
        results = []
        for ridx, row in enumerate(self.rows()):
            if ridx == pos_row:
                continue
            for cidx, val in enumerate(row):
                if cidx == pos_col:
                    continue
                results.append(val)
        return Matrix(results, tuple(x-1 for x in self.shape))
    
    # pylint: disable=unused-private-member
    def __get_row_0(self) -> Vector4:
        return Vector4.from_iter(self[0, :], dtype=tuple)
    
    def __get_row_1(self) -> Vector4:
        return Vector4.from_iter(self[1, :], dtype=tuple)
    
    def __get_row_2(self) -> Vector4:
        return Vector4.from_iter(self[2, :], dtype=tuple)
    
    def __get_row_3(self) -> Vector4:
        return Vector4.from_iter(self[3, :], dtype=tuple)
    
    def __set_row_0(self, values: Iterable) -> None:
        values = tuple(values)[:4]
        self[0, :] = list(map(float, values))
    
    def __set_row_1(self, values: Iterable) -> None:
        values = tuple(values)[:4]
        self[1, :] = list(map(float, values))
    
    def __set_row_2(self, values: Iterable) -> None:
        values = tuple(values)[:4]
        self[2, :] = list(map(float, values))
    
    def __set_row_3(self, values: Iterable) -> None:
        values = tuple(values)[:4]
        self[3, :] = list(map(float, values))
    
    __row0 = property(__get_row_0, __set_row_0, None, 'Row 0')
    __row1 = property(__get_row_1, __set_row_1, None, 'Row 1')
    __row2 = property(__get_row_2, __set_row_2, None, 'Row 2')
    __row3 = property(__get_row_3, __set_row_3, None, 'Row 3')
    
    x_axis = __row0
    right = __row0
    y_axis = __row1
    up = __row1
    z_axis = __row2
    forward = __row2
    translate = __row3
    
    def move(self,
             forward: int|float=None,
             right: int|float=None,
             # pylint: disable=invalid-name
             up: int|float=None
    ) -> None:
        """Changes the translation according to a direction vector.
        To move in opposite directions (i.e back, left and down), first
        negate the vector.
        
        forward -- Units to move in the 'forward' direction
        right -- Units to move in the 'right' direction
        up -- Units to move in the 'up' direction
        
        """
        
        if forward is not None:
            self.translate += self.forward * forward
        
        if right is not None:
            self.translate += self.right * right
        
        if up is not None:
            self.translate += self.up * up
    
    @classmethod
    def make_rotation_about_axis(cls, axis: Iterable, angle: float) -> 'Matrix44':
        """Makes a rotation Matrix44 around an axis.

        axis -- An iterable containing the axis (three values)
        angle -- The angle to rotate (in radians)

        """
        
        # pylint: disable=invalid-name
        c = math.cos(angle)
        s = math.sin(angle)
        omc = 1 - c
        x, y, z = axis
        
        results = [x*x*omc+c,   y*x*omc+z*s, x*z*omc-y*s, 0,
                   x*y*omc-z*s, y*y*omc+c,   y*z*omc+x*s, 0,
                   x*z*omc+y*s, y*z*omc-x*s, z*z*omc+c,   0,
                   0,           0,           0,           1]
        return cls(results)
    
    @classmethod
    def make_xyz_rotation(cls,
                          angle_x: float,
                          angle_y: float,
                          angle_z: float) -> 'Matrix44':
        "Makes a rotation Matrix44 about 3 axis."
        
        # pylint: disable=invalid-name
        cx = math.cos(angle_x)
        sx = math.sin(angle_x)
        cy = math.cos(angle_y)
        sy = math.sin(angle_y)
        cz = math.cos(angle_z)
        sz = math.sin(angle_z)
        
        sxsy = sx*sy
        cxsy = cx*sy
        
        # http://web.archive.org/web/20041029003853/http:/www.j3d.org/matrix_faq/matrfaq_latest.html#Q35
        #A = math.cos(angle_x)
        #B = math.sin(angle_x)
        #C = math.cos(angle_y)
        #D = math.sin(angle_y)
        #E = math.cos(angle_z)
        #F = math.sin(angle_z)
        
        #     |  CE      -CF       D   0 |
        #M  = |  BDE+AF  -BDF+AE  -BC  0 |
        #     | -ADE+BF   ADF+BE   AC  0 |
        #     |  0        0        0   1 |
        
        results = [ cy*cz,  sxsy*cz+cx*sz,  -cxsy*cz+sx*sz, 0,
                    -cy*sz, -sxsy*sz+cx*cz, cxsy*sz+sx*cz,  0,
                    sy,     -sx*cy,         cx*cy,          0,
                    0,      0,              0,              1]
        
        return cls(results)
    
    def trace(self) -> float:
        "Return sum of scale"
        return math.fsum((self[0, 0], self[1, 1], self[2, 2], self[3, 3]))
    
    def to_opengl(self):
        """Converts the matrix in to a list of values, suitable for using
        with glLoadMatrix*"""
        
        return self._m[:]

def run():
    "Run"

if __name__ == '__main__':
    print(f'{__title__}\nProgrammed by {__author__}.')
    run()
