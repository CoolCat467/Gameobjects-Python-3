#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Matrix Module

"Matrix module"

# Programmed by CoolCat467

__title__ = 'Matrix'
__author__ = 'CoolCat467'
__version__ = '0.0.2'
__ver_major__ = 0
__ver_minor__ = 0
__ver_patch__ = 2

import math
from typing import Any, Iterable, Union, Callable
from functools import wraps

def mmathop(function):
    "Matrix math operator decorator"
    @wraps(function)
    def wrapped_op(self, rhs, *args, **kwargs):
        if hasattr(rhs, '__len__'):
            if len(rhs) == len(self):
                return function(self, rhs, *args, **kwargs)
            raise TypeError('Operand length is not same as own')
        return function(self, [rhs]*len(self), *args, **kwargs)
##        raise AttributeError('Operand has no length attribute')
    return wrapped_op

def mapop(function: Callable) -> Callable:
    "Return new `self` class instance built from application of function on items of self."
    @wraps(function)
    def operator(self):
        return self.__class__(map(function, iter(self)), shape=self.shape)
    return operator

def simpleop(function):
    "Return new `self` class instance built from application of function on items of self and rhs."
    def apply(values):
        return function(*values)
    @wraps(function)
    def operator(self, rhs):
        return self.__class__(map(apply, zip(self, rhs)), shape=self.shape)
    return operator

def onlysquare(function):
    "Return wrapper that only runs function if matrix is square."
    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if len(set(self.shape)) == 1:
            return function(self, *args, **kwargs)
        raise TypeError('Matrix is not a square matrix!')
    return wrapper

def onlydims(n_dims) -> Callable[[Callable], Callable]:
    "Return wrapper that only runs function if matrix is square."
    def get_wrapper(function) -> Callable[['Matrix', Any], Any]:
        "Return a wrapper for function"
        @wraps(function)
        def wrapper(self, *args, **kwargs) -> Any:
            if len(self.shape) == n_dims:
                return function(self, *args, **kwargs)
            raise TypeError(f'Matrix is not a {n_dims} dimentional matrix!')
        return wrapper
    return get_wrapper

def combine_end(data: Iterable, final: str='and') -> str:
    "Join values of text, and have final with the last one properly."
    data = list(data)
    if len(data) >= 2:
        data[-1] = final+' ' + data[-1]
    if len(data) > 2:
        return ', '.join(data)
    return ' '.join(data)

def onlytype(*types, names: str=None) -> Callable:
    "Return wrapper that only runs function if all items are instances of types."
    if names is None:
        names = combine_end((t.__name__+'s' for t in types), 'or')
    def wrapper(function) -> Callable:
        @wraps(function)
        def wrapped_func(self: Iterable, *args, **kwargs) -> Any:
            for value in iter(self):
                if not isinstance(value, types):
                    raise TypeError(f'Matrix is not composed entirely of {names}')
            return function(self, *args, **kwargs)
        return wrapped_func
    return wrapper

def onlytypemath(*types, name: str=None) -> Callable:
    "Return wrapper that only runs function if all items are instances of types."
    if name is None:
        name = combine_end((t.__name__+'s' for t in types), 'or')
    def wrapper(function) -> Callable:
        @wraps(function)
        def wrapped_func(self: Iterable, rhs: Iterable, *args, **kwargs) -> Any:
            for value in iter(self):
                if not isinstance(value, types):
                    raise TypeError(f'Matrix is not composed entirely of {name}')
            for value in iter(rhs):
                if not isinstance(value, types):
                    raise TypeError(f'Operand is not composed entirely of {name}')
            return function(self, rhs, *args, **kwargs)
        return wrapped_func
    return wrapper

def boolop(combine: str='all') -> Callable:
    "Return matrix boolian simple operator. Combine can by ('any', 'all')"
    if not combine in {'any', 'all'}:
        raise ValueError("Combine must be either 'any' or 'all'!")
    def wrapper(function) -> Callable[[Iterable, Iterable], bool]:
        "Matrix boolian operator decorator"
        def apply(values: Iterable) -> bool:
            return function(*values)
        if combine == 'any':
            def operator(self, rhs) -> bool:
                return any(map(apply, zip(self, rhs)))
        else:
            def operator(self, rhs) -> bool:
                return all(map(apply, zip(self, rhs)))
        return wraps(function)(operator)
    return wrapper

class Matrix:
    "Matrix Class"
    __slots__ = ('__m','__shape')
    def __init__(self, data, shape: tuple, dtype: type=tuple):
        self.__shape = shape
        items = math.prod(self.__shape)
        data = list(data)
        if len(data) != items:
            data = sum(data, [])
        if len(data) != items:
            size = 'x'.join(map(str, self.shape))
            raise ValueError(f'Unequal number of elements for {size} matrix!')
        self.__m = dtype(data)
    
    def __repr__(self) -> str:
        elem = [len(s) for s in map(str, self.elements)]
        leng = min(8, math.ceil(sum(elem)/len(elem)))
        elem = [len(str(round(e, leng))) for e in self.elements]
        leng = min(8, math.ceil(sum(elem)/len(elem)))
        rows = ['  ['+', '.join(str(round(r, leng))[:leng].rjust(leng) for r in row)+']'
                for row in self.rows()]
        args = '[\n'+',\n'.join(rows)+'\n]'+', '+str(self.shape)
        return f'{self.__class__.__name__}({args})'
    
    @property
    def shape(self) -> tuple:
        "Shape of this Matrix"
        return self.__shape
    
    @property
    def elements(self):
        "Unwrapped elements of this matrix"
        return self.__m
    
    @property
    def T(self) -> 'Matrix':# pylint: disable=invalid-name
        "Transpose of this matrix"
        return self.transpose()
    
    def rows(self):
        "Return rows of this matrix"
        rows = []
        for ridx in range(self.shape[0]):
            rows.append(self.elements[ridx*self.shape[1]:(ridx+1)*self.shape[1]])
        return rows
    
    def __len__(self):
        return len(self.__m)
    
    def __iter__(self):
        return iter(self.__m)
    
    def __get_active_indices(self, index) -> Union[int, list[int]]:
        "Return either list of indices or single indice to act apon for given index."
        data = list(range(len(self)))
        shape = list(self.shape)
        all_reg = True
        last_dim = 1
        for iidx, part in enumerate(index):
            # print(part)
            next_dim = shape.pop()
            slicelen = math.prod(shape)
            if not isinstance(part, slice):
##                print(f'{part=}')
                if all_reg:
                    if iidx+1 < len(index) and index[iidx+1] == slice(None) and slicelen < next_dim:
                        part = slice(part*next_dim, (part+1)*next_dim)
                    else:
                        part = slice(part*slicelen, (part+1)*slicelen)
                elif self.shape[1] == 1:
                    part = slice(None, None, last_dim)
                else:
                    part = slice(part, None, last_dim)
##                print(f'new {part=}\n')
            else:
                all_reg = False
                start, stop, step = part.indices(len(self))
                part = slice(start*next_dim, stop*next_dim, step)
            data = data[part]
            last_dim *= next_dim
        if not data:
##            print(self)
##            print(f'2 {index=}')
            raise IndexError('Invalid index for matrix!')
        if all_reg:
            return data[0]
        return data
    
    def __getitem__(self, index) -> Union[list, Union[int, float]]:
        # (row, column)
        if not isinstance(index, tuple) or len(index) < len(self.__shape):
            size = 'x'.join(map(str, self.shape))
            raise IndexError(f'Not enough arguments for {size} matrix.')
        if len(index) > len(self.__shape):
            size = 'x'.join(map(str, self.shape))
            raise IndexError(f'Too many arguments for {size} matrix.')
        index = self.__get_active_indices(index)
        if isinstance(index, int):
            return self.__m[index]
        return [self.__m[idx] for idx in index]
    
    def __setitem__(self, index, value) -> None:
        if not isinstance(index, tuple) or len(index) < len(self.__shape):
            size = 'x'.join(map(str, self.shape))
            raise IndexError(f'Not enough arguments for {size} matrix.')
        if len(index) > len(self.__shape):
            size = 'x'.join(map(str, self.shape))
            raise IndexError(f'Too many arguments for {size} matrix.')
        positions = self.__get_active_indices(index)
        if isinstance(positions, int):
            self.__m[positions] = value
            return
        if not hasattr(value, '__len__'):
            for idx in positions:
                self.__m[idx] = value
        elif hasattr(value, '__iter__'):
            values: Iterable = iter(value)
            for idx, val in zip(positions, values):
                self.__m[idx] = val
    
    @classmethod
    def from_iter(cls, iterable: Iterable, shape: tuple) -> 'Matrix':
        "Return Matrix from iterable."
        return cls(iterable, shape=shape)
    
    @classmethod
    def zeros(cls, shape: tuple) -> 'Matrix':
        "Return Matrix of zeros in given shapes."
        return cls([0]*math.prod(shape), shape=shape)
    
    @classmethod
    def identity(cls, size: int) -> 'Matrix':
        "Return square identity Matrix of given size."
        values = []
        next_ = 0
        for i in range(size**2):
            if i == next_:
                values.append(1)
                next_ += size+1
            else:
                values.append(0)
        return cls(values, shape=(size, size))
    
    def copy(self) -> 'Matrix':
        "Return a copy of this matrix"
        return self.from_iter(self.__m, shape=self.shape)
    
    def __reversed__(self) -> reversed:
        "Return a copy of self, but order of elements is reversed."
        return reversed(self.__m)
    
    def __contains__(self, value: Union[int, float]) -> bool:
        "Return if self contains value"
        return value in self.__m
    
    @mapop
    def __pos__(self) -> 'Matrix':
        "Return unary positive of self"
        return +self
    
    @mapop
    def __neg__(self) -> 'Matrix':
        "Return negated matrix"
        return -self
    
    @onlytype(int)
    @mapop
    def __invert__(self) -> 'Matrix':
        "Return bitwise NOT of self if all items are intigers"
        return ~self
    
    @mapop
    def __abs__(self) -> 'Matrix':
        "Return abs'd matrix"
        return abs(self)
    
    def __round__(self, ndigits: int=None) -> 'Matrix':
        "Return matrix but each element is rounded"
        return self.from_iter((round(x, ndigits) for x in self.__m), shape=self.shape)
    
    @mapop
    def __ceil__(self) -> 'Matrix':
        "Return matrix but each element is ceil ed"
        return math.ceil(self)
    
    @mapop
    def __floor__(self) -> 'Matrix':
        "Return matrix but each element is floored"
        return math.floor(self)
    
    @mapop
    def __trunc__(self) -> 'Matrix':
        "Return matrix but each element is trunc ed"
        return math.trunc(self)
    
    def __bool__(self):
        "Return True if any element is true, False otherwise"
        return any(self.__m)
    
    @mmathop
    @simpleop
    def __add__(self, rhs) -> 'Matrix':
        "Add number to each element"
        return self + rhs
    __radd__ = __add__
    
    @mmathop
    @simpleop
    def __sub__(self, rhs):
        "Subtract number from each element"
        return self - rhs
    
    @mmathop
    @simpleop
    def __rsub__(self, lhs):
        "Subtract but from left hand side"
        return lhs - self
    
    @mmathop
    @simpleop
    def __mul__(self, rhs):
        "Multiply each element by number"
        return self * rhs
    __rmul__ = __mul__
    
    @mmathop
    @simpleop
    def __truediv__(self, rhs):
        "Divide each element by number"
        return self / rhs
    
    @mmathop
    @simpleop
    def __rtruediv__(self, lhs):
        "Division but from left hand side"
        return lhs / self
    
    @mmathop
    @simpleop
    def __floordiv__(self, rhs) -> 'Matrix':
        "Floor divide each element by number"
        return self // rhs
    
    @mmathop
    @simpleop
    def __rfloordiv__(self, lhs) -> 'Matrix':
        "Floor division but from left hand side"
        return lhs // self
    
    @mmathop
    @simpleop
    def __pow__(self, rhs):
        "Get element to the power of number for each element"
        return self ** rhs

    @mmathop
    @simpleop
    def __rpow__(self, lhs):
        "Power, but from left hand side"
        return lhs ** self
    
    @mmathop
    @simpleop
    def __mod__(self, rhs) -> 'Matrix':
        "Return remainder of division (modulo) of self by rhs"
        return self % rhs
    
    @mmathop
    @simpleop
    def __rmod__(self, lhs) -> 'Matrix':
        "Modulo but from left hand side."
        return lhs % self
    
    def __divmod__(self, rhs) -> tuple[Union[int, float, complex], Union[int, float]]:
        "Return tuple of (self // rhs, self % rhs)"
        return self // rhs, self % rhs
    
    def __rdivmod__(self, lhs) -> tuple[Union[int, float, complex], Union[int, float]]:
        "Divmod but from left hand side"
        return lhs // self, lhs % self
    
    @mmathop
    @boolop('all')
    def __eq__(self, rhs) -> bool:
        "Return True if all elements of both matrixs are equal."
        return self == rhs
    
    @mmathop
    @boolop('any')
    def __ne__(self, rhs) -> bool:
        "Return True if any element is not equal to it's counterpart in the other matrix"
        return self != rhs
    
    @mmathop
    @boolop('all')
    def __lt__(self, rhs) -> bool:
        "Return True if all elements of self are less than corresponding element in rhs."
        return self < rhs
    
    @mmathop
    @boolop('all')
    def __gt__(self, rhs) -> bool:
        "Return True if all elements of self are greater than corresponding element in rhs."
        return self > rhs
    
    @mmathop
    @boolop('all')
    def __le__(self, rhs) -> bool:
        "Return True if all elements of self are less than or equal to corresponding element."
        return self <= rhs
    
    @mmathop
    @boolop('all')
    def __ge__(self, rhs) -> bool:
        "Return True if all elements of self are greater than or equal to corresponding element."
        return self >= rhs
    
    def __hash__(self):
        return hash(self.__m)
    
    @mapop
    def conv_ints(self):
        "Return copy of self, but all items are intigers"
        return int(self)
    
    @mapop
    def conv_floats(self):
        "Return copy of self, but all items are floats"
        return float(self)
    
    # Intiger operators
    @mmathop
    @onlytypemath(int)
    @simpleop
    def __and__(self, rhs) -> 'Matrix':
        "Return bitwise AND of self and rhs if both are composed of intigers"
        return self & rhs
    __rand__ = __and__
    
    @mmathop
    @onlytypemath(int)
    @simpleop
    def __or__(self, rhs) -> 'Matrix':
        "Return bitwise OR of self and rhs if both are composed of intigers"
        return self | rhs
    __ror__ = __or__
    
    @mmathop
    @onlytypemath(int)
    @simpleop
    def __lshift__(self, rhs) -> 'Matrix':
        "Return bitwise left shift of self by rhs if both are composed of intigers"
        return self << rhs
    
    @mmathop
    @onlytypemath(int)
    @simpleop
    def __rlshift__(self, lhs) -> 'Matrix':
        "Bitwise left shift but from left hand side"
        return lhs << self
    
    @mmathop
    @onlytypemath(int)
    @simpleop
    def __rshift__(self, rhs) -> 'Matrix':
        "Return bitwise right shift of self by rhs if both are composed of intigers"
        return self >> rhs
    
    @mmathop
    @onlytypemath(int)
    @simpleop
    def __rrshift__(self, lhs) -> 'Matrix':
        "Bitwise right shift but from left hand side"
        return lhs >> self
    
    @mmathop
    @onlytypemath(int)
    @simpleop
    def __xor__(self, rhs) -> 'Matrix':
        return self ^ rhs
    __rxor__ = __xor__
    
    def __matmul__(self, rhs):
        "Return matrix multiply with rhs."
        if not hasattr(rhs, 'shape'):
            raise AttributeError('Right hand side has no `shape` attribute')
        if len(rhs.shape) != 2:
            raise ValueError('Right hand side is more than a two dimensional matrix')
        if self.shape[1] != rhs.shape[0]:
            raise ArithmeticError('Right hand side is of an incompatable shape for matrix'
                                  'multiplication')
        results = []
        m_a = self.elements
        m_b = tuple(rhs)
        for row in range(self.shape[0]):
            for col in range(rhs.shape[1]):
                value = math.fsum(m_a[self.shape[1] * row + x] * m_b[rhs.shape[1] * x + col]
                                  for x in range(self.shape[1]))
                results.append(value)
        return self.__class__(results, (self.shape[0], rhs.shape[1]))
    
    def minor(self, index: tuple) -> 'Matrix':
        "Return new matrix without index location in any shape"
        dims = len(self.shape)
        if len(index) != dims:
            size = 'x'.join(map(str, self.shape))
            raise IndexError(f'Invalid number of arguments for {size} matrix.')
        results = []
        head = [0]*dims
        for elem in self.elements:
            keep = True
            for dim, avoid in enumerate(index):
                if head[dim] == avoid:
                    keep = False
                    break
            if keep:
                results.append(elem)
            head[-1] += 1
            for idx, dim in reversed(tuple(enumerate(self.shape))):
                if head[idx] >= dim:
                    head[idx] = 0
                    if idx != 0:
                        head[idx-1] += 1
        return self.__class__(results, tuple(x-1 for x in self.shape))
    
    @onlydims(2)
    @onlysquare
    def determinent(self) -> Union[int, float]:
        "Return the determinent of this matrix."
        if len(self.elements) == 1:
            return self.elements[0]
        
        value = 0
        for i in range(self.shape[0]):
            sign = 2 * (i % 2) - 1
            value -= sign * self[0, i] * self.minor((0, i)).determinent()
        return value
    
    @onlydims(2)
    @onlysquare
    def cofactor_element(self, index) -> Union[int, float]:
        "Return cofactor of item at index in this matrix"
        return (-1) ** sum(index) * self.minor(index).determinent()
        # return ((sum(index)%2)*2-1) * self.minor(index).determinent()
    
    @onlydims(2)
    @onlysquare
    def cofactor(self) -> 'Matrix':
        "Return cofactor of self"
        values = [
            self.cofactor_element((r, c))
            for r in range(self.shape[0])
            for c in range(self.shape[1])
        ]
        return self.__class__(values, self.shape)
    
    @onlydims(2)
    def transpose(self) -> 'Matrix':
        "Return transpose of self"
        values = [self[:,x] for x in range(self.shape[1])]
        return self.__class__(values, tuple(reversed(self.shape)))
    
    @onlydims(2)
    @onlysquare
    def adjugate(self) -> 'Matrix':
        "Return adjugate of self"
        return self.cofactor().transpose()
    adjoint = adjugate
    
    @onlysquare
    def inverse(self):
        "Return the inverse of this matrix"
        det = self.determinent()
        if det == 0:
            raise ZeroDivisionError('Determinent of this matrix is zero!')
        return self.adjugate() / det



def test():
    "Test"
    # pylint: disable=invalid-name
    A = Matrix([0, 0, 2, 1, 3, -2, 1, -2, 1], shape=(3, 3))
    X = Matrix([1, -2, 3], shape=(3, 1))
    print(f'{A @ X == [6, -11, 8] = }')
    
    A = Matrix([0, 0, 2, 1, 3, -2, 1, -2, 1], shape=(3, 3))
    X = Matrix([-1.1, 1.7, 4.5], shape=(3, 1))
    print(f'{A @ X == [9, -5, 0] = }')
    
    A = Matrix([-3, 4, 0, 2, -5, 1, 0, 2, 3], (3, 3))
    print(f'{A.determinent() == 27 = }')
    print(f'{round(A.inverse() @ A) == Matrix.identity(3) = }')
    
    A = Matrix([
      [-2,  1,  1,  0],
      [-1,  2, -1,  1],
      [-2,  3,  3,  2],
      [ 1,  1,  2,  1]
    ], (4, 4))
    # print(A)
    # print(A.inverse())
    print(f'{A.inverse() @ Matrix([0, 1, 6, 5], (4, 1)) == [1, 1, 1, 1] = }')
    
    A = Matrix([3, -3, -1,
                2, -2, 4],
               (2, 3))
    B = Matrix([3, -3, 2], (3, 1))
    print(f'{A @ B == [16, 20] = }')
    
    A = Matrix([
      [-2,  3],
      [-4,  5]
    ], (2, 2))
    print(f'{A @ A.inverse() == Matrix.identity(2) = }')

if __name__ == '__main__':
    print(f'{__title__} v{__version__}\nProgrammed by {__author__}.')
    test()
