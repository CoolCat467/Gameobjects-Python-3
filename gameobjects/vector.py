#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Vector Module

"Vector module"

# Programmed by CoolCat467

__title__ = 'Vector'
__author__ = 'CoolCat467'
__version__ = '0.0.0'
__ver_major__ = 0
__ver_minor__ = 0
__ver_patch__ = 0

import math
from types import GenericAlias
from typing import Any, Iterable, Union, TypeVar, Callable, overload
from functools import wraps

##Number = TypeVar('Number', int, float, complex)

def vmathop(function: Callable) -> Callable:
    "Vector math operator decorator"
    @wraps(function)
    def wrapped_op(self, rhs, *args, **kwargs):
        if hasattr(rhs, '__len__'):
            if len(rhs) == len(self):
                return function(self, rhs, *args, **kwargs)
            raise TypeError('Operand length is not same as own')
        return function(self, [rhs]*len(self), *args, **kwargs)
    return wrapped_op

def mapop(function: Callable) -> Callable:
    "Return new `self` class instance built from application of function on items of self."
    @wraps(function)
    def operator(self):
        return self.__class__(*map(function, iter(self)), dtype=self.dtype)
    return operator

def simpleop(function: Callable) -> Callable:
    "Return new `self` class instance built from application of function on items of self and rhs."
    def apply(values: Any) -> Any:
        return function(*values)
    @wraps(function)
    def operator(self, *args: Any):
        return self.__class__(*map(apply, zip(self, *args)), dtype=self.dtype)
    return operator

def onlylen(length: int) -> Callable:
    "Return wrapper that only runs function if length matches length."
    def wrapper(function) -> Callable:
        @wraps(function)
        def wrapped_func(self, *args, **kwargs) -> Any:
            if len(self) == length:
                return function(self, *args, **kwargs)
            raise TypeError(f'Vector is not a {length}d vector!')
        return wrapped_func
    return wrapper

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
                    raise TypeError(f'Vector is not composed entirely of {names}')
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
                    raise TypeError(f'Vector is not composed entirely of {name}')
            for value in iter(rhs):
                if not isinstance(value, types):
                    raise TypeError(f'Operand is not composed entirely of {name}')
            return function(self, rhs, *args, **kwargs)
        return wrapped_func
    return wrapper

def boolop(combine: str='all') -> Callable:
    "Return vector boolian simple operator. Combine can by ('any', 'all')"
    if not combine in {'any', 'all'}:
        raise ValueError("Combine must be either 'any' or 'all'!")
    def wrapper(function) -> Callable[[Iterable, Iterable], bool]:
        "Vector boolian operator decorator"
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

class Vector:
    """Vector Object. Takes n arguments as input and creates n length vector, or type length vector.
    dtype argument changes internal data type."""
    __slots__ = ('__v', 'dtype')
    def __init__(self, *args: Union[int, float, complex], dims: int=None, dtype: type=tuple):
        self.dtype = dtype
        if not hasattr(dtype, '__getitem__'):
            raise TypeError('Data type class is not subscriptable!')
        if dims is None:
            self.__v = self.dtype(args)
        else:
            args = args[:dims]
            self.__v = self.dtype(list(args) + [0]*(dims-len(args)))
    
    def __repr__(self) -> str:
        args = ', '.join(repr(x) for x in self.__v)
        return f'{self.__class__.__name__}({args})'
    
    @classmethod
    def __class_getitem__(cls, value: Any) -> GenericAlias:
        return GenericAlias(cls, value)
    
    def __len__(self) -> int:
        return len(self.__v)
    
##    @property
##    def shape(self) -> tuple:
##        return (len(self), 1)
    
    def __iter__(self) -> Iterable:
        return iter(self.__v)
        
    def __getitem__(self, index: int) -> Union[int, float, complex]:
        if not isinstance(index, int):
            raise TypeError('Index is not an intiger.')
        if index > len(self):
            raise IndexError('Index out of range for this vector')
##        if isinstance(index, str):
##            index = ord(index.upper())-88
        return self.__v[index]
    
    def __setitem__(self, index: int, value: Union[int, float, complex]) -> None:
        if not hasattr(self.__v, '__setitem__'):
            dtype = self.dtype.__name__
            raise TypeError(f"'{dtype}' does not support item assignment. Change vector dtype.")
        if not isinstance(index, int):
            raise TypeError('Index is not an intiger.')
        if index > len(self):
            raise IndexError('Index out of range for this vector')
        self.__v[index] = value
    
    @classmethod
    def from_iter(cls, iterable: Iterable, dims: int=None, dtype: type=tuple) -> 'Vector':
        "Return Vector from iterable."
        return cls(*iterable, dims=dims, dtype=dtype)
    
    @classmethod
    def from_radians(cls, radians: Union[int, float]) -> 'Vector':
        "Return 2d unit vector from measure in radians"
        return cls(math.cos(radians), math.sin(radians))
    
    @classmethod
    def from_points(cls, point1: Union[Iterable, Union[int, float, complex]], point2: Iterable) -> 'Vector':
        "Create a vector from point1 toward point2."
        if not hasattr(point2, '__len__'):
            raise AttributeError('Operand 2 has no length attribute')
        return cls.from_iter(point2) - point1
    
    @property
    def magnitude(self) -> float:
        "Magnitude of this vector"
##        return math.sqrt(sum(self ** 2))
        return math.hypot(*self.__v)
    
    def copy(self) -> 'Vector':
        "Return a copy of this vector"
        return self.from_iter(self.__v, dtype=self.dtype)
    
    def __reversed__(self) -> reversed:
        "Return a copy of self, but order of elements is reversed."
        return reversed(self.__v)
    
    def __contains__(self, value) -> bool:
        "Return if self contains value"
        return value in self.__v
    
    def normalize(self) -> 'Vector':
        "Normalize this vector **IN PLACE**"
        self.__v = self.dtype(self / self.magnitude)
        return self
    
    def normalized(self) -> 'Vector':
        "Return normalized copy of this vector"
        return self / self.magnitude
    
    @mapop
    def __pos__(self) -> 'Vector':
        "Return unary positive of self"
        return +self
    
    @mapop
    def __neg__(self) -> 'Vector':
        "Return negated vector"
        return -self
    
    @onlytype(int)
    @mapop
    def __invert__(self) -> 'Vector':
        "Return bitwise NOT of self if all items are intigers"
        return ~self
    
    @mapop
    def __abs__(self) -> 'Vector':
        "Return abs'd vector"
        return abs(self)
    
    def __round__(self, ndigits: int=None) -> 'Vector':
        "Return vector but each element is rounded"
        return self.__class__.from_iter((round(x, ndigits) for x in self.__v), dtype=self.dtype)
    
    @mapop
    def __ceil__(self):# -> 'Vector'
        "Return vector but each element is ceil ed"
        return math.ceil(self)
    
    @mapop
    def __floor__(self):# -> 'Vector'
        "Return vector but each element is floored"
        return math.floor(self)
    
    @mapop
    def __trunc__(self):# -> 'Vector'
        "Return vector but each element is trunc ed"
        return math.trunc(self)
    
    def __bool__(self) -> bool:
        "Return True if any element is true, False otherwise"
        return any(self.__v)
    
    @vmathop
    @simpleop
    def __add__(self, rhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Add two vectors/iterables or add Union[int, float, complex] to each element"
        return self + rhs
    __radd__ = __add__
    
    @vmathop
    @simpleop
    def __sub__(self, rhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Subtract two vectors/iterables or subtract Union[int, float, complex] from each element"
        return self - rhs
    
    @vmathop
    @simpleop
    def __rsub__(self, lhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Subtract but from left hand side"
        return lhs - self
    
    @vmathop
    @simpleop
    def __mul__(self, rhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Multiply two vectors/iterables or multiply each element by number"
        return self * rhs
    __rmul__ = __mul__
    
    @vmathop
    @simpleop
    def __truediv__(self, rhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Divide two vectors/iterables or divide each element by number"
        return self / rhs
    
    @vmathop
    @simpleop
    def __rtruediv__(self, lhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Division but from left hand side"
        return lhs / self
    
    @vmathop
    @simpleop
    def __floordiv__(self, rhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Floor divide two vectors/iterables or flor divide each element by number"
        return self // rhs
    
    @vmathop
    @simpleop
    def __rfloordiv__(self, lhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Floor division but from left hand side"
        return lhs // self
    
    @vmathop
    @simpleop
    def __pow__(self, rhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Get element to the power of Union[int, float, complex] or matching item in vector/iterable for each element"
        return self ** rhs
    
    @vmathop
    @simpleop
    def __rpow__(self, lhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Power, but from left hand side"
        return lhs ** self
    
    @vmathop
    @simpleop
    def __mod__(self, rhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Return remainder of division (modulo) of self by rhs"
        return self % rhs
    
    @vmathop
    @simpleop
    def __rmod__(self, lhs: Union[Union[int, float, complex], Iterable]) -> 'Vector':
        "Modulo but from left hand side."
        return lhs % self
    
    def __divmod__(self, rhs: Union[Union[int, float, complex], Iterable]) -> tuple[Union[int, float, complex], Union[int, float]]:
        "Return tuple of (self // rhs, self % rhs)"
        return self // rhs, self % rhs
    
    def __rdivmod__(self, lhs: Union[Union[int, float, complex], Iterable]) -> tuple[Union[int, float, complex], Union[int, float]]:
        "Divmod but from left hand side"
        return lhs // self, lhs % self
    
    @vmathop
    @boolop('all')
    def __eq__(self, rhs: Union[Union[int, float, complex], Iterable]) -> bool:
        "Return True if all elements of both vectors are equal."
        return self == rhs
    
    @vmathop
    @boolop('any')
    def __ne__(self, rhs: Union[Union[int, float, complex], Iterable]) -> bool:
        "Return True if any element is not equal to it's counterpart in the other vector"
        return self != rhs

    @vmathop
    @boolop('all')
    def __lt__(self, rhs: Union[Union[int, float, complex], Iterable]) -> bool:
        "Return True if all elements of self are less than corresponding element in rhs."
        return self < rhs
    
    @vmathop
    @boolop('all')
    def __gt__(self, rhs: Union[Union[int, float, complex], Iterable]) -> bool:
        "Return True if all elements of self are greater than corresponding element in rhs."
        return self > rhs
    
    @vmathop
    @boolop('all')
    def __le__(self, rhs: Union[Union[int, float, complex], Iterable]) -> bool:
        "Return True if all elements of self are less than or equal to corresponding element."
        return self <= rhs
    
    @vmathop
    @boolop('all')
    def __ge__(self, rhs: Union[Union[int, float, complex], Iterable]) -> bool:
        "Return True if all elements of self are greater than or equal to corresponding element."
        return self >= rhs
    
    def __hash__(self) -> int:
        "Return hash of internal data"
        return hash(self.__v)
    
    def set_length(self, new_length: Union[int, float, complex]) -> 'Vector':
        "Set length of this vector by normalizing it and then scaling it. **IN PLACE**"
        self.__v = self.dtype(self * new_length / self.magnitude)
        return self
    
    def as_length(self, length: Union[int, float, complex]) -> 'Vector':
        "Return this vector scaled to new length"
        return self * (length / self.magnitude)
    
    get_length = lambda self: self.magnitude
    
    def lerp(self, other: Iterable, value: float) -> 'Vector':
        "Return linear interpolation between self and another vector. Value is float from 0 to 1."
        if value < 0 or value > 1:
            raise ValueError('Lerp value is not a float in range of 0 to 1!')
        return self + (other - self) * value
    
    @onlylen(3)
    @vmathop
    def cross(self, other: Iterable) -> 'Vector':
        "Returns the cross product of this vector with another IF both are 3d vectors"
        # pylint: disable=C0103
        x, y, z = self.__v
        bx, by, bz = other
        return self.__class__( y*bz - by*z,
                               z*bx - bz*x,
                               x*by - bx*y )
    
    def dot(self, other: Union[Iterable, int, float]) -> float:
        "Return the dot product of this vector with another"
##        return sum(self * other)
        return math.fsum(self * other)
    
    def __matmul__(self, rhs):
        "Return dot product of self and right hand side."
        return self.dot(rhs)
    
    @onlylen(2)
    def get_heading(self) -> float:
        "Returns the arc tangent (measured in radians) of self.y/self.x."
        # pylint: disable=C0103
        x, y = self.__v
        return math.atan2(y, x)
    
    @onlylen(1)
    def __index__(self) -> int:
        "Return value of self as int"
        if not isinstance(self[0], (int, float)):
            raise ValueError('Value is not an intiger or float.')
        return int(self[0])
    
    @onlylen(1)
    def __float__(self) -> float:
        "Return value of self as float"
        if not isinstance(self[0], (int, float)):
            raise ValueError('Value is not an intiger or float.')
        return float(self[0])
    
    @mapop
    def conv_ints(self):
        "Return copy of self, but all items are intigers"
        return int(self)
    
    @mapop
    def conv_floats(self):
        "Return copy of self, but all items are floats"
        return float(self)
    
    # Intiger operators
    @vmathop
    @onlytypemath(int)
    @simpleop
    def __and__(self, rhs: Union[int, Iterable[int]]) -> 'Vector':
        "Return bitwise AND of self and rhs if both are composed of intigers"
        return self & rhs
    __rand__ = __and__
    
    @vmathop
    @onlytypemath(int)
    @simpleop
    def __or__(self, rhs: Union[int, Iterable[int]]) -> 'Vector':
        "Return bitwise OR of self and rhs if both are composed of intigers"
        return self | rhs
    __ror__ = __or__
    
    @vmathop
    @onlytypemath(int)
    @simpleop
    def __lshift__(self, rhs: Union[int, Iterable[int]]) -> 'Vector':
        "Return bitwise left shift of self by rhs if both are composed of intigers"
        return self << rhs
    
    @vmathop
    @onlytypemath(int)
    @simpleop
    def __rlshift__(self, lhs: Union[int, Iterable[int]]) -> 'Vector':
        "Bitwise left shift but from left hand side"
        return lhs << self
    
    @vmathop
    @onlytypemath(int)
    @simpleop
    def __rshift__(self, rhs: Union[int, Iterable[int]]) -> 'Vector':
        "Return bitwise right shift of self by rhs if both are composed of intigers"
        return self >> rhs
    
    @vmathop
    @onlytypemath(int)
    @simpleop
    def __rrshift__(self, lhs: Union[int, Iterable[int]]) -> 'Vector':
        "Bitwise right shift but from left hand side"
        return lhs >> self
    
    @vmathop
    @onlytypemath(int)
    @simpleop
    def __xor__(self, rhs: Union[int, Iterable[int]]) -> 'Vector':
        return self ^ rhs
    __rxor__ = __xor__

def get_surface_normal(vec1: Vector, vec2: Vector, vec3: Vector) -> Vector:
    "Return the surface normal of a triangle."
    # pylint: disable=line-too-long
##    return Vector(
##        vec1[1] * (vec2[2] - vec3[2]) + vec2[1] * (vec3[2] - vec1[2]) + vec3[1] * (vec1[2] - vec2[2]),
##        vec1[2] * (vec2[0] - vec3[0]) + vec2[2] * (vec3[0] - vec1[0]) + vec3[2] * (vec1[0] - vec2[0]),
##        vec1[0] * (vec2[1] - vec3[1]) + vec2[0] * (vec3[1] - vec1[1]) + vec3[0] * (vec1[1] - vec2[1])
##    )
    return (vec2-vec1).cross(vec3-vec1)

class Vector1(Vector):
    "Vector1. Same as Vector, but stuck being 1d and has x attribute."
    __slots__: tuple = tuple()
    def __init__(self, x: Union[int, float, complex]=0, **kwargs):
        if not 'dtype' in kwargs:
            kwargs['dtype'] = list
        super().__init__(x, **kwargs)
    
    @classmethod
    def from_iter(cls, iterable: Iterable, dims: int=None, dtype: type=list) -> 'Vector1':
        "Return Vector1 from iterable."
        nxt = iter(iterable).__next__
        return cls(nxt(), dtype=dtype)
    
    def _get_x(self) -> Union[int, float, complex]:
        return self[0]
    
    def _set_x(self, value: Union[int, float, complex]) -> None:
        self[0] = value
    
    x = property(_get_x, _set_x, doc='X component')

class Vector2(Vector1):
    "Vector2. Same as Vector, but stuck being 2d and has x and y attributes."
    __slots__: tuple = tuple()
    def __init__(self, x: Union[int, float, complex]=0, y: Union[int, float, complex]=0, **kwargs):
        # pylint: disable=super-init-not-called,non-parent-init-called
        if not 'dtype' in kwargs:
            kwargs['dtype'] = list
        Vector.__init__(self, x, y, **kwargs)
    
    @classmethod
    def from_iter(cls, iterable: Iterable, dims: int=None, dtype: type=list, **_) -> 'Vector2':
        "Return Vector2 from iterable."
        nxt = iter(iterable).__next__
        return cls(nxt(), nxt(), dtype=dtype)
    
    def _get_y(self) -> Union[int, float, complex]:
        return self[1]
    
    def _set_y(self, value: Union[int, float, complex]) -> None:
        self[1] = value
    
    y = property(_get_y, _set_y, doc='Y component')

class Vector3(Vector2):
    "Vector3. Same as Vector, but stuck being 3d and has x, y, and z attributes."
    __slots__: tuple = tuple()
    def __init__(self, x: Union[int, float, complex]=0, y: Union[int, float, complex]=0, z: Union[int, float, complex]=0, **kwargs):
        # pylint: disable=super-init-not-called,non-parent-init-called
        if not 'dtype' in kwargs:
            kwargs['dtype'] = list
        Vector.__init__(self, x, y, z, **kwargs)
    
    @classmethod
    def from_iter(cls, iterable: Iterable, dims: int=None, dtype: type=list, **_) -> 'Vector3':
        "Return Vector3 from iterable."
        nxt = iter(iterable).__next__
        return cls(nxt(), nxt(), nxt(), dtype=dtype)
    
    def _get_z(self) -> Union[int, float, complex]:
        return self[2]
    
    def _set_z(self, value: Union[int, float, complex]) -> None:
        self[2] = value
    
    z = property(_get_z, _set_z, doc='Z component')

class Vector4(Vector3):
    "Vector4. Same as Vector, but stuck being 4d and has x, y, z, and w attributes."
    __slots__: tuple = tuple()
    def __init__(self, x: Union[int, float, complex]=0, y: Union[int, float, complex]=0, z: Union[int, float, complex]=0, w: Union[int, float, complex]=0, **kwargs):
        # pylint: disable=super-init-not-called,non-parent-init-called
        if not 'dtype' in kwargs:
            kwargs['dtype'] = list
        Vector.__init__(self, x, y, z, w, **kwargs)
    
    @classmethod
    def from_iter(cls, iterable: Iterable, dims: int=None, dtype: type=list, **_) -> 'Vector4':
        "Return Vector4 from iterable."
        nxt = iter(iterable).__next__
        return cls(nxt(), nxt(), nxt(), nxt(), dtype=dtype)
    
    def _get_w(self) -> Union[int, float, complex]:
        return self[3]
    
    def _set_w(self, value: Union[int, float, complex]) -> None:
        self[3] = value
    
    w = property(_get_w, _set_w, doc='W component')

class Vector5(Vector3):
    "Vector5. Same as Vector, but stuck being 5d and has x, y, z, u, and v attributes."
    __slots__: tuple = tuple()
    def __init__(self, x: Union[int, float, complex]=0, y: Union[int, float, complex]=0, z: Union[int, float, complex]=0, u: Union[int, float, complex]=0, v: Union[int, float, complex]=0, **kwargs):
        # pylint: disable=super-init-not-called,non-parent-init-called,too-many-arguments
        if not 'dtype' in kwargs:
            kwargs['dtype'] = list
        Vector.__init__(self, x, y, z, u, v, **kwargs)
    
    @classmethod
    def from_iter(cls, iterable: Iterable, dims: int=None, dtype: type=list, **_) -> 'Vector5':
        "Return Vector5 from iterable."
        nxt = iter(iterable).__next__
        return cls(nxt(), nxt(), nxt(), nxt(), nxt(), dtype=dtype)
    
    def _get_u(self) -> Union[int, float, complex]:
        return self[3]
    
    def _set_u(self, value: Union[int, float, complex]) -> None:
        self[3] = value
    
    def _get_v(self) -> Union[int, float, complex]:
        return self[4]
    
    def _set_v(self, value: Union[int, float, complex]) -> None:
        self[4] = value
    
    u = property(_get_u, _set_u, doc='U component')
    v = property(_get_v, _set_v, doc='V component')

if __name__ == '__main__':
    print(f'{__title__} v{__version__}\nProgrammed by {__author__}.')
