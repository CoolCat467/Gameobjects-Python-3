#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Vector Module

"Vector module"

# Programmed by CoolCat467

__title__ = 'Vector'
__author__ = 'CoolCat467'
__version__ = '0.0.0'
_ver_major__ = 0
_ver_minor__ = 0
_ver_patch__ = 0

import math
from functools import wraps

def vmathop(function):
    "Vector math operator decorator"
    @wraps(function)
    def wrapped_op(self, rhs, *args, **kwargs):
        if hasattr(rhs, '__len__'):
            if len(rhs) == len(self):
                return function(self, rhs, *args, **kwargs)
            raise TypeError('Operand length is not same as own')
        return function(self, [rhs]*len(self), *args, **kwargs)
##        raise AttributeError('Operand has no length attribute')
    return wrapped_op

def simpleop(function):
    "Return new `self` class instance built from application of function on items of self and rhs."
    def apply(values):
        return function(*values)
    @wraps(function)
    def operator(self, rhs):
        return self.__class__(*map(apply, zip(self, rhs)), dtype=self.dtype)
    return operator

def onlylen(length):
    "Return wrapper that only runs function if length matches length."
    def wrapper(function):
        "Wrapper that only runs function if lengths match"
        @wraps(function)
        def wrapped_func(self, *args, **kwargs):
            if len(self) == length:
                return function(self, *args, **kwargs)
            raise TypeError(f'Vector is not a {length}d vector!')
        return wrapped_func
    return wrapper

def boolop(combine='all'):
    "Return vector boolian simple operator. Combine can by ('any', 'all')"
    if not combine in {'any', 'all'}:
        raise ValueError('Combine must be either "any" or "all"!')
    def wrapper(function):
        "Vector boolian operator decorator"
        def apply(values):
            return function(*values)
        if combine == 'any':
            def operator(self, rhs):
                return any(map(apply, zip(self, rhs)))
        else:
            def operator(self, rhs):
                return all(map(apply, zip(self, rhs)))
        return wraps(function)(operator)
    return wrapper

def to_float_int(value):
    "Convert value to float or int, whatever is more appropriate"
    value = float(value)
    if value.is_integer():
        value = int(value)
    return value

class Vector:
    """Vector Object. Takes n arguments as input and creates n length vector, or type length vector.
    dtype argument changes internal data type."""
    __slots__ = ('_v','dtype')
    def __init__(self, *args, dims=None, dtype=tuple):
        self.dtype = dtype
        if args:
            if hasattr(args[0], '__iter__'):
                args = tuple(args[0])
        args = tuple(map(to_float_int, args))
        if not hasattr(dtype, '__getitem__'):
            raise TypeError('Data type class is not subscriptable!')
        if dims is None:
##            self._v = tuple(args)
            self._v = self.dtype(args)
        else:
            args = args[:dims]
##            self._v = tuple(list(args) + [0]*(dims-len(args)))
            self._v = self.dtype(list(args) + [0]*(dims-len(args)))
    
    def __repr__(self):
        args = ', '.join(map(repr, self._v))
        return f'{self.__class__.__name__}({args})'
    
    def __len__(self):
        return len(self._v)
    
    def __iter__(self):
        return iter(self._v)
    
    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError('Index is not an intiger.')
        if index > len(self):
            raise IndexError('Index out of range for this vector')
##        if isinstance(index, str):
##            index = ord(index.upper())-88
        return self._v[index]
    
    def __setitem__(self, index, value):
        if not hasattr(self._v, '__setitem__'):
            dtype = self.dtype.__name__
            raise TypeError(f"'{dtype}' does not support item assignment. Change vector dtype.")
        if not isinstance(index, int):
            raise TypeError('Index is not an intiger.')
        if index > len(self):
            raise IndexError('Index out of range for this vector')
        self._v[index] = value
    
    @classmethod
    def from_iter(cls, iterable, dims=None, dtype=tuple):
        "Return Vector from iterable."
        return cls(*iterable, dims=dims, dtype=dtype)
    
    @classmethod
    def from_radians(cls, radians):
        "Return 2d unit vector from measure in radians"
        return cls(math.cos(radians), math.sin(radians))
    
    @property
    def magnitude(self):
        "Magnitude of this vector"
##        return math.sqrt(sum(self ** 2))
        return math.hypot(*self._v)
    
    def copy(self):
        "Return a copy of this vector"
        return self.from_iter(self._v, dtype=self.dtype)
    __copy__ = copy
    
    def reverse(self):
        "Return a copy of self, but order of elements is reversed."
        return self.from_iter(reversed(self))
    
    def normalize(self):
        "Normalize this vector."
        mag = self.magnitude
##        self._v = tuple(map(lambda x:x/mag, self._v))
        self._v = self.dtype(map(lambda x:x/mag, self._v))
##        self._v = list(map(lambda x:x/mag, self._v))
        return self
    
    def normalized(self):
        "Return normalized vector"
        mag = self.magnitude
        return self / mag
    
    def __neg__(self):
        "Return negated vector"
        return self.from_iter(-x for x in self._v)

    def __pos__(self):
        "Return unary positive of self"
        return self.from_iter(+x for x in self._v)
    
    def __abs__(self):
        "Return abs'd vector"
        return self.from_iter(abs(x) for x in self._v)
    
    def __round__(self):
        "Return vector but each element is rounded"
        return self.from_iter(round(x) for x in self._v)
    
    def __bool__(self):
        "Return True if any element is true, False otherwise"
        return any(self._v)
    
    @vmathop
    @simpleop
    def __add__(self, rhs):
        "Add two vectors/iterables or add number to each element"
        return self + rhs
    __radd__ = __add__
    
    @vmathop
    @simpleop
    def __sub__(self, rhs):
        "Subtract two vectors/iterables or subtract number from each element"
        return self - rhs
    
    @vmathop
    @simpleop
    def __rsub__(self, lhs):
        "Subtract but from left hand side"
        return lhs - self
    
    @vmathop
    @simpleop
    def __mul__(self, rhs):
        "Multiply two vectors/iterables or multiply each element by number"
        return self * rhs
    __rmul__ = __mul__
    
    @vmathop
    @simpleop
    def __truediv__(self, rhs):
        "Divide two vectors/iterables or divide each element by number"
        return self / rhs
    
    @vmathop
    @simpleop
    def __rtruediv__(self, lhs):
        "Division but from left hand side"
        return lhs / self
    
    @vmathop
    @simpleop
    def __pow__(self, rhs):
        "Get element to the power of number or matching item in vector/iterable for each element"
        return self ** rhs

    @vmathop
    @simpleop
    def __rpow__(self, lhs):
        "Power, but from left hand side"
        return lhs ** self
    
    @vmathop
    @boolop('all')
    def __eq__(self, rhs):
        return self == rhs
##        return self._v == rhs
    
    @vmathop
    @boolop('any')
    def __ne__(self, rhs):
        print((self, '!=', rhs))
        return self != rhs
##        return self._v != rhs
    
##    @vmagop
##    def __gt__(x, y):
##        return x > y
##    
##    @vmagop
##    def _ge__(x, y):
##        return x >= y
##    
##    @vmagop
##    def __lt__(x, y):
##        return x < y
##    
##    @vmagop
##    def __le__(x, y):
##        return x <= y
    
    def __hash__(self):
        return hash(self._v)
    
    def set_length(self, new_length):
        "Set length of this vector by normalizing it and then scaleing it."
        cmag = self.magnitude
        if cmag == 0:
            mag = 0
        else:
            mag = new_length / cmag
##        self._v = (self * ([mag]*len(self)))._v
        return self * mag
    
    @onlylen(3)
    @vmathop
    def cross(self, other):
        "Returns the cross product of this vector with another IF both are 3d vectors"
        # pylint: disable=C0103
        x, y, z = self
        bx, by, bz = other
        return self.__class__( y*bz - by*z,
                               z*bx - bz*x,
                               x*by - bx*y )
    
    def dot(self, other):
        "Return the dot product of this vector with another"
##        return sum(self * other)
        return math.fsum(self * other)
    
    @onlylen(2)
    def get_heading(self):
        "Returns the arc tangent (measured in radians) of self.y/self.x."
        # pylint: disable=C0103
        x, y = self
        return math.atan2(y, x)

class Vector1(Vector):
    "Vector1. Same as Vector, but stuck being 1d and has x attribute."
    _gameobjects_vector = 1
    def __init__(self, x=0, **kwargs):
##        kwargs['dims'] = 1
        if not 'dtype' in kwargs:
            kwargs['dtype'] = list
        super().__init__(x, **kwargs)
    
    @classmethod
    def from_iter(cls, iterable, dtype=list):
        "Return Vector2 from iterable."
        nxt = iter(iterable).__next__
        return cls(nxt(), dtype=dtype)
    
    def _get_x(self):
        return self[0]
    
    def _set_x(self, value):
        self[0] = value
    
    x = property(_get_x, _set_x, doc='X component')
    
    def __iadd__(self, rhs):
        self._v = self.dtype(self + rhs)
        return self
    def __isub__(self, rhs):
        self._v = self.dtype(self - rhs)
        return self
    def __imul__(self, rhs):
        self._v = self.dtype(self * rhs)
        return self
    def __idiv__(self, rhs):
        self._v = self.dtype(self / rhs)
        return self
    
    def _get_length(self):
        "Return length of this vector (same as self.magnitude property)"
        return self.magnitude
    get_length = _get_length
    get_magnitude = get_length
    def _set_length(self, length):
        self._v = self.from_iter( self.set_length(length) )
    length = property(_get_length, _set_length, None, "Length of the vector")
    
    @classmethod
    def from_floats(cls, *args):
        return self.from_iter(args)
    
    _from_float_sequence = from_iter
    
    @classmethod
    def from_points(cls, point1, point2):
        """Creates a Vector2 object between two points.
        @param p1: First point
        @param p2: Second point

        """
        point1 = cls.from_iter(point1)
        point2 = cls.from_iter(point2)
        return point2 - point1
    
    def __str__(self):
        args = map(str, self)
        return '('+', '.join(args)+')'
    
    def __call__(self, keys):
        """Used to swizzle a vector.

        @type keys: string
        @param keys: A string containing a list of component names
        >>> vec = Vector(1, 2)
        >>> vec('yx')
        (1, 2)

        """
        
        ord_x = ord('x')
        return tuple( self[ord(c) - ord_x] for c in keys )
    
    def as_tuple(self):
        return tuple(self)
    
    def normalize(self):
        return super().normalize()
    def get_normalised(self):
        return self.normalized()
    get_normalized = get_normalised
    normalise = normalize
    unit = get_normalised
    
    def get_distance_to(self, point):
        "Return distance to point"
        return (point - self).magnitude
    
    def set(self, *args):
        for idx, val in enumerate(args):
            if idx > len(self):
                break
            self[idx] = val
    
    def scalar_mul(self, scalar):
        self.__imul__(scalar)
##        return self *= scalar
        return self
    vector_mul = scalar_mul
    
    def get_scalar_mul(self, scalar):
        return self * scalar
    get_vector_mul = get_scalar_mul
    
    def scalar_div(self, scalar):
##        return self /= scalar
        self.__idiv__(scalar)
        return self
    vector_div = scalar_div
    
    def get_scalar_div(self, scalar):
        return self / scalar
    get_vector_div = get_scalar_div
    scale = scalar_mul
    
    def get_distance_to_squared(self, point):
        return self.dot(point)
    
    def cross_tuple(self, other):
        return tuple(self.cross(other))

class Vector2(Vector1):
    "Vector2. Same as Vector, but stuck being 2d and has x and y attributes."
    _gameobjects_vector = 2
    def __init__(self, x=0, y=0, **kwargs):
##        kwargs['dims'] = 2
        if not 'dtype' in kwargs:
            kwargs['dtype'] = list
        Vector.__init__(self, x, y, **kwargs)
    
    @classmethod
    def from_iter(cls, iterable, dtype=list):
        "Return Vector2 from iterable."
        nxt = iter(iterable).__next__
        return cls(nxt(), nxt(), dtype=dtype)
    
    def _get_y(self):
        return self[1]
    
    def _set_y(self, value):
        self[1] = value
    
    y = property(_get_y, _set_y, doc='Y component')

class Vector3(Vector2):
    "Vector3. Same as Vector, but stuck being 3d and has x, y, and z attributes."
    _gameobjects_vector = 3
    def __init__(self, x=0, y=0, z=0, **kwargs):
##        kwargs['dims'] = 3
        if not 'dtype' in kwargs:
            kwargs['dtype'] = list
        Vector.__init__(self, x, y, z, **kwargs)
    
    @classmethod
    def from_iter(cls, iterable, dtype=list):
        "Return Vector2 from iterable."
        nxt = iter(iterable).__next__
        return cls(nxt(), nxt(), nxt(), dtype=dtype)
    
    def _get_z(self):
        return self[2]
    
    def _set_z(self, value):
        self[2] = value
    
    z = property(_get_z, _set_z, doc='Z component')

class Vector4(Vector3):
    "Vector4. Same as Vector, but stuck being 4d and has x, y, z, and w attributes."
    _gameobjects_vector = 4
    def __init__(self, x=0, y=0, z=0, w=0, **kwargs):
##        kwargs['dims'] = 4
        if not 'dtype' in kwargs:
            kwargs['dtype'] = list
        Vector.__init__(self, x, y, z, w, **kwargs)
    
    @classmethod
    def from_iter(cls, iterable, dtype=list):
        "Return Vector2 from iterable."
        nxt = iter(iterable).__next__
        return cls(nxt(), nxt(), nxt(), nxt(), dtype=dtype)
    
    def _get_w(self):
        return self[3]
    
    def _set_w(self, value):
        self[3] = value
    
    w = property(_get_w, _set_w, doc='W component')

if __name__ == '__main__':
    print(f'{__title__} v{__version__}\nProgrammed by {__author__}.')
