# import libraries
from __future__ import annotations
import abc
import numpy as np
from math import tanh, exp

from ml_exceptions import ObjectCompiliationError, DimensionError
from decorators import requires_compilation, if_full_validation, requires_activation_computation

# create abstract base class for a cost function object
class Activation_Function_Interface(abc.ABC):
    @abc.abstractmethod
    def compute_activation(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def compute_activation_gradient(self) -> np.ndarray:
        pass

    def __init__(self) -> None:
        self._is_compiled = False
        self._vector_size = None
        self._batch_size = None
        self._X = None
        self._full_validation = True
        self._activation_is_computed = False


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Activation_Function_Interface):
            return False

        # delete cache on both and don't check it
        self._activation_is_computed = False
        other._activation_is_computed = False
        if any((
            self._is_compiled != other._is_compiled,
            self._vector_size != other._vector_size,
            self._batch_size != other._batch_size,          
        )):
            return False
        
        return True

    def set_vector_size(self, vector_size: int) -> Activation_Function_Interface:
        if not isinstance(vector_size, int): 
            raise TypeError(f"vector_size must be an intiger")
        if not (1 <= vector_size): 
            raise ValueError(f"vector_size must be a natural number >=1")
        
        self._is_compiled = self._is_compiled and (self._vector_size == vector_size)
        self._vector_size = vector_size
        return self

    def set_batch_size(self, batch_size: int) -> Activation_Function_Interface:
        if not isinstance(batch_size, int): 
            raise TypeError(f"batch_size must be an intiger")
        if not (1 <= batch_size): 
            raise ValueError(f"batch_size must be a natural number >=1")
        
        self._is_compiled = self._is_compiled and (self._batch_size == batch_size)
        self._batch_size = batch_size
        return self    
    
    def set_full_validation(self, full_validation: bool) -> Activation_Function_Interface:
        if not isinstance(full_validation, bool): 
            raise TypeError(f"Full validation must be of type bool")
        self._full_validation = full_validation
        # recompilation not needed so don't check or change compile flag
        return self

    def compile(self) -> Activation_Function_Interface:
        # assume not compiled
        if self._is_compiled:
            return self
        
        # complete presence checks
        properties_to_examine = {
            "vector_size": self._vector_size,
            "batch_size": self._batch_size,
        }
        for property_name, property_value in properties_to_examine.items():
            if property_value is None:
                raise ObjectCompiliationError(f"Cannot compile object without {property_name} provided")

        # compute dependant properties
        self._expected_X_shape = (self._vector_size, self._batch_size)

        # declare compiled
        self._is_compiled = True

        return self

    @if_full_validation
    def _validate_X(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"parameter X must be of type numpy array")
        if X.shape != self._expected_X_shape:
            raise DimensionError(f"X.shape not the expected: {self._expected_X_shape}")

        if not all(isinstance(element, np.floating) for row in X for element in row):
            raise TypeError(f"X must contain exclusivly floats of type np.floating")


class Identity_Function(Activation_Function_Interface): 
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if not isinstance(other, Identity_Function):
            return False
        return True

    @requires_compilation
    def compute_activation(self, X: np.ndarray) -> np.ndarray:
        self._validate_X(X)
        self._X = X

        return self._X

    @requires_activation_computation
    @requires_compilation
    def compute_activation_gradient(self) -> np.ndarray:        
        return np.array([
            np.eye(self._X.shape[0], self._X.shape[0])
            for _ in range(self._batch_size)
        ]).reshape((
            self._vector_size,
            self._vector_size,
            self._batch_size
        ))
    


class Element_Wise_Activation_Function_Interface(Activation_Function_Interface):
    @abc.abstractmethod
    def _scalar_activation_function(x: float) -> float:
        pass

    @abc.abstractmethod
    def _scalar_activation_derivative_function(x: float) -> float:
        pass

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if not isinstance(other, Element_Wise_Activation_Function_Interface):
            return False
        return True

    def __init__(self):
        super().__init__()
        self._X = None
    
        self._vector_activation_function = np.vectorize(self._scalar_activation_function) 
        self._vector_activation_derivative_function = np.vectorize(self._scalar_activation_derivative_function) 


    @requires_compilation
    def compute_activation(self, X: np.ndarray) -> np.ndarray:
        self._validate_X(X)

        self._X = X
        return self._vector_activation_function(self._X)

    @requires_activation_computation
    @requires_compilation
    def compute_activation_gradient(self) -> np.ndarray:
        return np.array([
            np.diag(
                self._vector_activation_derivative_function(self._X[:,i])
            )            
            for i in range(self._batch_size)
        ]).reshape((
            self._vector_size,
            self._vector_size,
            self._batch_size
        ))
    

class Sigmoid(Element_Wise_Activation_Function_Interface):
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if not isinstance(other, Sigmoid):
            return False

    def _scalar_activation_function(self, x: float) -> float:
        return 1/(1 + exp(-x))
        
    def _scalar_activation_derivative_function(self, x: float) -> float:
        return self._scalar_activation_function(x) * (1 - self._scalar_activation_function(x))



class TANH(Element_Wise_Activation_Function_Interface):
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if not isinstance(other, TANH):
            return False

    def _scalar_activation_function(self, x: float) -> float:
        return tanh(x)
        
    def _scalar_activation_derivative_function(self, x: float) -> float:
        return 1-(tanh(x)**2)


class Adaptive_RELU(Element_Wise_Activation_Function_Interface):
    def __init__(self, left_slope: float, right_slope: float, discontinuity_point: float):
        parameters =  {"left_slope": left_slope, "right_slope": right_slope, "discontinuity_point": discontinuity_point}
        for p_name, p_value in parameters.items():
            if not isinstance(p_value, float):
                raise TypeError(f"pratameter {p_name} must be of type float")

        super().__init__()
        self._left_slope = left_slope
        self._right_slope = right_slope
        self._discontinuity_point = discontinuity_point

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if not isinstance(other, Adaptive_RELU):
            return False
        
        if any((
            self._left_slope != other._left_slope,
            self._right_slope != other._right_slope,
            self._discontinuity_point != other._discontinuity_point,
        )):
            return False

        return True

    def _scalar_activation_function(self, x: float) -> float:
        if x < self._discontinuity_point:
            return self._left_slope*x
        else:
            return self._right_slope*x
        
    def _scalar_activation_derivative_function(self, x: float) -> float:
        if x < self._discontinuity_point:
            return self._left_slope
        else:
            return self._right_slope
        
class Leaky_RELU(Adaptive_RELU):
    def __init__(self):
        super().__init__(
            left_slope=0.01, 
            right_slope=1.0, 
            discontinuity_point=0.0
        )

class RELU(Adaptive_RELU):
    def __init__(self):
        super().__init__(
            left_slope=0.0, 
            right_slope=1.0, 
            discontinuity_point=0.0
        )

class Absolute(Adaptive_RELU):
    def __init__(self):
        super().__init__(
            left_slope=-1.0, 
            right_slope=1.0, 
            discontinuity_point=0.0
        )

    
    

