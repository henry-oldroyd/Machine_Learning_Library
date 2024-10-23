# import libraries
import abc
import numpy as np

# create abstract base class for a cost function object
class Activation_Function_Interface(abc.ABC):
    @abc.abstractmethod
    def compute_activation(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def compute_activation_gradient(self, X: np.ndarray = None) -> np.ndarray:
        pass


class Identity_Function(Activation_Function_Interface):
    def __init__(self, batch_size: int, vector_size: int):
        self._vector_size = vector_size
        self._batch_size = batch_size

    def compute_activation(self, X: np.ndarray) -> np.ndarray:
        return X

    def compute_activation_gradient(self, X: np.ndarray = None) -> np.ndarray:
        derivative_matricies_by_batch_item = np.array([
            np.eye(X.shape[0], X.shape[0])
            for _ in range(self._batch_size)
        ])
        derivative_matricies_by_batch_item.reshape((
            self._vector_size,
            self._vector_size,
            self._batch_size
        ))
        return derivative_matricies_by_batch_item

class Element_Wise_Activation_Function_Interface(Activation_Function_Interface):
    @abc.abstractmethod
    def _scalar_activation_function(x: float) -> float:
        pass

    @abc.abstractmethod
    def _scalar_activation_derivative_function(x: float) -> float:
        pass

    def __init__(self, batch_size: int, vector_size: int):
        self._vector_size = vector_size
        self._batch_size = batch_size
        self._X = None
    
        self._vector_activation_function = np.vectorize(self._scalar_activation_function) 
        self._vector_activation_derivative_function = np.vectorize(self._scalar_activation_derivative_function) 

    def compute_activation(self, X: np.ndarray) -> np.ndarray:
        self._X = X
        return self._vector_activation_function(X)

    def compute_activation_gradient(self, X: np.ndarray = None) -> np.ndarray:
        if (X is None) and (self._X is None):
            raise ValueError(f"compute_activation_gradient: is X is not provided then it must have been cached from a previous activation call")
    
        X = self._X if X is None else X
        return np.array([
            np.diag(
                self._vector_activation_derivative_function(X[:,i])
            )            
            for i in range(self._batch_size)
        ]).reshape((
            self._vector_size,
            self._vector_size,
            self._batch_size
        ))


        # return self._vector_activation_derivative_function(X)


class Adaptive_RELU(Element_Wise_Activation_Function_Interface):
    def __init__(self, batch_size: int, vector_size: int, left_slope: float, right_slope: float, discontinuity_point: float):
        super().__init__(batch_size, vector_size)
        self._left_slope = left_slope
        self._right_slope = right_slope
        self._discontinuity_point = discontinuity_point

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
    def __init__(self, batch_size: int, vector_size: int):
        super().__init__(
            batch_size=batch_size,
            vector_size=vector_size, 
            left_slope=0.01, 
            right_slope=1.0, 
            discontinuity_point=0.0
        )

class RELU(Adaptive_RELU):
    def __init__(self, batch_size: int, vector_size: int):
        super().__init__(
            batch_size=batch_size,
            vector_size=vector_size, 
            left_slope=0.0, 
            right_slope=1.0, 
            discontinuity_point=0.0
        )

class Absolute(Adaptive_RELU):
    def __init__(self, batch_size: int, vector_size: int):
        super().__init__(
            batch_size=batch_size,
            vector_size=vector_size, 
            left_slope=-1.0, 
            right_slope=1.0, 
            discontinuity_point=0.0
        )

    
    

