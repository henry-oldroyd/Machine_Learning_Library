# import libraries
import abc
import numpy as np

# create abstract base class for a cost function object
class Abstract_Loss_Function_Interface(abc.ABC):
    @abc.abstractmethod
    def compute_loss(self, P: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def compute_loss_gradient(self, P: np.ndarray = None, Y: np.ndarray = None) -> np.ndarray:
        pass


    def __init__(self, batch_size: int, vector_size: int):
        self._vector_size = vector_size
        self._batch_size = batch_size

    def _validate_inputs(self, P: np.ndarray, Y: np.ndarray) -> np.floating:
        # complete type validation
        if not isinstance(P, np.ndarray): 
            raise TypeError(f"P must be a numpy array")
        if not isinstance(Y, np.ndarray):
            raise TypeError(f"Y must be a numpy array")
        
        # complete dimension validation
        expected_shape = (self._vector_size, self._batch_size)
        if P.shape != expected_shape:
            raise ValueError(f"P must be of dimensions (vector_size, batch_size) = {expected_shape}")
        if Y.shape != expected_shape:
            raise ValueError(f"Y must be of dimensions (vector_size, batch_size) = {expected_shape}")


# create a cost function MSE
class MSE(Abstract_Loss_Function_Interface):
    def compute_loss(self, P: np.ndarray=None, Y: np.ndarray=None) -> np.floating:
        self._validate_inputs(P=P, Y=Y)

        self._diff = (P-Y)

        return (1/self._batch_size) * np.trace(self._diff.T @ self._diff)

    def compute_loss_gradient(self, P: np.ndarray=None, Y: np.ndarray=None) -> np.floating:
        if (P is None) ^ (Y is None):
            raise ValueError(f"MSE.compute_loss: Either both P and Y must be provided as parameters of neither")
        
        if P is not None:
            self._validate_inputs(P=P, Y=Y)
            self._diff = (P-Y)

        return (2/self._batch_size) * self._diff


