# import libraries
from __future__ import annotations
import abc
import numpy as np

from ml_exceptions import ObjectCompiliationError, DimensionError
from decorators import requires_compilation, if_full_validation, requires_loss_computation


# create abstract base class for a cost function object
class Loss_Function_Interface(abc.ABC):
    @abc.abstractmethod
    def compute_loss(self, P: np.ndarray, Y: np.ndarray) -> np.floating:
        pass

    @abc.abstractmethod
    def compute_loss_gradient(self) -> np.ndarray:
        pass

    # def compute_cost_variance(self) -> np.ndarray:
    #     pass

    def __init__(self) -> None:
        self._is_compiled = False
        self._vector_size = None
        self._batch_size = None
        self._full_validation = True
        self._loss_is_computed = False

    def set_vector_size(self, vector_size: int) -> Loss_Function_Interface:
        if not isinstance(vector_size, int): 
            raise TypeError(f"vector_size must be an intiger")
        if not (1 <= vector_size): 
            raise ValueError(f"vector_size must be a natural number >=1")
        self._is_compiled = False
        self._vector_size = vector_size
        return self

    def set_batch_size(self, batch_size: int) -> Loss_Function_Interface:
        if not isinstance(batch_size, int): 
            raise TypeError(f"batch_size must be an intiger")
        if not (1 <= batch_size): 
            raise ValueError(f"batch_size must be a natural number >=1")
        self._is_compiled = False
        self._batch_size = batch_size
        return self

    def set_full_validation(self, full_validation: bool) -> Loss_Function_Interface:
        if not isinstance(full_validation, bool): 
            raise TypeError(f"Full validation must be of type bool")
        self._full_validation = full_validation
        # recompilation not needed so don't check or change compile flag
        return self

    def compile(self) -> Loss_Function_Interface:
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
            
        # define dependant properties
        self._expected_shape_PY = (self._vector_size, self._batch_size)

        # declare compiled
        self._is_compiled = True

        return self

    @if_full_validation
    def _validate_inputs(self, P: np.ndarray, Y: np.ndarray) -> None:
        parameters_to_examine = {
            "P": P,
            "Y": Y
        }
        for p_name, p_value in parameters_to_examine.items():
            if not isinstance(p_value, np.ndarray):
                raise TypeError(f"{p_name} must be a numpy array")

            if not all(isinstance(element, np.floating) for row in p_value for element in row):
                raise TypeError(f"{p_value} must contain exclusivly floats of type np.floating")

            if p_value.shape != self._expected_shape_PY:
                raise DimensionError(f"{p_name} must be of dimensions {self._expected_shape_PY}")


# create a cost function mean cost where cost is mean squared error
class MC_MSE(Loss_Function_Interface):
    @requires_compilation
    def compute_loss(self, P: np.ndarray, Y: np.ndarray) -> np.floating:
        self._validate_inputs(P=P, Y=Y)


        self._matrix_size = self._batch_size * self._vector_size
        self._error = (P-Y)

        loss = (1/self._matrix_size) * np.trace(self._error.T @ self._error)
    
        self._loss_computed = True
        return loss

    @requires_loss_computation
    @requires_compilation
    def compute_loss_gradient(self) -> np.ndarray:
        return (2/self._matrix_size) * self._error


# create a function for mean varaince cost where cost is mean squared error
class MVC_MSE(Loss_Function_Interface):
    @requires_compilation
    def compute_loss(self, P: np.ndarray, Y: np.ndarray) -> np.floating:
        self._validate_inputs(P=P, Y=Y)

        self._matrix_size = self._batch_size * self._vector_size
        self._error = (P-Y)
        self._error_product = self._error.T @ self._error

        self._mean_cost_across_batch = np.trace(self._error_product) / self._matrix_size
        self._mean_of_squared_cost_across_batch = np.trace(self._error_product.T @ self._error_product) / self._matrix_size

        self._variance_cost_across_batch = self._mean_of_squared_cost_across_batch - self._mean_cost_across_batch**2

        # option 1
        loss = self._mean_cost_across_batch * (self._variance_cost_across_batch + 1)**(1/2)
        
        # option 2
        # loss = self._mean_cost_across_batch * (self._variance_cost_across_batch + 1)**1/4 

        # option 3 MSE
        # loss = self._mean_cost_across_batch



        self._loss_computed = True
        return loss
 
    
    @requires_loss_computation
    @requires_compilation
    def compute_loss_gradient(self) -> np.ndarray:
        dMCdP = (2/self._matrix_size) * self._error
        dVCdP = (2/self._matrix_size) * ( (self._error @ self._error_product) - (2 * self._mean_cost_across_batch / self._matrix_size) * self._error )

        # option 1
        dldP = dMCdP * (self._variance_cost_across_batch + 1)**(1/2) 
        + self._mean_cost_across_batch * (self._variance_cost_across_batch + 1)**(-1/2) * dVCdP
        
        # option 2
        # dldP = dMCdP * (self._variance_cost_across_batch**1/4 + 1) + self._mean_cost_across_batch * (self._variance_cost_across_batch + 1)**-3/4 * dVCdP

        # option 3 MSE
        # dldP = dMCdP


        return dldP
    
    @requires_loss_computation
    @requires_compilation
    def get_mean_cost_variance_cost(self) -> (float, float):
        return self._mean_cost_across_batch, self._variance_cost_across_batch