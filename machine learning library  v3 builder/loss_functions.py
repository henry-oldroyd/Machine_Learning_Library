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

    # @abc.abstractclassmethod
    # def get_mean_cost_variance_cost(self) -> tuple[float, float]:
    #     pass

    def __init__(self) -> None:
        self._is_compiled = False
        self._vector_size = None
        self._batch_size = None
        self._full_validation = True
        self._loss_is_computed = False


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Loss_Function_Interface):
            return False
        

        # clear cache and don't compare by it
        self._loss_is_computed = False
        other._loss_is_computed = False

        if any((
            self._is_compiled != other._is_compiled,
            self._vector_size != other._vector_size,
            self._batch_size != other._batch_size,
            self._full_validation != other._full_validation
        )):
            return False

        return True

    def set_vector_size(self, vector_size: int) -> Loss_Function_Interface:
        if not isinstance(vector_size, int): 
            raise TypeError(f"vector_size must be an intiger")
        if not (1 <= vector_size): 
            raise ValueError(f"vector_size must be a natural number >=1")
        
        self._is_compiled = self._is_compiled and (self._vector_size == vector_size)
        self._vector_size = vector_size
        return self


    def set_batch_size(self, batch_size: int) -> Loss_Function_Interface:
        if not isinstance(batch_size, int): 
            raise TypeError(f"batch_size must be an intiger")
        if not (1 <= batch_size): 
            raise ValueError(f"batch_size must be a natural number >=1")

        self._is_compiled = self._is_compiled and (self._batch_size == batch_size)
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





# # not sure if this function still works after trying to make its time to execute linear with batch size
# # create a function for mean varaince cost where cost is mean squared error
# class MVC_MSE(Loss_Function_Interface):
#     def __init__(self, variance_weighting: float = None, mean_weighting: float = None) -> None:
#         # defensive design allow ints

#         parameters_to_examine = {
#             "variance_weighting": variance_weighting,
#             "mean_weighting": mean_weighting,
#         }
#         for p_name, p_value in parameters_to_examine.items():
#             if p_value is not None:
#                 if not (isinstance(p_value, float) or isinstance(p_value, int)):
#                     raise TypeError(f"{p_name} must be of type float (or int)")
#                 if 0 > p_value:
#                     raise ValueError(f"{p_name} must be a positive or 0 value")

#         variance_weighting = float(variance_weighting) if variance_weighting is not None else 0.0
#         mean_weighting = float(mean_weighting) if mean_weighting is not None else 1.0


#         if round(variance_weighting + mean_weighting, 8) == 0:
#             raise ValueError("Invalid parameters, cannot have both mean and variance weighting cannot be 0")


#         super().__init__()
#         self._variance_weighting = variance_weighting
#         self._mean_weighting = mean_weighting

#         self._loss_computed = False
#         self._loss_grad_computed = False

#         self._error = None



#     def __eq__(self, other: object) -> bool:
#         if not super().__eq__(other):
#             return False
        
#         if not isinstance(other, MVC_MSE):
#             return False
        
#         def floats_are_equal(a, b):
#             return round(a-b, 8) == 0.0

#         if any((
#             not floats_are_equal(self._variance_weighting, other._variance_weighting),
#             not floats_are_equal(self._mean_weighting, other._mean_weighting),
#         )):
#             return False


#     @requires_compilation
#     def compute_loss(self, P: np.ndarray, Y: np.ndarray) -> np.floating:
#         self._validate_inputs(P=P, Y=Y)

#         if not np.array_equal(self._error, (P-Y)):
#             self._loss_computed = False
#             self._loss_grad_computed = False
        
#         if self._loss_computed:
#             return self._loss

#         self._matrix_size = self._batch_size * self._vector_size
#         self._error = (P-Y)


# #         self._mean_cost_across_batch = np.trace(self._error_product) / self._matrix_size

# #         self._mean_of_squared_cost_across_batch = np.trace(self._error_product.T @ self._error_product) / self._matrix_size


#         self._mean_cost_across_batch = np.sum(self._error**2) / self._matrix_size

#         self._mean_of_squared_cost_across_batch = np.sum(self._error**4) / self._matrix_size

#         self._variance_cost_across_batch = self._mean_of_squared_cost_across_batch - self._mean_cost_across_batch**2


#         self._processed_variance = (self._variance_cost_across_batch + 1)**(self._variance_weighting/2)
#         self._processed_mean = self._mean_cost_across_batch**self._mean_weighting

#         self._overall_product_exponent = 1/(self._mean_weighting + self._variance_weighting) if round(self._mean_weighting + self._variance_weighting, 8) != 0 else 1

#         self._loss = (self._mean_cost_across_batch * self._processed_variance)**self._overall_product_exponent

#         # self._processed_variance = (self._variance_cost_across_batch)*self._variance_weighting
#         # self._overall_sum_multiplier = 1/(1+self._variance_weighting)
#         # loss = (self._mean_cost_across_batch + self._processed_variance) * self._overall_sum_multiplier


#         self._loss_computed = True
#         return self._loss
 

    
#     @requires_loss_computation
#     @requires_compilation
#     def compute_loss_gradient(self) -> np.ndarray:
#         if self._loss_grad_computed:
#             return self._loss_gradient
        
# #         dMCdP = (2/self._matrix_size) * self._error
# #         dVCdP = (2/self._matrix_size) * ( (self._error @ self._error_product) - (2 * self._mean_cost_across_batch / self._matrix_size) * self._error )



#         dMCdP = (2/self._matrix_size) * self._error
#         dVCdP = (4/self._matrix_size) * np.sum(self._error **3) - 2 * self._mean_cost_across_batch * dMCdP




#         if round(self._variance_weighting, 8) == 0:
#             dldP = dMCdP
#         else:
#             # loss = MC * PVC
#             # by produce rule
#             dPVCdP = (self._variance_cost_across_batch + 1)**((self._variance_weighting/2) -1) * dVCdP if round(self._variance_weighting, 8) != 0 else 0
#             dPMCdP = (self._mean_cost_across_batch)**(self._mean_weighting -1) * dMCdP if round(self._mean_weighting, 8) != 0 else 0

#             dldP = (self._mean_cost_across_batch * self._processed_variance)**(self._overall_product_exponent-1) * (dPMCdP * self._processed_variance + self._mean_cost_across_batch * dPVCdP)
            

#             # dPVCdP = self._variance_weighting * dVCdP
#             # dldP = self._overall_sum_multiplier * (dMCdP + dPVCdP)
        
#         self._loss_grad_computed = True
#         self._loss_gradient = dldP
#         return self._loss_gradient
    
#     @requires_loss_computation
#     @requires_compilation
#     def get_mean_cost_variance_cost(self) -> tuple[float, float]:
#         return self._mean_cost_across_batch, self._variance_cost_across_batch
    




# class MC_MSE(MVC_MSE):
#     def __init__(self) -> None:
#         super().__init__(
#             variance_weighting=0,
#             mean_weighting=1
#         )
    

# create a cost function mean cost where cost is mean squared error
class MC_MSE(Loss_Function_Interface):
    @requires_compilation
    def compute_loss(self, P: np.ndarray, Y: np.ndarray) -> np.floating:
        self._validate_inputs(P=P, Y=Y)


        self._matrix_size = self._batch_size * self._vector_size
        self._error = (P-Y)

        loss = (1/self._matrix_size) * np.sum(self._error ** 2)
    
        self._loss_computed = True
        return loss

    @requires_loss_computation
    @requires_compilation
    def compute_loss_gradient(self) -> np.ndarray:
        return (2/self._matrix_size) * self._error


