# import libraries
from __future__ import annotations
import abc
import numpy as np
import random

from activation_functions import Activation_Function_Interface
from ml_exceptions import ObjectCompiliationError, DimensionError, DiverganceError
from decorators import requires_compilation, if_full_validation
from utility_functions import compare_np_array_with_cache


# # used for debugging
# seed = 239435901
# np.random.seed(seed)
# random.seed(seed)



class Layer_Interface(abc.ABC):
    @abc.abstractmethod
    def foreward_propagate(self, previous_activation: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def back_propagate(self, loss_activation_gradient: np.ndarray) -> np.ndarray:
        pass


    @abc.abstractmethod
    def compile(self) -> Layer_Interface: 
        pass

    def __init__(self) -> None:
        self._input_size = None
        self._output_size = None
        self._batch_size = None
        self._full_validation = True
        self._is_first_layer = False

        self._is_compiled = False
        self._keep_parameters_on_compilation = False


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layer_Interface):
            return False
        
        # clear cache of both for comparison

        if any((
            self._input_size != other._input_size,
            self._output_size != other._output_size,
            self._batch_size != other._batch_size,
            self._full_validation != other._full_validation,
            self._is_first_layer != other._is_first_layer,
            self._is_compiled != other._is_compiled,
            self._keep_parameters_on_compilation != other._keep_parameters_on_compilation,
        )):
            return False
        
        return True

    @requires_compilation
    def get_output_size(self) -> int:
        return self._output_size
    
    @requires_compilation
    def get_input_size(self) -> int:
        return self._input_size

    def set_batch_size(self, batch_size: int) -> Layer_Interface:
        if not isinstance(batch_size, int): 
            raise TypeError(f"batch_size must be an intiger")
        if not (1 <= batch_size): 
            raise ValueError(f"batch_size must be a natural number >=1")
        
        self._is_compiled = self._is_compiled and (self._batch_size == batch_size)
        self._batch_size = batch_size
        return self

    def set_input_size(self, input_size: int) -> Layer_Interface:
        if not isinstance(input_size, int): 
            raise TypeError(f"input_size must be an intiger")
        if not (1 <= input_size): 
            raise ValueError(f"input_size must be a natural number >=1")
        
        self._is_compiled = self._is_compiled and (self._input_size == input_size)
        self._keep_parameters_on_compilation = self._is_compiled
        self._input_size = input_size
        return self

    def set_full_validation(self, full_validation: bool) -> Neural_Layer:
        if not isinstance(full_validation, bool): 
            raise TypeError(f"full_validation must be of type bool")
        
        self._is_compiled = self._is_compiled and (self._full_validation == full_validation)
        self._full_validation = full_validation
        return self

    def set_is_first_layer(self, is_first_layer: bool) -> Neural_Layer:
        if not isinstance(is_first_layer, bool):
            raise TypeError(f"full_validation must be of type bool")
        self._is_first_layer = is_first_layer

        # recompilation not needed so don't check or change compile flag for efficency
        return self



    @if_full_validation
    def _validate_previous_activations(self, previous_activation: np.ndarray):
        if not isinstance(previous_activation, np.ndarray):
            raise TypeError(f"previous_activation parameter must be of type numpy array")
        if previous_activation.shape != self._expected_Ap_dimensions:
            raise DimensionError(f"previous_activation parameter must be of dimensions {self._expected_Ap_dimensions}")
        if not all(isinstance(element, np.floating) for row in previous_activation for element in row):
            raise TypeError(f"{previous_activation} must contain exclusivly floats of type np.floating")

    @if_full_validation
    def _validate_loss_activation_gradient(self, loss_activation_gradient: np.ndarray):
        if not isinstance(loss_activation_gradient, np.ndarray):
            raise TypeError(f"loss_activation_gradient must be of type numpy array")
        if loss_activation_gradient.shape != self._expected_dcdA_dimensions:
            raise DimensionError(f"loss_activation_gradient should have dimesions   {self._expected_dcdA_dimensions}")
        if not all(isinstance(element, np.floating) for row in loss_activation_gradient for element in row):
            raise TypeError(f"{loss_activation_gradient} must contain exclusivly floats of type np.floating")

    @if_full_validation
    def _validate_learning_rate(self, learning_rate: float):
        if not isinstance(learning_rate, float): 
            raise TypeError(f"Learning rate must be of type float")
        if not (0 < learning_rate):
            raise ValueError(f"Learning rate must be greater than 0")


class Trainable_Layer_Interface(Layer_Interface):
    @abc.abstractmethod
    def update_parameters(self, learning_rate: float) -> None:
        pass

    @abc.abstractmethod
    def print_parameters(self) -> None:
        pass

    @abc.abstractmethod
    def get_parameters(self) -> dict[str, np.ndarray]:
        pass


    @abc.abstractmethod
    def initialise_random_parameters(self) -> None:
        pass

    @abc.abstractmethod
    def reset_trainabe_parameters_on_next_compile(self) -> Trainable_Layer_Interface:
        pass


    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if not isinstance(other, Trainable_Layer_Interface):
            return False
        return True


class Neural_Layer(Trainable_Layer_Interface):
    def __init__(self, number_neurons: int, activation_function: Activation_Function_Interface):
        if not isinstance(number_neurons, int): 
            raise TypeError(f"num_neurons must be an intiger")
        if not (1 <= number_neurons):
            raise ValueError(f"num_neurons must be a natural number >=1")
        if not isinstance(activation_function, Activation_Function_Interface):
            raise TypeError("Activation function must be of type Activation_Function_Interface")


        super().__init__()

        self._keep_parameters_on_compilation = False

        self._activation_function = activation_function

        self._num_prev_neurons = None
        self._num_neurons = number_neurons

        # initialise non dependant parameters        
        self._default_weight_range = (-1.0, 1.0)
        self._default_bias_range = (0.1, 0.1)

        # initialize dimension parameters
        self._wieghts_m_dimensions = None
        self._bias_v_dimensions = None
        self._bias_m_dimensions = None

        self._expected_Ap_dimensions = None
        self._expected_dcdA_dimensions = None

        # initalise parameters for cacheing
        # cache built after foreward propagation
        self._cache_exists_foreward_propagation = False
        self._cache_activations_prev_m = None
        self._cache_activations_m = None
        self._cache_weighted_sums_m = None

        # cache built after backpropagation
        self._cache_exists_back_propagation = False
        self._cache_grad_dldA = None
        self._cache_grad_dldZ = None
        self._cache_grad_dldAp = None
        

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if not isinstance(other, Neural_Layer):
            return False
        
        self._clear_cache()
        other._clear_cache()

        # don't compare parameters derived from other parameters
        if any((
            self._num_neurons != other._num_neurons,
            self._activation_function != other._activation_function,
        )):
            return False

        return True


    def change_activatation_function(self, activation_function: Activation_Function_Interface) -> Neural_Layer:
        if not isinstance(activation_function, Activation_Function_Interface):
            raise TypeError("Activation function must be of type Activation_Function_Interface")

        self._is_compiled = self._is_compiled and (self._activation_function == activation_function)      
        self._keep_parameters_on_compilation = self._is_compiled

        self._clear_cache()
        self._activation_function = activation_function

        return self

  

    def compile(self) -> Neural_Layer:
        if self._is_compiled:
            return self
        
        # complete presense checks
        properties_to_examine = {
            "input_size": self._input_size,
            "batch_size": self._batch_size,
            "activation_function": self._activation_function
        }
        for property_name, property_value in properties_to_examine.items():
            if property_value is None:
                raise ObjectCompiliationError(f"Cannot compile object without {property_name} provided")
       

        # compute dependant parameters
        self._num_prev_neurons = self._input_size

        self._output_size = self._num_neurons

        # # compute dimension parameters
        # wieght_dimensions_changed = (self._wieghts_m_dimensions != (self._num_neurons, self._num_prev_neurons))
        # bias_dimensions_changed = (self._bias_v_dimensions != (self._num_neurons,))

        self._wieghts_m_dimensions = (self._num_neurons, self._num_prev_neurons)
        self._bias_v_dimensions = (self._num_neurons,)
        self._bias_m_dimensions = (self._num_neurons, self._batch_size)

        self._expected_Ap_dimensions = (self._num_prev_neurons, self._batch_size)
        self._expected_dcdA_dimensions = (self._num_neurons, self._batch_size)

        # manage the activation function and ensure it is compiles
        self._activation_function\
            .set_batch_size(self._batch_size)\
            .set_vector_size(self._num_neurons)\
            .set_full_validation(self._full_validation)\
            .compile()
        
        self._clear_cache()

        # declare compiled
        self._is_compiled = True

        # initialise trainable parameters
        # can only execute method if compiled
        # only do this is parameter dimensions were different (or none implied) before

        if not self._keep_parameters_on_compilation:
            self.initialise_random_parameters()
            
        self._keep_parameters_on_compilation = True 
        return self
    
    def reset_trainabe_parameters_on_next_compile(self) -> Neural_Layer:
        self._keep_parameters_on_compilation = False
        self._is_compiled = False
        return self

    @requires_compilation
    def initialise_random_parameters(self, weight_range: tuple[float, float]=None, bias_range: tuple[float, float]=None):

        # initialise parameters with random values (0 for bias)
        if weight_range is None:
            weight_range = self._default_weight_range
        if bias_range is None:
            bias_range = self._default_bias_range

        parameters_to_examine = {
            "weight_range": weight_range,
            "bias_range": bias_range
        }

        for label, value in parameters_to_examine.items():
            if not isinstance(value, tuple):
                raise TypeError(f"{label} must be of type tuple")
            if not all(isinstance(item, float) or isinstance(item, int) for item in value):
                raise TypeError(f"{label} must contain only item of type float or int")
            if len(value) != 2:
                raise TypeError(f"{label} must be of length 2")
        
        chosen_bias_value = random.uniform(*bias_range)
        self._bias_v = np.full(*self._bias_v_dimensions, chosen_bias_value)
        self._weights_m = np.random.uniform(*weight_range, self._wieghts_m_dimensions)

    @requires_compilation
    def back_propagate(self, loss_activation_gradient: np.ndarray) -> np.ndarray:
        self._validate_loss_activation_gradient(loss_activation_gradient)

        # all values of cache for foreward propagation tied together
        if not self._cache_exists_foreward_propagation:
            raise ValueError(f"Cannot call back_propagate without a cached foreward propagatation values")
        
        # cache_relevant = np.equal(loss_activation_gradient, self._cache_grad_dldA).all() if self._cache_grad_dldA is not None else None
        cache_relevant = compare_np_array_with_cache(loss_activation_gradient, self._cache_grad_dldA)

        if not (self._cache_exists_back_propagation and cache_relevant):
            self._cache_exists_back_propagation = True
            self._cache_grad_dldA = loss_activation_gradient

            dAdZ = self._compute_dAdZ()


            # dldZ = dAdZ @ dldA
            # dldAp = dZdAp.T @ dldZ


            # self._cache_grad_dldZ = dAdZ @ self._cache_grad_dldA 
            self._cache_grad_dldZ = np.array([
                dAdZ[:, :, i] @ self._cache_grad_dldA[:, i]
                for i in range(self._batch_size)
            ]).reshape(
                self._num_neurons,
                self._batch_size
            )
            if self._is_first_layer:
                self._cache_grad_dldAp = None
            else:
                dZdAp = self._compute_dZdAp()
                self._cache_grad_dldAp = dZdAp.T @ self._cache_grad_dldZ

        return self._cache_grad_dldAp
    
    def _compute_parameter_gradients(self):
        dZdW = self._compute_dZdW()
        dBmdBv = self._compute_dBmdBv()

        # may need to replace with outer
        # dldW = np.outer(self._cache_grad_dldZ, dZdW.T)
        dldW = self._cache_grad_dldZ @ dZdW.T


        # dZdBm is just the identity matrix so we can ignore that
        dldB = dBmdBv @ self._cache_grad_dldZ.T
        # correct summation rather than mean effect
        dldB /= self._batch_size

        dldB = dldB.reshape(self._bias_v_dimensions)        

        return dldW, dldB
    
    def _check_for_parameter_divergance(self):
        def contains_diverging_element(X):
            return np.isnan(X).any() or np.any(X > 10**12)

        if contains_diverging_element(self._weights_m):
            raise DiverganceError(f"Weights parameter contains 1 or more elements diverging to infintiy (nan overflows or greater than 10^12)")
        if contains_diverging_element(self._bias_v):
            raise DiverganceError(f"Bias parameter contains 1 or more elements diverging to infintiy (nan overflows 10^12)")


    @requires_compilation
    def update_parameters(self, learning_rate: float) -> None:
        self._validate_learning_rate(learning_rate)

        # implies cache for foreward propagation should also exist
        if not self._cache_exists_back_propagation:
            raise ValueError(f"Cannot call update_parameters without a cached backpropagatation values")
        
        dldW, dldB = self._compute_parameter_gradients()      

        self._weights_m -= learning_rate * dldW
        self._bias_v -= learning_rate * dldB

        self._check_for_parameter_divergance()

    @requires_compilation
    def print_parameters(self) -> None:
        print(f"Weights {self._wieghts_m_dimensions}:   {self._weights_m}")
        print(f"Bias {self._bias_v_dimensions}:   {self._bias_v}")

    @requires_compilation
    def get_parameters(self) -> dict[str, np.ndarray]:
        return {
            "W": self._weights_m,
            "B": self._bias_v
        }

    @requires_compilation
    def foreward_propagate(self, previous_activation: np.ndarray) -> np.ndarray:
        self._validate_previous_activations(previous_activation)
        # if cache not up to date then update cache
        # cache_relevant = np.equal(self._cache_activations_prev_m, previous_activation).all() if self._cache_activations_prev_m is not None else None
        cache_relevant = compare_np_array_with_cache(self._cache_activations_prev_m, previous_activation)

        if not (self._cache_exists_foreward_propagation and cache_relevant): 
            self._clear_cache()  
            self._cache_exists_foreward_propagation = True
            self._cache_activations_prev_m = previous_activation

            self._cache_weighted_sums_m = (self._weights_m @ self._cache_activations_prev_m)
            self._cache_weighted_sums_m = self._cache_weighted_sums_m + self._create_bias_matrix()
            
            self._cache_activations_m = self._activation_function.compute_activation(self._cache_weighted_sums_m)

        return self._cache_activations_m

    def _create_bias_matrix(self):
        return np.outer(
            self._bias_v,
            np.ones(self._batch_size)
        )
    
    def _clear_cache(self):
        self._cache_exists_foreward_propagation = False
        self._cache_exists_back_propagation = False

    # derivative of activation with respect to weighted sum
    def _compute_dAdZ(self):
        assert self._cache_exists_foreward_propagation, "Failure no cache, method depends on foreward propagation cache"
        # return self._activation_function.compute_activation_gradient(self._cache_weighted_sums_m)
        return self._activation_function.compute_activation_gradient()

    # derivative of weighted sum with respect to weights matrix
    def _compute_dZdW(self):
        assert self._cache_exists_back_propagation, "Failure no cache, method depends on backpropagation cache"
        return self._cache_activations_prev_m

    # derivative of weighted sum with respect to bias vector
    def _compute_dBmdBv(self):
        assert self._cache_exists_back_propagation, "Failure no cache, method depends on backpropagation cache"
        # ones becuase of the fill method to make the bias matrix
        return np.ones((1, self._batch_size))

    # derivative of weighted sum with respect to previous activation
    def _compute_dZdAp(self):
        assert self._cache_exists_foreward_propagation, "Failure no cache, method depends on foreward propagation cache"
        return self._weights_m


class Dropout_Layer(Layer_Interface):
    def __init__(self, dropout_rate: float) -> None:
        self._validate_dropout_rate(dropout_rate)
        
        super().__init__()
        self._dropout_rate = dropout_rate

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if not isinstance(other, Dropout_Layer):
            return False
        
        # don't compare parameters derived from other parameters
        if any((
            self._dropout_rate != other._dropout_rate,
        )):
            return False

        return True

    # this method doesn't require compilation before or recompilation after
    def set_dropout_rate(self, dropout_rate: float) -> None:
        self._validate_dropout_rate(dropout_rate)
        self._dropout_rate = dropout_rate

    def _validate_dropout_rate(self, dropout_rate: float) -> None:
        if not isinstance(dropout_rate, float):
            raise TypeError(f"Dropout argument must be of type float")
        if not (0 <= dropout_rate < 1):
            raise ValueError(f"Dropout rate must be in interval [0, 1)")
        

    def _create_dropout_matrix(self):
        zeros = int(self._input_size * self._dropout_rate)
        ones = self._input_size - zeros
        random_vector = np.array([0] * zeros + [1] * ones)
        np.random.shuffle(random_vector)

        # can cause to many of the activations to be lost
        # random_vector = np.random.choice(
        #         size = self._input_size,
        #         a=[0, 1],
        #         p=[self._dropout_rate, 1-self._dropout_rate]
        # )

        return np.diag(random_vector)
    
    def compile(self) -> Dropout_Layer:
        # assume not compiled
        if self._is_compiled:
            return self
        
        # complete presence checks
        properties_to_examine = {
            "input_size": self._input_size,
            "batch_size": self._batch_size,
        }
        for property_name, property_value in properties_to_examine.items():
            if property_value is None:
                raise ObjectCompiliationError(f"Cannot compile object without {property_name} provided")

        # compute dependant properties
        self._output_size = self._input_size

        self._expected_Ap_dimensions = (self._input_size, self._batch_size)
        self._expected_dcdA_dimensions = (self._input_size, self._batch_size)

        # declare compiled
        self._is_compiled = True

        return self


    @requires_compilation
    def foreward_propagate(self, previous_activation: np.ndarray) -> np.ndarray:
        self._validate_previous_activations(previous_activation)

        return self._create_dropout_matrix() @ previous_activation

    @requires_compilation
    def back_propagate(self, loss_activation_gradient: np.ndarray) -> np.ndarray:
        self._validate_loss_activation_gradient(loss_activation_gradient)

        # drop out rate derivative is identity
        return loss_activation_gradient


class Neural_Layer_Momentum(Neural_Layer):
    def __init__(self, number_neurons: int, activation_function: Activation_Function_Interface):
        super().__init__(number_neurons, activation_function)#
        self._momentum_coefficient = None
        self._bias_velocity_v: np.ndarray = None
        self._weights_velocity_m: np.ndarray = None
    

    def set_momentum_coefficient(self, momentum_coefficient: float) -> Neural_Layer_Momentum:
        # this funciton sets the momentum hyper parameter
        if not isinstance(momentum_coefficient, float):
            raise TypeError("momentum coefficient must be of type float")
        if not (0 <= momentum_coefficient < 1):
            raise ValueError("momentum coefficent must be in the interval [0, 1)")
        
        def absolute(x): return x if x>=0 else -x
        if self._momentum_coefficient is None:
            momentum_coefficient_changed = True
        else:
            momentum_coefficient_changed = absolute(self._momentum_coefficient - momentum_coefficient) > 10**-12
        
        if momentum_coefficient_changed:
            self._momentum_coefficient = momentum_coefficient
        self._is_compiled = self._is_compiled and not momentum_coefficient_changed      

        return self


    def compile(self) -> Neural_Layer:
        if self._is_compiled:
            return self
        
        super().compile()

        if self._momentum_coefficient is None:
            raise ObjectCompiliationError("momentum_coefficient not provided")
        
        self._bias_velocity_v = np.zeros(shape=self._bias_v_dimensions)
        self._weights_velocity_m = np.zeros(shape=self._wieghts_m_dimensions)

        self._is_compiled = True
        return self

    
    def update_parameters(self, learning_rate: float) -> None:
        self._validate_learning_rate(learning_rate)

        # implies cache for foreward propagation should also exist
        if not self._cache_exists_back_propagation:
            raise ValueError(f"Cannot call update_parameters without a cached backpropagatation values")
        
        dldW, dldB = self._compute_parameter_gradients()      

        self._weights_velocity_m: np.ndarray = (
            self._momentum_coefficient * self._weights_velocity_m 
            + (1-self._momentum_coefficient) * dldW
        )
        self._bias_velocity_v: np.ndarray = (
            self._momentum_coefficient * self._bias_velocity_v 
            + (1-self._momentum_coefficient) * dldB
        )

        self._weights_m -= learning_rate * self._weights_velocity_m
        self._bias_v -= learning_rate * self._bias_velocity_v

        self._check_for_parameter_divergance()
