# import libraries
import abc
import numpy as np
from random import uniform

from activation_functions_no_builder import Activation_Function_Interface


class DimensionError(Exception):
    pass

class DiverganceError(Exception):
    pass

class Transformation_Layer_Inteface(abc.ABC):
    def __init__(self, input_size: int, batch_size:int) -> None:
        for name, value in ((f"batch_size", batch_size), (f"input_size", input_size)):
            if not isinstance(value, int):
                raise TypeError(f"{name} must be of type int")
            if not (1 <= value):
                raise ValueError(f"{name} must be a natural number 1 or greater")

        self._input_size = input_size
        self._batch_size = batch_size

    @abc.abstractmethod
    def foreward_propagate(self, previous_activation: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def back_propagate(self, loss_activation_gradient: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def update_parameters(self, learning_rate: float) -> None:
        pass


class Fully_Connected_Neuron_Layer(Transformation_Layer_Inteface):
    def __init__(self, input_size: int, output_size: int, batch_size:int, activation_function: Activation_Function_Interface, *args, **kwargs) -> None:
        if not isinstance(output_size, int):
            raise TypeError(f"num_neurons must be an intiger")
        if not (1 <= output_size):
            raise ValueError(f"num_neurons must be a natural number >=1")
        
        if not isinstance(activation_function, Activation_Function_Interface):
            raise TypeError(f"activation function must be of type Activation_Function_Interface")
        
        self._output_size = output_size
        self._activation_function = activation_function
        super().__init__(
            input_size=input_size,
            batch_size=batch_size
        )

        # define other properties
        self._num_neurons = self._output_size
        self._num_prev_neurons = self._input_size

        self._wieghts_m_dimensions = (self._num_neurons, self._num_prev_neurons)
        self._bias_v_dimensions = (self._num_neurons,)
        self._bias_m_dimensions = (self._num_neurons, self._batch_size)

        self._expected_Ap_dimensions = (self._num_prev_neurons, self._batch_size)
        self._expected_dcdA_dimensions = (self._num_neurons, self._batch_size)

        self._default_weight_range = (-1.0, 1.0)
        self._default_bias_range = (0.0, 0.0)
        self.initialise_random_parameters()

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

    def initialise_random_parameters(self, weight_range: tuple[float, float]=None, bias_range: tuple[float, float]=None):
        # initialise parameters with random values (0 for bias)
        if weight_range is None:
            weight_range = self._default_weight_range
        if bias_range is None:
            bias_range = self._default_bias_range

        for label, value in ((f"weight_range", weight_range), (f"bias_range", bias_range)):
            if not isinstance(value, tuple):
                raise TypeError(f"{label} must be of type tuple")
            if not all(isinstance(item, float) or isinstance(item, int) for item in value):
                raise TypeError(f"{label} must contain only item of type float or int")
            if len(value) != 2:
                raise TypeError(f"{label} must be of length 2")
        
        chosen_bias_value = uniform(*bias_range)
        self._bias_v = np.full(*self._bias_v_dimensions, chosen_bias_value)
        self._weights_m = np.random.uniform(*weight_range, self._wieghts_m_dimensions)


    def _create_bias_matrix(self):
        return np.outer(
            self._bias_v,
            np.ones(self._batch_size)
        )
    
    def _wipe_cache(self):
        self._cache_exists_foreward_propagation = False
        self._cache_exists_back_propagation = False

    def foreward_propagate(self, previous_activation: np.ndarray) -> np.ndarray:
        if not isinstance(previous_activation, np.ndarray):
            raise TypeError(f"previous_activation parameter must be of type numpy array")
        if previous_activation.shape != self._expected_Ap_dimensions:
            raise DimensionError(f"previous_activation parameter must be of dimensions {self._expected_Ap_dimensions}")

        # if cache not up to date then update cache
        cache_relevant = np.equal(self._cache_activations_prev_m, previous_activation).all() if self._cache_activations_prev_m is not None else None
        if not (self._cache_exists_foreward_propagation and cache_relevant): 
            self._wipe_cache()  
            self._cache_exists_foreward_propagation = True
            self._cache_activations_prev_m = previous_activation
            self._cache_weighted_sums_m = (self._weights_m @ self._cache_activations_prev_m) + self._create_bias_matrix()
            self._cache_activations_m = self._activation_function.compute_activation(self._cache_weighted_sums_m)

        return self._cache_activations_m


    # derivative of activation with respect to weighted sum
    def _compute_dAdZ(self):
        assert self._cache_exists_foreward_propagation, "Failure no cache, method depends on foreward propagation cache"
        return self._activation_function.compute_activation_gradient(self._cache_weighted_sums_m)

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
    

    def back_propagate(self, loss_activation_gradient: np.ndarray = None) -> np.ndarray:
        # all values of cache for foreward propagation tied together
        if not isinstance(loss_activation_gradient, np.ndarray):
            raise TypeError(f"loss_activation_gradient must be of type numpy array")
        if loss_activation_gradient.shape != self._expected_dcdA_dimensions:
            raise DimensionError(f"loss_activation_gradient should have dimesions   {self._expected_dcdA_dimensions}")

        if not self._cache_exists_foreward_propagation:
            raise ValueError(f"Cannot call back_propagate without a cached foreward propagatation values")
        
        cache_relevant = np.equal(loss_activation_gradient, self._cache_grad_dldA).all() if self._cache_grad_dldA is not None else None
        if not (self._cache_exists_back_propagation and cache_relevant):
            self._cache_exists_back_propagation = True
            self._cache_grad_dldA = loss_activation_gradient

            dAdZ = self._compute_dAdZ()
            dZdAp = self._compute_dZdAp()

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
            



            self._cache_grad_dldAp = dZdAp.T @ self._cache_grad_dldZ

        return self._cache_grad_dldAp
        
    def update_parameters(self, learning_rate: float) -> None:
        # implies cache for foreward propagation should also exist
        if not self._cache_exists_back_propagation:
            raise ValueError(f"Cannot call update_parameters without a cached backpropagatation values")
        
        if not isinstance(learning_rate, float): 
            raise TypeError(f"Learning rate must be of type float")
        if not (0 < learning_rate):
            raise ValueError(f"Learning rate must be greater than 0")

        dZdW = self._compute_dZdW()
        dBmdBv = self._compute_dBmdBv()

        # may need to replace with outer
        # dldW = np.outer(self._cache_grad_dldZ, dZdW.T)
        dldW = self._cache_grad_dldZ @ dZdW.T
        # print(f"dldW = self._cache_grad_dldZ @ dZdW.T")
        # print(f"{dldW.shape} = {self._cache_grad_dldZ.shape} @ {dZdW.T.shape}")


        # dZdBm is just the identity matrix so we can ignore that
        dldB = dBmdBv @ self._cache_grad_dldZ.T
        # correct summation rather than mean effect
        dldB /= self._batch_size

        dldB = dldB.reshape(self._bias_v_dimensions)        

        self._weights_m -= learning_rate * dldW
        self._bias_v -= learning_rate * dldB

        def contains_diverging_element(X):
            return np.isnan(X).any() or np.any(X > 1_000_000)

        if contains_diverging_element(self._weights_m):
            raise DiverganceError(f"Weights parameter contains 1 or more elements diverging to infintiy (nan overflows or greater than 10^6)")
        if contains_diverging_element(self._bias_v):
            raise DiverganceError(f"Bias parameter contains 1 or more elements diverging to infintiy (nan overflows 10^6)")

    def print_parameters(self):
        print(f"Weights {self._wieghts_m_dimensions}:   {self._weights_m}")
        print(f"Bias {self._bias_v_dimensions}:   {self._bias_v}")


    def get_parameters(self):
        return {
            "W": self._weights_m,
            "B": self._bias_v
        }

class Dropout_Layer(Transformation_Layer_Inteface):
    def __init__(self, input_size: float, batch_size: float, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(
            input_size=input_size, 
            batch_size=batch_size
        )

        if not isinstance(dropout_rate, float):
            raise TypeError(f"Dropout argument must be of type float")
        if not (0 <= dropout_rate < 1):
            raise ValueError(f"Dropout rate must be in interval [0, 1)")
        self._dropout_rate = dropout_rate

    def _create_dropout_matrix(self):
        zeros = int(self._input_size * self._dropout_rate)
        ones = self._input_size - zeros
        random_vector = np.array([0] * zeros + [1] * ones)
        np.random.shuffle(random_vector)

        # can cause to many of the activation to be lost
        # random_vector = np.random.choice(
        #         size = self._input_size,
        #         a=[0, 1],
        #         p=[self._dropout_rate, 1-self._dropout_rate]
        # )

        return np.diag(random_vector)

    def foreward_propagate(self, previous_activation: np.ndarray) -> np.ndarray:
        return self._create_dropout_matrix() @ previous_activation

    def back_propagate(self, loss_activation_gradient: np.ndarray) -> np.ndarray:
        return loss_activation_gradient
    
    def update_parameters(self, learning_rate: float) -> None:
        return None