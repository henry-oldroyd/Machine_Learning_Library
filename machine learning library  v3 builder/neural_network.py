from __future__ import annotations
import numpy as np
from copy import deepcopy


from decorators import requires_compilation, if_full_validation
from layers import Layer_Interface, Trainable_Layer_Interface, Neural_Layer_Momentum
from loss_functions import Loss_Function_Interface
from ml_exceptions import ObjectCompiliationError, DimensionError, MethodSequencingError, ParameterChangeZero
from utility_functions import compare_np_array_with_cache

class Neural_Network:
    def __init__(
        self, 
        num_input_neurons: int = None,
        layers: list[Layer_Interface]=None,
        loss_function: Loss_Function_Interface=None,
        batch_size: int = None,
        learning_rate: float = None,
    ) -> None:

        self._is_compiled = False
        self._keep_parameters_on_compilation = False
        self._full_validation = True


        self._layers: tuple[Layer_Interface] = None

        self._batch_size = None
        self._learning_rate = None
        self._loss_function = None
        self._num_input_neurons = None
        self._num_output_neurons = None

        self._cache_exists_foreward_propagation = False
        self._cache_exists_loss = False
        self._cache_exists_back_propagation = False

        self._cache_FP_X = None
        self._cache_FP_Y = None
        self._cache_FP_P = None
        self._cache_FP_loss = None
        

        if layers is not None:
            self.set_layers(layers)
        if loss_function is not None:
            self.set_loss_function(loss_function)
        if num_input_neurons is not None:
            self.set_num_input_neurons(num_input_neurons)
        if batch_size is not None:
            self.set_batch_size(batch_size)
        if learning_rate is not None:
            self.set_learning_rate(learning_rate)
        

    def _clear_cache(self):
        self._cache_exists_foreward_propagation = False
        self._cache_exists_loss = False
        self._cache_exists_back_propagation = False


    def set_batch_size(self, batch_size: int) -> Neural_Network:
        if not isinstance(batch_size, int): 
            raise TypeError(f"batch_size must be an intiger")
        if not (1 <= batch_size): 
            raise ValueError(f"batch_size must be a natural number >=1")
        
        self._is_compiled = self._is_compiled and (self._batch_size == batch_size)
        self._batch_size = batch_size
        return self
    
    @requires_compilation
    def get_batch_size(self) -> int:
        return self._batch_size

    @requires_compilation
    def get_input_size(self) -> int:
        return self._num_input_neurons


    @requires_compilation
    def get_output_size(self) -> int:
        return self._num_output_neurons

    def set_num_input_neurons(self, num_input_neurons: int) -> Neural_Network:
        if not isinstance(num_input_neurons, int): 
            raise TypeError(f"num_input_neurons must be an integer")
        if not (1 <= num_input_neurons): 
            raise ValueError(f"num_input_neurons must be a natural number >=1")

        self._is_compiled = self._is_compiled and (self._num_input_neurons == num_input_neurons)
        self._keep_parameters_on_compilation = self._is_compiled
        self._num_input_neurons = num_input_neurons
        return self

    def set_learning_rate(self, learning_rate: float) -> Neural_Network:
        if not isinstance(learning_rate, float):
            raise TypeError(f"Learning rate must be of type float")
        # if not (0 < learning_rate < 1):
        if not (-10**-12 < learning_rate):
            raise ValueError(f"Learning rate must be in the interval (0, infinity)")

        self._is_compiled = self._is_compiled and ((self._learning_rate - learning_rate) <= 10**-12)
        self._learning_rate = learning_rate
        return self
    
        
    @requires_compilation
    def get_learning_rate(self) -> float:
        return self._learning_rate
    

    def set_loss_function(self, loss_function: Loss_Function_Interface) -> Neural_Network:
        if not isinstance(loss_function, Loss_Function_Interface):
            raise TypeError("Loss function must be of type Loss_Function_Interface")
        
        self._is_compiled = self._is_compiled and self._loss_function == loss_function
        self._keep_parameters_on_compilation = self._is_compiled

        self._loss_function = loss_function
        self._clear_cache()


        return self

    def set_layers(self, new_layers: tuple[Layer_Interface]) -> Neural_Network:
        # defensive design allow list
        if isinstance(new_layers, list):
            new_layers = tuple(new_layers)

        if not isinstance(new_layers, tuple):
            raise TypeError("new_layers parameter must be of type tuple (or list)")
        if len(new_layers) == 0:
            raise ValueError("new_layers parameter cannot be an empty list")
        if any(
            not isinstance(layer, Layer_Interface)
            for layer in new_layers
        ):
            raise TypeError("each item in new_layers parameter must be of type Layer_Interface")

        self._is_compiled = self._is_compiled and (self._layers == new_layers)
        self._keep_parameters_on_compilation = self._is_compiled

        self._layers = new_layers
        self._clear_cache()

        return self


    def set_full_validation(self, full_validation: bool) -> Neural_Network:
        if not isinstance(full_validation, bool): 
            raise TypeError(f"full_validation must be of type bool")
        
        self._is_compiled = self._is_compiled and (self._full_validation == full_validation)        
        self._full_validation = full_validation
        return self

    

    @requires_compilation
    def get_layers(self) -> tuple[Layer_Interface]:
        return self._layers
    
    @requires_compilation
    def get_learning_rate(self) -> float:
        return self._learning_rate
    
    
    # def reset_parameters_on_next_compile(self) -> Neural_Network:
    #     self._keep_parameters_on_compilation = False
    #     self._is_compiled = False
    #     return self
    
    def reset_parameters(self) -> Neural_Network:
        self._keep_parameters_on_compilation = False
        self._is_compiled = False
        self.compile()
        return self

    def is_compiled(self) -> bool:
        return self._is_compiled
    


    def compile(self) -> Neural_Network:
        # assume not compiled
        if self._is_compiled:
            return self
        

        # complete presence check
        properties_to_examine = {
            "batch_size": self._batch_size,
            "num_input_neurons": self._num_input_neurons,
            "learning_rate": self._learning_rate,
            "loss_function": self._loss_function,
            "layers": self._layers,
        }

        for property_name, property_value in properties_to_examine.items():
            if property_value is None:
                raise ObjectCompiliationError(f"Cannot compile object without {property_name} provided")


        # configure composed objects
        self._layers[0].set_is_first_layer(True)

        layer_input_size = self._num_input_neurons
        for layer in self._layers:
            layer\
                .set_input_size(layer_input_size)\
                .set_batch_size(self._batch_size)\
                .set_full_validation(self._full_validation)\


            if isinstance(layer, Trainable_Layer_Interface) and (not self._keep_parameters_on_compilation):
                layer.reset_trainabe_parameters_on_next_compile()\
                
            layer.compile()
            layer_input_size = layer.get_output_size()


        # compute dependant properties
        # depends on layers being compiled and loss compilation depends on this 
        self._num_output_neurons = self._layers[-1].get_output_size()
        self._expected_dimensions_input_batch = (self._num_input_neurons, self._batch_size)
        self._expected_dimensions_expected_output = (self._num_output_neurons, self._batch_size)


        self._loss_function\
            .set_batch_size(self._batch_size)\
            .set_full_validation(self._full_validation)\
            .set_vector_size(self._num_output_neurons)\
            .compile()


        self._clear_cache()

        # declare compiled
        self._is_compiled = True
        self._keep_parameters_on_compilation = True

        return self

    @requires_compilation
    def train_model_on_minibatch(self, X_train_data: np.ndarray, Y_train_data: np.ndarray) -> float:
        # complete foreward propagation, back propagation, update parameters and return loss
        self._validate_input_and_expected_output_bathes(
            input_batch=X_train_data,
            expected_output_batch=Y_train_data,
        )

        _, loss = self._foreward_propagate(
            input_batch=X_train_data,
            expected_output_batch=Y_train_data
        )

        self._back_propogate()
        self._update_parameters()
                    
        return loss


    @requires_compilation
    def evaluate_model_on_test_data(self, X_test_data: np.ndarray, Y_test_data: np.ndarray) -> float:
        # change internal batch size to process all test data at once (shouldn't loose trained parameters)
        # compute loss across the whole batch

        self._validate_input_and_expected_output_bathes(
            input_batch=X_test_data,
            expected_output_batch=Y_test_data,
            expected_output_batch_required=True,
            check_dimensions=False
        )

        if X_test_data.shape[1] != Y_test_data.shape[1]:
            raise DimensionError("Different amounnt of X and Y test data provided")
        
        if (X_test_data.shape[0] != self._num_input_neurons) or (len(X_test_data.shape) != 2):
            raise DimensionError("X test data not of dimensions (input_size, test_dataset_size)")
        if (Y_test_data.shape[0] != self._num_output_neurons) or (len(Y_test_data.shape) != 2):
            raise DimensionError("Y test data not of dimensions (output_size, test_dataset_size)")

        test_dataset_size = X_test_data.shape[1]
        if test_dataset_size < 1:
            raise ValueError("Test data set must not be empty")

        previous_batch_size = self._batch_size

        self.set_batch_size(test_dataset_size)
        self.compile()

        _, loss  = self._foreward_propagate(X_test_data, Y_test_data)

        self.set_batch_size(previous_batch_size)
        self.compile()

        return loss

    
    @requires_compilation
    def make_predicitons(self, X_data):
        self._validate_input_and_expected_output_bathes(
            input_batch=X_data,
            expected_output_batch=None,
            expected_output_batch_required=False,
            check_dimensions=False,
        )

        if (X_data.shape[0] != self._num_input_neurons) or (len(X_data.shape) != 2):
            raise DimensionError("Prediction X data not of dimension (num_input_neurons, prediction_data_set_size)")
        
        prediction_data_set_size = X_data.shape[1]
        if prediction_data_set_size < 1:
            raise ValueError("Pediction X_data must not be empty")


        if prediction_data_set_size != self._batch_size:
            previous_batch_size = self._batch_size

            self.set_batch_size(prediction_data_set_size)
            self.compile()

            predictions, _ = self._foreward_propagate(X_data)

            self.set_batch_size(previous_batch_size)
            self.compile()
        else:
            predictions, _ = self._foreward_propagate(X_data)


        return predictions


    @if_full_validation
    def _validate_input_and_expected_output_bathes(
        self, 
        input_batch: np.ndarray, 
        expected_output_batch: np.ndarray, 
        expected_output_batch_required: bool = True, 
        check_dimensions: bool = True
    ):
        parameters_to_examine = {
            "input_batch": input_batch,
        }
        # 2nd parameter is optional
        if expected_output_batch is not None:
            parameters_to_examine["expected_output_batch"] = expected_output_batch

        if not isinstance(input_batch, np.ndarray):
            raise TypeError("input_batch must be of type np.ndarray")
        if check_dimensions and (input_batch.shape != self._expected_dimensions_input_batch):
            raise DimensionError("input_batch must be of dimensions (input_size, batch_size)")
        if any(
            not isinstance(element, np.floating)
            for batch_item in input_batch
            for element in batch_item
        ): 
            raise TypeError("each item in each vector in input_batch must be of type np.floating")
    
        if (expected_output_batch is not None) or expected_output_batch_required:
            if not isinstance(expected_output_batch, np.ndarray):
                raise TypeError("expected_output_batch must be of type np.ndarray")
            if check_dimensions and (expected_output_batch.shape != self._expected_dimensions_expected_output):
                raise DimensionError("expected_output_batch must be of dimensions (output_size, batch_size)")
            if any(
                not isinstance(element, np.floating)
                for batch_item in expected_output_batch
                for element in batch_item
            ): 
                raise TypeError("each item in each vector in expected_output_batch must be of type np.floating")

    @requires_compilation
    def get_parameters(self):
        parameters = []
        for layer in self._layers:
            if isinstance(layer, Trainable_Layer_Interface):
                layer: Trainable_Layer_Interface
                parameters.append(
                    layer.get_parameters()
                )
        return parameters
    
    @requires_compilation
    def print_parameters(self):
        for layer_i, layer in enumerate(self._layers):
            if isinstance(layer, Trainable_Layer_Interface):
                layer: Trainable_Layer_Interface
                print(f"Layer {layer_i}:")
                layer.print_parameters()



    def _foreward_propagate(self, input_batch: np.ndarray, expected_output_batch: np.ndarray = None) -> tuple[np.ndarray, float]:
        if not compare_np_array_with_cache(input_batch, self._cache_FP_X):
            self._clear_cache()
        elif not compare_np_array_with_cache(expected_output_batch, self._cache_FP_Y):
            self._cache_exists_loss = False
            self._cache_exists_back_propagation = False

        if not self._cache_exists_foreward_propagation:

            neuron_activations_batch_matrix = input_batch
            for layer in self._layers:
                neuron_activations_batch_matrix = layer.foreward_propagate(neuron_activations_batch_matrix)



            self._cache_FP_X = input_batch
            self._cache_FP_P = neuron_activations_batch_matrix

            self._cache_exists_foreward_propagation = True

        if expected_output_batch is None:
            self._cache_FP_loss = None
        elif not self._cache_exists_loss:
            self._cache_FP_loss = self._loss_function.compute_loss(
                Y=expected_output_batch, 
                P=self._cache_FP_P
            )
            self._cache_exists_loss = True
        
        return self._cache_FP_P, self._cache_FP_loss


    def _back_propogate(self) -> None: 
        if not all((
            self._cache_exists_foreward_propagation,
            self._cache_exists_loss
        )):
            raise MethodSequencingError("Cannot call backpropogate before foreward propagation and loss calculation")

        if not self._cache_exists_back_propagation:
            dldA = self._loss_function.compute_loss_gradient()
            for layer in reversed(self._layers):
                dldA = layer.back_propagate(dldA)
            self._cache_exists_back_propagation = True


    def _update_parameters(self) -> None:
        if not self._cache_exists_back_propagation:
            raise MethodSequencingError("Cannot call update parameters before backpropagation")

        old_parameters = deepcopy(
            [
                layer.get_parameters() 
                for layer in self._layers 
                if isinstance(layer, Trainable_Layer_Interface)
            ]
        )


        for layer in self._layers:
            if isinstance(layer, Trainable_Layer_Interface):
                layer.update_parameters(self._learning_rate)


        new_paremters = [
            layer.get_parameters() 
            for layer in self._layers 
            if isinstance(layer, Trainable_Layer_Interface)
        ]


        for i, (p_old, p_new) in enumerate(zip(old_parameters, new_paremters)):
            # assert not np.equal(p_old["B"], p_new["B"]).all(), f"Bias of layer {i} did not change"
            if np.equal(p_old["B"], p_new["B"]).all():
                raise ParameterChangeZero(f"Bias of layer {i} did not change")
            # assert not np.equal(p_old["W"], p_new["W"]).all(), f"Weights of layer {i} did not change"
            if np.equal(p_old["W"], p_new["W"]).all():
                raise ParameterChangeZero(f"Weights of layer {i} did not change")




        self._clear_cache()



class Neural_Network_Momentum(Neural_Network):
    def __init__(
            self,
            num_input_neurons: int = None, 
            layers: list[Layer_Interface] = None, 
            loss_function: Loss_Function_Interface = None, 
            batch_size: int = None, 
            learning_rate: float = None,
            momentum_coefficient: float = None,
        ) -> None:

        super().__init__(num_input_neurons, layers, loss_function, batch_size, learning_rate)

        if not isinstance(momentum_coefficient, float):
            raise TypeError("momentum coefficient must be of type float")
        if not (0 <= momentum_coefficient < 1):
            raise ValueError("momentum coefficent must be in the interval [0, 1)")

        self._momentum_coefficient = momentum_coefficient


    def compile(self) -> Neural_Network:
        if self._is_compiled:
            return self
        
        for layer in self._layers:
            if isinstance(layer, Neural_Layer_Momentum):
                layer: Neural_Layer_Momentum
                layer.set_momentum_coefficient(self._momentum_coefficient)

        super().compile()


        self._is_compiled = True
        return self