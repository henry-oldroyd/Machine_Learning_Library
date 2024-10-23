from __future__ import annotations
import abc
import numpy as np
import math

from decorators import if_full_validation, requires_compilation 
from ml_exceptions import ObjectCompiliationError

class Data_Encoding_Interface(abc.ABC):

    @abc.abstractmethod
    def encode_dataset(self, dataset_raw: list) -> np.ndarray:
        pass

    @abc.abstractmethod
    def decode_dataset(self, dataset_vector: np.ndarray) -> list:
        pass


    def __init__(self) -> None:
        self._full_validation = True


    @if_full_validation
    def _validate_raw_data(self, dataset_raw: list) -> None:
        # defensive design allow tuple
        if isinstance(dataset_raw, tuple):
            dataset_raw = list(dataset_raw)

        if not isinstance(dataset_raw, list):
            raise TypeError(f"dataset_raw must be of type tuple or list")

    @if_full_validation
    def _validate_vector_data(self, dataset_vector: np.ndarray) -> None:
        if not isinstance(dataset_vector, np.ndarray):
            raise TypeError(f"dataset_raw must be of type np.ndarray")

    def set_full_validation(self, full_validation: bool) -> Data_Encoding_Interface:
        if not isinstance(full_validation, bool): 
            raise TypeError(f"Full validation must be of type bool")
        
        self._full_validation = full_validation
        # recompilation not needed so don't check or change compile flag
        return self
    


class Scalar_Encoding(Data_Encoding_Interface):
    def __init__(self, scalar_range: list[float, float], num_neurons: int):
        if not isinstance(num_neurons, int):
            raise TypeError("num_neurons must be of type int")
        if num_neurons < 1:
            raise ValueError("num_neurons must be a positive intiger")
        

        # defensive design allow tuple
        if isinstance(scalar_range, tuple):
            scalar_range = list(scalar_range)
        if not isinstance(scalar_range, list):
            raise TypeError("Scalar range parameter must be of type list")
        if len(scalar_range) != 2:
            raise ValueError("Scalar range parameter must be of length 2") 
        
        # defensive design allow ints
        scalar_range = [float(e) if isinstance(e, int) else e for e in scalar_range]

        if any(not isinstance(e, float) for e in scalar_range):
            raise TypeError("Elements in scalar range must both be of type float")


        super().__init__()

        self._scalar_range = scalar_range
        self._num_neurons = num_neurons

    @if_full_validation
    def _validate_raw_data(self, dataset_raw: list) -> None:
        super()._validate_raw_data(dataset_raw)

        # defensive design allow ints
        if any(
            type(element) not in (int, float) 
            for element in dataset_raw
        ):
            raise TypeError(f"Each element in dataset_raw parameter must be of type float")
        
        a, b = self._scalar_range
        if any(
            not a <= element <= b
            for element in dataset_raw
        ):
            raise ValueError(f"Each element in dataset_raw parameter must be in scalar range provided [{self._scalar_range[0]}, {self._scalar_range[1]}]")
    
    @if_full_validation
    def _validate_vector_data(self, dataset_vector: np.ndarray) -> None:
        super()._validate_vector_data(dataset_vector)

        if any(
            not isinstance(element, np.floating) 
            for vector in dataset_vector 
            for element in vector
        ):
            raise TypeError("Each element in dataset_vector parameter must be of type np.ndarray")
        
        if any(
            not (self._scalar_range[0] < element < self._scalar_range[1])
            for vector in dataset_vector 
            for element in vector 
        ):
            raise ValueError("Each element in dataset_vector must be in interval [0.0, 1.0]")
        
    def encode_dataset(self, dataset_raw: list) -> np.ndarray:
        self._validate_raw_data(dataset_raw)

        def cap_0_to_1(x):
            return max(0.0, min(1.0, x))

        dataset_vector = []
        a, b = self._scalar_range
        for scalar_item in dataset_raw:
            coded_scalar = (scalar_item - a)  * (self._num_neurons / (b-a))
            encoded_vector = [
                cap_0_to_1(coded_scalar - i)
                for i in range(self._num_neurons)
            ]
            dataset_vector.append(encoded_vector)
        
        return np.array(dataset_vector)

    def decode_dataset(self, dataset_vector: np.ndarray) -> list:
        self._validate_vector_data(dataset_vector)

        dataset_raw = []
        a, b = self._scalar_range

        for data_item in dataset_vector:
            coded_scalar = sum(data_item)
            scalar = coded_scalar * ((b-a) / self._num_neurons) + a
            dataset_raw.append(scalar)

        return dataset_raw
    

class One_Hot_Encoding(Data_Encoding_Interface):
    def __init__(self, num_catagories: int) -> None:
        if not isinstance(num_catagories, int):
            raise TypeError("num_catagories must be of type int")
        if num_catagories < 1:
            raise ValueError("num_catagories must be a positive intiger")
        
        super().__init__()
        self._num_neurons = num_catagories
        self._num_catagories = num_catagories

    @if_full_validation
    def _validate_raw_data(self, dataset_raw: list) -> None:
        super()._validate_raw_data(dataset_raw)

        # floats not allowed
        if any(
            not isinstance(element, int)
            for element in dataset_raw
        ):
            raise TypeError(f"Each element in dataset_raw parameter must be of type int")
        
        if any(
            not 0 <= element < self._num_catagories
            for element in dataset_raw
        ):
            raise ValueError(f"Each element in dataset_raw parameter must be an initger in the interval [0, {self._num_catagories-1}]")

    
    @if_full_validation
    def _validate_vector_data(self, dataset_vector: np.ndarray) -> None:
        super()._validate_vector_data(dataset_vector)

        if any(
            not isinstance(element, np.floating)
            for vector in dataset_vector 
            for element in vector
        ):
            raise TypeError("Each element in dataset_vector parameter must be of type np.ndarray")
        
        if any(
            round(element, 8) not in (0.0, 1.0)
            for vector in dataset_vector 
            for element in vector 
        ):
            raise ValueError("Each element in dataset_vector must be either 1.0 or 0.0")
        

    def encode_dataset(self, dataset_raw: list) -> np.ndarray:
        self._validate_raw_data(dataset_raw)

        dataset_vector = []
        zeros_vector = [0.0 for _ in range(self._num_neurons)]
        for int_item in dataset_raw:
            encoded_vector = zeros_vector.copy()
            encoded_vector[int_item] = 1.0
            
            dataset_vector.append(encoded_vector)
        
        return np.array(dataset_vector)

    def decode_dataset(self, dataset_vector: np.ndarray) -> list:
        self._validate_vector_data(dataset_vector)

        dataset_raw = []

        for data_item in dataset_vector:
            catagory_int = [i for i in range(self._num_neurons) if round(data_item[i], 8) == 1.0][0]
            dataset_raw.append(catagory_int)
            
        return dataset_raw
    


class Coordinate_Features_2d_Encoder(Data_Encoding_Interface):
    def __init__(self, num_forier_features: int, num_taylor_features: int) -> None:
        # 1 is an overlapping term between these feature sets and so it will be added in anyway in addition
        parameters_to_examine = {
            "num_forier_features": num_forier_features,
            "num_taylor_features": num_taylor_features,
        }
        for parameter_name, parameter_value in parameters_to_examine.items():
            if not isinstance(parameter_value, int):
                raise TypeError(f"{parameter_name} must be of type int")
            if parameter_value < 0: 
                raise ValueError(f"{parameter_name} must be an int >= 0")

        super().__init__()
        self._num_forier_features = num_forier_features
        self._num_taylor_features = num_taylor_features


    def _generate_forier_features(self, x, y):
        multiplier = 1
        terms_produced = 0
        
        functions_of_multiplier = (
            lambda m: math.sin(m * x),
            lambda m: math.sin(m * y),
            lambda m: math.cos(m * x),
            lambda m: math.cos(m * y)
        )

        while True:
            for function_of_multiplier in functions_of_multiplier:
                if terms_produced == self._num_forier_features:
                    return None
                yield function_of_multiplier(multiplier)
                terms_produced += 1

            multiplier += 1


    def _generate_taylor_features(self, x, y):
        exponent_total = 1
        terms_produced = 0

        while True:
            for x_exponent in range(exponent_total, -1, -1):
                if terms_produced == self._num_taylor_features:
                    return None

                y_exponent = exponent_total - x_exponent
                yield x**x_exponent * y**y_exponent
                terms_produced += 1

            exponent_total += 1

    @if_full_validation
    def _validate_raw_data(self, dataset_raw: list) -> None:
        super()._validate_raw_data(dataset_raw)

        # list required but tuples are allowed 
        # if any(
        #     type(element) not in (tuple, list)
        #     for element in dataset_raw
        # ):
        #     raise TypeError(f"Each coordinate list in dataset_raw parameter must be of type list or tuple")


        for element in dataset_raw:
            if type(element) not in (tuple, list):
                raise TypeError(f"Each coordinate list in dataset_raw parameter must be of type list or tuple")



        if any(
            len(data_item) != 2
            for data_item in dataset_raw
        ):
            raise ValueError(f"Each coordinate list in dataset_raw must be of length 2")
        
        # floats required but ints are allowed 
        if any(
            type(element) not in (int, float)
            for coordinate_pair in dataset_raw
            for element in coordinate_pair
        ):
            raise TypeError(f"Each coordinate in dataset_raw parameter must be of type int or float")

    def encode_dataset(self, dataset_raw: list) -> np.ndarray:
        self._validate_raw_data(dataset_raw)

        dataset_vector = []
        for x, y in dataset_raw:
            input_vector = list(self._generate_taylor_features(x=x, y=y)) + list(self._generate_forier_features(x=x, y=y))
            input_vector = [float(element) for element in input_vector]
            dataset_vector.append(input_vector)

        return np.array(dataset_vector)

    def decode_dataset(self, dataset_vector: np.ndarray) -> list:
        raise NotImplemented("Coordinate_Features_2d_Encoder provides only encoding logic and the decoder method is not supported") 
    



# add validation when you have time

class Identity_Encoding(Data_Encoding_Interface):
    def encode_dataset(self, dataset_raw: list) -> np.ndarray:
        # self._validate_raw_data(dataset_raw)
        if isinstance(dataset_raw, np.ndarray):
            return dataset_raw
        else:
            return np.array(dataset_raw)

    def decode_dataset(self, dataset_vector: np.ndarray) -> list:
        # self._validate_vector_data(dataset_vector)
        if isinstance(dataset_vector, list):
            return dataset_vector
        else:
            return list(dataset_vector)