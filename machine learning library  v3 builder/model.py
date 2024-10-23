from __future__ import annotations
import numpy as np


from neural_network import Neural_Network
from data_encoding import Data_Encoding_Interface, Identity_Encoding
from ml_exceptions import ParameterChangeZero

class ML_Model():
    def __init__(self,
        neural_network: Neural_Network,
        whole_dataset_raw: np.array,
        X_encoding: Data_Encoding_Interface | None = None,
        Y_encoding: Data_Encoding_Interface | None = None,
        validation_batch_frequency: int = 50,
        learning_rate_half_life: int = -1,
        # learning_rate_decay_frequency: int = 1000,
        # learning_rate_decay_rate: float = 1.0,
    ) -> None:
        if not isinstance(neural_network, Neural_Network):
            raise TypeError("neural network parameter must be of type Neural Network")
        if not (isinstance(whole_dataset_raw, list) or isinstance(whole_dataset_raw, np.ndarray)):
            raise ValueError("whole_dataset_raw set must be of type list of np array")    
    
        if not (X_encoding is None or isinstance(X_encoding, Data_Encoding_Interface)):
            raise TypeError("if provided X_encoding must be of type Data_Encoding_Interface")
        if not (Y_encoding is None or isinstance(Y_encoding, Data_Encoding_Interface)):
            raise TypeError("if provided Y_encoding must be of type Data_Encoding_Interface")

        if not neural_network.is_compiled():
            raise ValueError("Provided neural network must be compiled")

        if not len(whole_dataset_raw[0]) == len(whole_dataset_raw[1]):
            raise ValueError("The same amound of X and Y data must be provided")

        # if not isinstance(learning_rate_decay_rate, float):
        #     raise TypeError("Learning rate must be of type float")
        # if not (0 < learning_rate_decay_rate <= 1):
        #     raise ValueError("Learning rate decay must be in the interval (0, 1]")
        
        # parameters_to_examine = {"validation_batch_frequency": validation_batch_frequency, "learning_rate_decay_frequency": learning_rate_decay_frequency}
        # for p_name, p_value in parameters_to_examine.items():
        #     if not isinstance(p_value, int):
        #         raise TypeError(f"{p_name} must be of type int")
        #     if not (0 < p_value):
        #         raise ValueError(f"{p_name} must be greater than 0]")


        
        if not isinstance(learning_rate_half_life, int):
            raise TypeError("learning_rate_half_life must be of type int")
        if not ((0 < learning_rate_half_life) or learning_rate_half_life == -1) :
            raise ValueError("Learning rate half life must be greater than 0 or be -1 for no decrease")

        if not isinstance(validation_batch_frequency, int):
            raise TypeError(f"validation_batch_frequency must be of type int")
        if not (0 < validation_batch_frequency):
            raise ValueError(f"validation_batch_frequency must be greater than 0")




        # add more validation in if you have time

        self._neural_network: Neural_Network = neural_network
        self._X_encoding: Data_Encoding_Interface = X_encoding if X_encoding is not None else Identity_Encoding()
        self._Y_encoding: Data_Encoding_Interface = Y_encoding if Y_encoding is not None else Identity_Encoding()

        # self._learning_rate_decay_rate = learning_rate_decay_rate
        # self._learning_rate_decay_frequency = learning_rate_decay_frequency

        self._learning_rate_decay_frequency = 100
        self._learning_rate_decay_rate = 2**(-learning_rate_half_life / self._learning_rate_decay_frequency) if learning_rate_half_life != -1 else 1

        self._validation_frequency = validation_batch_frequency

        # assert self._X_encoding is not None and self._Y_encoding is not None

        self._network_input_size = self._neural_network.get_input_size()
        self._network_output_size = self._neural_network.get_output_size()


        self._whole_dataset_size = len(whole_dataset_raw[0])
        self._whole_dataset = {}

        self._whole_dataset["X"] = self._X_encoding.encode_dataset(whole_dataset_raw[0])
        self._whole_dataset["Y"] = self._Y_encoding.encode_dataset(whole_dataset_raw[1])


        self._whole_dataset["X"] = self._whole_dataset["X"].reshape((
            self._whole_dataset_size,
            self._network_input_size,
        ))

        self._whole_dataset["Y"] = self._whole_dataset["Y"].reshape((
            self._whole_dataset_size,
            self._network_output_size,
        ))

        

        self._whole_dataset_indexes = list(range(self._whole_dataset_size))
        np.random.shuffle(self._whole_dataset_indexes)

        train_ratio = 0.7
        validatate_ratio = 0.1
        test_ratio = 0.2

        assert round(train_ratio + validatate_ratio + test_ratio, 12) == 1

        self._train_dataset_size = int(self._whole_dataset_size * train_ratio)
        self._validate_dataset_size = int(self._whole_dataset_size * validatate_ratio)
        self._test_dataset_size = int(self._whole_dataset_size * test_ratio)

        train_validate_partition = self._train_dataset_size
        validate_test_partition = self._train_dataset_size + self._validate_dataset_size

        self._train_dataset_indexes = self._whole_dataset_indexes[:train_validate_partition]
        self._validate_dataset_indexes = self._whole_dataset_indexes[train_validate_partition:validate_test_partition]
        self._test_dataset_indexes = self._whole_dataset_indexes[validate_test_partition:]


        self._validation_dataset = {}
        self._validation_dataset["X"] = np.array([self._whole_dataset["X"][index] for index in self._validate_dataset_indexes]).T
        self._validation_dataset["Y"] = np.array([self._whole_dataset["Y"][index] for index in self._validate_dataset_indexes]).T


        self._test_dataset = {}
        self._test_dataset["X"] = np.array([self._whole_dataset["X"][index] for index in self._test_dataset_indexes]).T
        self._test_dataset["Y"] = np.array([self._whole_dataset["Y"][index] for index in self._test_dataset_indexes]).T

        self._batch_size = self._neural_network.get_batch_size()
        self._minibatches_per_epoch = int(self._train_dataset_size / self._batch_size)


    def _generate_training_minibatches_1_epoch(self):
        np.random.shuffle(self._train_dataset_indexes)
        for minibatch_index in range(self._minibatches_per_epoch):
            minibatch_data_indicies = self._train_dataset_indexes[
                self._batch_size * minibatch_index : self._batch_size * (minibatch_index+1)
            ]
            X = np.array([self._whole_dataset["X"][index] for index in minibatch_data_indicies]).T
            Y = np.array([self._whole_dataset["Y"][index] for index in minibatch_data_indicies]).T

            yield X, Y

    def train(self, epochs: int) -> tuple[int, list[float], list[float]]:
        # returns validation loss, epochs completed

        if not isinstance(epochs, int):
            raise TypeError("epochs must be of type int")
        if epochs <= 0:
            raise ValueError("epochs must be a positive intiger")


        vaidation_losses = []
        learning_rates = []

        last_validation_loss = None
        # previous_validation_losses = [None for _ in range(self._validation_window)]
        # validation_index = 0
        end_training = False

        learning_rate = self._neural_network.get_learning_rate()
        for epoch_i in range(epochs):
            if end_training: break
            

            minibatch_iterator = self._generate_training_minibatches_1_epoch()
            for batch_i, (X, Y) in enumerate(minibatch_iterator):
                if end_training: break


                try:
                    training_loss = self._neural_network.train_model_on_minibatch(X, Y)
                except ParameterChangeZero:
                    end_training = True


                if batch_i % self._learning_rate_decay_frequency == 0:
                    learning_rate *= self._learning_rate_decay_rate
                    self._neural_network.set_learning_rate(learning_rate).compile()



                if batch_i % self._validation_frequency == 0:
                    new_validation_loss = self._neural_network.evaluate_model_on_test_data(self._validation_dataset["X"], self._validation_dataset["Y"])

                    if last_validation_loss is not None:
                        if new_validation_loss == last_validation_loss:
                            raise ValueError("validation loss not changing with training")

                        if new_validation_loss > last_validation_loss:
                            # end training 
                            end_training = True
                    
                    last_validation_loss = new_validation_loss
                    vaidation_losses.append(new_validation_loss)
                    learning_rates.append(learning_rate)




        return epoch_i+1, vaidation_losses, learning_rates

    def test(self):
        loss = self._neural_network.evaluate_model_on_test_data(self._test_dataset["X"], self._test_dataset["Y"])

        return loss
    
    def predict(self, X_data_raw):
        X_data_encoded = self._X_encoding.encode_dataset(X_data_raw).T
        P_data_encoded = self._neural_network.make_predicitons(X_data_encoded)
        P_data_decoded = self._Y_encoding.decode_dataset(P_data_encoded)

        return P_data_decoded


    def get_neural_network(self) -> Neural_Network:
        return self._neural_network