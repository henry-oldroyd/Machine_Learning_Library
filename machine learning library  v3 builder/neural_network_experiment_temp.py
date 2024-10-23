# %%
from neural_network import *
from activation_functions import *
from loss_functions import *
from layers import *

# %% [markdown]
# First I will get it working with 1 input neuron and one output neuron

# %%
traning_cycles = 1_000
training_reports = 20
learning_rate = 2**(-6)
num_input_neurons = 1
batch_size = 1
test_data_items = 10_000
x_range = [-10.0, 10.0]

# %%
def network_factory():
    return Neural_Network(
        num_input_neurons = num_input_neurons,
        loss_function = MC_MSE(),
        batch_size = batch_size,
        learning_rate = learning_rate,
        layers = [
            Neural_Layer(1, Identity_Function()),
        ]
    )\
        .set_full_validation(False)\
        .compile()

# %%
target_network = network_factory()
training_network = network_factory()

# %%
# heart surgery so provate attributes accessed
target_network._layers[0]._weights_m = np.array([[3.0]])
target_network._layers[0]._bias_v = np.array([5.0])



# %%
def run_test():
    X_data = np.random.uniform(
        low = x_range[0],
        high = x_range[1],
        size = (num_input_neurons, test_data_items)
    )
    Y_data = target_network.make_predicitons(X_data)


    loss = training_network.evaluate_model_on_test_data(
        X_test_data=X_data,
        Y_test_data=Y_data,
    )
    
    # training_network.get_layers()[0].print_parameters()

# %%
loss = run_test()

