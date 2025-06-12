import mnist_loader
from chapter1_network import Network as net

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# First experiment
# print("Initializing the first experiment:")
# net_1 = net([784,30,10])
# net_1.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Second experiment
# print("Initializing the second experiment:")
# net_2 = net([784,100,10])
# net_2.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Third experiment
# print("Initializing the third experiment:")
# net_3 = net([784,100,10])
# net_3.SGD(training_data, 30, 10, 0.001, test_data=test_data)

# Fourth experiment
# print("Initializing the fourth experiment:")
# net_4 = net([784,100,10])
# net_4.SGD(training_data, 30, 10, 0.01, test_data=test_data)

# Fifth experiment
# print("Initializing the fifth experiment:")
# net_5 = net([784,30,10])
# net_5.SGD(training_data, 30, 10, 100.0, test_data=test_data)

# Sixth experiment -- Last exercise
# Goal: to create a network with just two layers and to optimize the learning rate
print("Initializing the sixth experiment:")
net_6 = net([784,10])
net_6.SGD(training_data, 15, 10, 1.0, test_data=test_data)
