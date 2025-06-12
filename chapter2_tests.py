import mnist_loader
from chapter1_network import Network as net1
from chapter2_network import Network as net2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print('Initializing comparison experiment.')
print('Unvectorized network results:')
test1 = net1([784,100,10])
test1.SGD(training_data, 30, 10, 3.0, test_data=test_data)

print('Vectorized network results:')
test2 = net2([784,100,10])
test2.SGD(training_data, 30, 10, 3.0, test_data=test_data)
