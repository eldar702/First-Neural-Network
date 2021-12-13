# Eldar Shlomi 205616634
##########################                    Import & Globals :                    ####################################
import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_derivative = lambda x: sigmoid(x) * (1 - sigmoid(x))
softmax = lambda x: np.exp(x) / np.sum(np.exp(x))


#######################################################################################################################
##########################                  Neural Network Class :                  ####################################
#######################################################################################################################
class NN:
    def __init__(self, X_train, Y_train, X_test):
        ## data preparation ##
        self.divide_train_fraction = 0.8
        self.divide_test_fraction = 1 - self.divide_train_fraction
        self.classes_values, self.numOf_y_classes = discover_num_of_tags(Y_train)

        # create the new data sets for the training
        self.train_X = X_train
        self.train_Y = Y_train
        self.test_X = X_test
        self.XY_train = np.array(list(zip(self.train_X, self.train_Y)), dtype=object)
        self.new_XY_train, self.new_XY_test, self.new_X_test, self.new_Y_test, \
        self.new_X_train, self.new_Y_train = np.empty([6, 6])

        # hyper parameters
        self.hidden_layer_size = 128
        self.ETA = 0.00
        self.epochs_number = 10

        np.random.seed(42)
        self.network = np.empty([2, 2])


    def start(self):
        ## check hyper parameters: (uncomment stage 2 and put between stage 1 to 3)
        # checked_hidden_layer_size = [num for num in range(290, 300, 16)]
        # checked_ETA = [num / 1000 for num in range(10, 15, 5)]
        # checked_epoch_num = [num for num in range(25, 60, 5)]
        # for layer_size in checked_hidden_layer_size:
        #     for ETA in checked_ETA:
        #         for epoch in checked_epoch_num:
        #             self.network = self.init_weights_and_bias(self.test_X.shape[1])
        #             self.hidden_layer_size = layer_size
        #             self.ETA = ETA
        #             self.epochs_number = epoch
        #             self.stage_3_train()
        #             tested_arr = self.stage_4_test()
        #             accuracy_precent = accuracy(self.new_test_y, tested_arr)
        #             print_to_file(ETA, epoch, layer_size, accuracy_precent)
        # with open('choosing_hyper_parameters.txt') as file:
        #     lines = [line.split() for line in file]
        #     lines.sort(key=lambda s: s[-1])
        #     file.close()
        #     file = open("choosing_hyper_parameters.txt", "w")
        #     for line in lines:
        #         file.write(str(line))
        self.stage_1_prepering_the_data()
        self.stage_2_choose_hyper_parameters()
        self.stage_3_train()
        self.stage_4_test()

    # init the network (first weights and bias for the layers)
    def init_weights_and_bias(self, size_of_first_line):
        W1 = np.random.uniform(-0.08, 0.08, (size_of_first_line, self.hidden_layer_size))
        b1 = np.random.uniform(-0.08, 0.08, (self.hidden_layer_size, 1))
        W2 = np.random.uniform(-0.08, 0.08, (self.hidden_layer_size, self.numOf_y_classes))
        b2 = np.random.uniform(-0.08, 0.08, (self.numOf_y_classes, 1))
        weights_and_biases = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        return weights_and_biases

    def stage_1_prepering_the_data(self, ):
        input_size = len(self.train_X[0])
        new_train_length = int(input_size * self.divide_train_fraction)
        self.new_XY_train = self.XY_train[:new_train_length]
        self.new_X_train = np.array([x[0] for x in self.new_XY_train])
        self.new_Y_train = np.array([y[1] for y in self.new_XY_train])

        self.new_XY_test = np.array(self.XY_train[new_train_length:])
        self.new_X_test = np.array([x[0] for x in self.new_XY_test])
        self.new_Y_test = np.array([y[1] for y in self.new_XY_test])
        self.network = self.init_weights_and_bias(self.test_X.shape[1])

    def stage_2_choose_hyper_parameters(self):

        # hyper parameters
        self.hidden_layer_size = 290
        self.ETA = 0.01
        self.epochs_number = 20

    def stage_3_train(self):
        for epoch in range(self.epochs_number):
            self.train_X, self.train_Y = shuffle(self.train_X, self.train_Y)
            for x, y in zip(self.train_X, self.train_Y):
                forward_prepropagation_arr = self.forward_propagation(x.reshape(x.size, 1), y)
                backward_prepropagation_arr = self.back_propagation(forward_prepropagation_arr)
                for key in self.network:
                    self.network[key] = self.network[key] - (backward_prepropagation_arr[key] * self.ETA)

    def stage_4_test(self):

        w1, w2, b1, b2 = [self.network[key] for key in ('W1', 'W2', 'b1', 'b2')]
        W1t = np.transpose(w1)
        W2t = np.transpose(w2)
        file = open("test_y", "w")

        for x in self.test_X:
            x = x.reshape(x.size, 1)
            W1Tx = np.dot(W1t, x)
            W1Tx = W1Tx.reshape(len(W1Tx), 1)
            z1 = np.add(W1Tx, b1)
            h1 = sigmoid(z1)
            W2Tx = np.dot(W2t, h1)
            z2 = np.add(W2Tx, b2)
            h2 = softmax(z2)
            y_hat = np.argmax(h2)
            file.write(str(y_hat) + "\n")
        file.close()

    # forward propagation func
    def forward_propagation(self, x, y, ):
        W1, b1, W2, b2 = [self.network[key] for key in ('W1', 'b1', 'W2', 'b2')]
        W1t = np.transpose(W1)
        W2t = np.transpose(W2)
        W1Tx = np.dot(W1t, x)
        z1 = W1Tx + b1
        h1 = sigmoid(z1)
        W2Tx = np.dot(W2t, h1)
        z2 = W2Tx + b2
        y_hat = softmax(z2)

        result_arr = {'x': x, 'y': y, 'y_hat': y_hat, 'z1': z1, 'z2': z2, 'h1': h1}
        result_arr['W1'] = self.network['W1']
        result_arr['W2'] = self.network['W2']
        result_arr['b1'] = self.network['b1']
        result_arr['b2'] = self.network['b2']
        return result_arr

    # backward propagation func
    def back_propagation(self, forward_prepropagation_arr):
        x, y, y_hat, z1, h1, z2, W2 = [forward_prepropagation_arr[key] for key in
                                       ('x', 'y', 'y_hat', 'z1', 'h1', 'z2', 'W2')]
        y_prob_vec = np.zeros((10, 1))
        y_prob_vec[int(y)] = 1
        derivative_z2 = (y_hat - y_prob_vec)
        derivative_W2 = np.dot(derivative_z2, h1.T)
        derivative_b2 = np.copy(derivative_z2)
        derivative_z1 = np.dot(W2, derivative_z2) * sigmoid_derivative(z1)
        derivative_W1 = np.dot(derivative_z1, x.T)
        derivative_b1 = np.copy(derivative_z1)
        return {'W1': derivative_W1.T, 'W2': derivative_W2.T, 'b1': derivative_b1, 'b2': derivative_b2, }


#########################                Helper Functions :                  ##############################

# Function to discover how much different tags there is in the train y file
# In the example there is 10 - from 0 to 9.
def discover_num_of_tags(y_arr):
    counter = 0
    already_discovered = []
    for tag in y_arr:
        tag = int(tag)
        if tag in already_discovered:
            continue
        already_discovered.append(int(tag))
        counter += 1
    return already_discovered, counter


def init_matrix(x, y):
    layer = np.random.uniform(-1., 1., size=(x, y)) / np.sqrt(x * y)
    return layer.astype(np.float32)


# Shuffle function for two array
def shuffle(arr_1, arr_2):
    # suffle the data
    toShuffle = list(zip(arr_1, arr_2))
    np.random.shuffle(toShuffle)
    arr_1, arr_2 = zip(*toShuffle)
    return arr_1, arr_2


def accuracy(test_y, checked_y):
    counter = 0
    index = 0
    for y in test_y:
        if y == checked_y[index]:
            counter += 1
        index += 1
    return float(counter / len(test_y))


def print_to_file(eta, epoch_num, layer_size, accuracy):
    line = "the chosed hyper parameters are: layer_size " + str(layer_size) + ". ETA: " \
           + str(eta) + ". epochs number: " + str(epoch_num) + ".     the accuracy was: " + str(accuracy) + '\n'
    file = open("choosing_hyper_parameters.txt", "a")
    file.write(line)
    file.close()


#########################                Main Function :                  ##############################
def main():
    train_X = np.loadtxt("train_x")
    train_Y = np.loadtxt("train_y")
    test_X = np.loadtxt("test_x")

    # Normalize the data
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    # create and start the neural network
    first_NN = NN(train_X, train_Y, test_X)
    first_NN.start()


if __name__ == "__main__":
    main()
