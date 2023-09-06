import numpy as np


class NN:
    def __init__(self, learning_rate=0.01, params=None):
        self.learning_rate = learning_rate
        self.nn_architecture = [        # 当前神经网络结构（两层）
            {"input_dim": 218, "output_dim": 64, "activation": "relu"},    # layer1
            {"input_dim": 64, "output_dim": 1, "activation": "sigmoid"}    # layer2
        ]
        if params is None:
            self.params = self.init_layers()
        else:
            self.params = params

    def init_layers(self, seed=666):
        np.random.seed(seed)
        params_values = {}

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.01
            params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * 0.01

        return params_values

    def identity(self, Z):
        return Z

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def identity_backward(self, dA, Z):
        return np.ones_like(dA)

    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="relu"):
        '''单层前向传播'''
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "relu":
            activation_func = self.relu
        elif activation == "sigmoid":
            activation_func = self.sigmoid
        elif activation == "identity":
            activation_func = self.identity
        else:
            raise Exception('Non-supported activation function')

        return activation_func(Z_curr), Z_curr  # 当前层输出，当前层输入

    def full_forward_propagation(self, X, params_values):
        '''完整前向传播'''
        memory = {}
        A_curr = X

        for idx, layer in enumerate(self.nn_architecture):   # 逐层传递神经网络参数
            layer_idx = idx + 1
            A_prev = A_curr

            activ_function_curr = layer["activation"]
            W_curr = params_values["W" + str(layer_idx)]
            b_curr = params_values["b" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        return A_curr, memory   # 返回当前神经网络最后值，中间过程

    def get_cost_value(self, Y_hat, Y):
        '''所有输入(Y_hat,Y)都是矩阵形式的=>为了便于计算np.dot()'''
        m = Y_hat.shape[1]
        #########################################################
        # print(Y_hat)
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        # cost = 1 / m * np.sum((Y - Y_hat) ** 2)
        return np.squeeze(cost)     # 压缩到1维向量

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        '''单层后向传播'''
        m = A_prev.shape[1]

        if activation == "relu":
            backward_activation_func = self.relu_backward
        elif activation == "sigmoid":
            backward_activation_func = self.sigmoid_backward
        elif activation == "identity":
            backward_activation_func = self.identity_backward
        else:
            raise Exception('Non-supported activation function')

        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, Y_hat, Y, memory, params_values):
        '''完整后向传播1-只计算了gradient'''
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        # dA_prev = Y_hat - Y

        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def update(self, params_values, grads_values):
        '''完整后向传播2-正式更新各个参数'''
        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            params_values["W" + str(layer_idx)] -= self.learning_rate * grads_values["dW" + str(layer_idx)]
            params_values["b" + str(layer_idx)] -= self.learning_rate * grads_values["db" + str(layer_idx)]

        return params_values

    def train(self, X, Y, epochs=1):
        params_values = self.params
        cost_history = []

        for i in range(epochs):
            Y_hat, cashe = self.full_forward_propagation(X, params_values)
            cost = self.get_cost_value(Y_hat, Y)
            cost_history.append(cost)

            grads_values = self.full_backward_propagation(Y_hat, Y, cashe, params_values)
            # print(grads_values)
            params_values = self.update(params_values, grads_values)

        self.params = params_values

        return params_values, cost_history      # 如果正确，cost 应该是逐渐下降的


