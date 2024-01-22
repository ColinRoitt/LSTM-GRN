import numpy as np
import matplotlib.pyplot as plt

class ERNN:
    def __init__(self, units, weights_in, continuous = False) -> None:
        # singleton
        # global global_elstm
        # if(global_elstm != None):
        #     self.set_weights(weights_in)    
        #     return global_elstm
        
        self.units = units
        self.genome_size = ERNN.get_genome_size(self.units)
        self.set_weights(weights_in)
        self.use_dense = True
        self.continuous = continuous
        self.state = (np.zeros((1, self.units)), np.zeros((1, self.units)))
        self.state_history = []

    # static function to get genome size
    @staticmethod
    def get_genome_size(units):
        return 4*units + 4*units*units + 4*units + 1*units + 1*units

    def update_state(self, state):
        self.state_history.append(self.state)
        self.state = state

    def set_weights(self, weights_in):
        # throw error if weights_in is not the right size
        if len(weights_in) != self.genome_size:
            raise Exception("weights_in is not the right size, expected " + str(self.genome_size) + " but got " + str(len(weights_in)))
        
        self.weights_flat = weights_in

        # building the weights

        lstm_w_1_len = 4*self.units
        lstm_w_2_len = lstm_w_1_len*self.units
        lstm_b_1_len = lstm_w_1_len
        dense_w_1_len = 1*self.units
        dense_b_1_len = 1*self.units

        lstm_w_1 = np.array(
            self.weights_flat[:lstm_w_1_len]
        ).reshape((1, 4*self.units))

        weights_left = self.weights_flat[lstm_w_1_len:]

        lstm_w_2 = np.array(
            weights_left[:lstm_w_2_len]
        ).reshape((self.units, 4*self.units))

        weights_left = weights_left[lstm_w_2_len:]

        lstm_b_1 = np.array(
            weights_left[:lstm_b_1_len]
        ).reshape((4*self.units))

        weights_left = weights_left[lstm_b_1_len:]

        dense_w_1 = np.array(
            weights_left[:dense_w_1_len]
        ).reshape((self.units, 1))

        weights_left = weights_left[dense_w_1_len:]

        dense_b_1 = np.array(
            weights_left[:dense_b_1_len]
        ).reshape((self.units, 1))

        self.weights = [
            lstm_w_1,
            lstm_w_2,
            lstm_b_1,
            dense_w_1,
            dense_b_1,
        ]

        return self.weights
    
    def activation(self, x):
        # elment wise sigmoid over n x m matrix
        # return 1 / (1 + np.exp(-x))
    
        # elemtn wise tanh over n x m matrix
        return np.tanh(x) 
    
    def lstm_unit(self, Xt, ct_prev, ht_prev):

        k_i, k_f, k_c, k_o = np.split(self.weights[0], 4, 1)
        x_i = np.dot(Xt, k_i)
        x_f = np.dot(Xt, k_f)
        x_c = np.dot(Xt, k_c)
        x_o = np.dot(Xt, k_o)

        b_i, b_f, b_c, b_o = np.split(self.weights[2], 4, 0)
        x_i = np.add(x_i, b_i)
        x_f = np.add(x_f, b_f)
        x_c = np.add(x_c, b_c)
        x_o = np.add(x_o, b_o)

        i = self.activation(
            x_i + np.dot(ht_prev, self.weights[1][:, : self.units])
        )
        f = self.activation(
            x_f
            + np.dot(
                ht_prev, self.weights[1][:, self.units : self.units * 2]
            )
        )
        c = f * ct_prev + i * self.activation(
            x_c
            + np.dot(
                ht_prev,
                self.weights[1][:, self.units * 2 : self.units * 3],
            )
        )
        o = self.activation(
            x_o
            + np.dot(ht_prev, self.weights[1][:, self.units * 3 :])
        )
        return c, o
    
    def forward_dense(self, n):
        return self.activation(np.dot(n, self.weights[3]) + self.weights[4])
        
    def forward_lstm(self, input_series):
        # iterate over the input series, run through the lstm unit, return the final short term state
        input_series = np.reshape(input_series, (len(input_series), 1))

        if self.continuous:
            if len(input_series) != self.units:
                raise Exception("input is not the right size, expected " + str(self.units) + " but got " + str(len(input_series)))
            ht_prev = self.state[1]
            ct_prev = self.state[0]
            ct_prev, ht_prev = self.lstm_unit(input_series, ct_prev, ht_prev)
        else:
            ht_prev = np.zeros((1, self.units))
            ct_prev = np.zeros((1, self.units))
            for Xt in input_series:
                ct_prev, ht_prev = self.lstm_unit(Xt, ct_prev, ht_prev)
        
        self.update_state((ct_prev, ht_prev))
        return ht_prev
    
    def forward(self, input_series):
        # throw error if input_series is not 1d array or python list
        if not isinstance(input_series, (list, np.ndarray)):
            raise Exception("input_series is not a list or numpy array")
        # if isinstance(input_series, np.ndarray) and len(input_series.shape) != 1:
        #     raise Exception("input_series is not a 1d numpy array")
        if len(input_series) == 0:
            raise Exception("input_series is empty")
        # if not isinstance(input_series[0], (int, float)):
        #     raise Exception("input_series is not a 1d array of numbers")

        
        if(self.use_dense):
            return self.forward_dense(self.forward_lstm(input_series))
        else:
            return self.forward_lstm(input_series)
    

# write a test case to check the output
if __name__ == "__main__":
    # create a random weight vector
    units = 2
    seq_len = 20
    weights = np.random.rand(ELSTM_Dynamic.get_genome_size(units))
    # create an lstm object
    elstm = ELSTM_Dynamic(units, weights)
    # create a random input series
    input_series = np.random.rand(seq_len)
    # run the forward pass
    out = elstm.forward(input_series)
    print(out)
