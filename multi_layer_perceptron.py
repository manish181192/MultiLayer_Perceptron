import tensorflow as tf
class multilayer_perceptron:

    input_size = None
    output_size = None
    W = []
    B = []
    in_ = []
    out = []
    out_activated = []
    no_layers = None
    hidden_layer_size = None
    reg_L2 = None

    def __init__(self, input_size, output_size, no_layers=2, hidden_layer_size=128, reg_L2 = 0.0):

        self.input_size = input_size
        self.output_size = output_size
        self.no_layers = no_layers
        self.hidden_layer_size = hidden_layer_size
        self.reg_L2 = reg_L2

        self.l2_loss = tf.constant(0.0, name= "l2_loss")
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name="state_placeholder")
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name="reward_placeholder")
        self.dropout = tf.placeholder(dtype= tf.float32, name="dropout_input")

        self.in_.append(self.inputs)
        for i in range(self.no_layers):
            if i == 0:
                self.W.append(tf.Variable(tf.truncated_normal(shape=[self.input_size, self.hidden_layer_size], stddev= 0.01), dtype=tf.float32, name="weights" + str(i)))
                self.B.append(tf.Variable(tf.truncated_normal(shape=[self.hidden_layer_size], stddev= 0.01), dtype=tf.float32, name= "Bias"+str(i)))
            elif i == self.no_layers - 1:
                self.W.append(tf.Variable(tf.truncated_normal(shape=[self.hidden_layer_size, self.output_size], stddev= 0.01), dtype=tf.float32, name= "weights"+str(i)))
                self.B.append(tf.Variable(tf.truncated_normal(shape=[self.output_size], stddev= 0.01), dtype=tf.float32, name= "Bias"+str(i)))
            else:
                self.W.append(
                    tf.Variable(tf.truncated_normal(shape=[self.hidden_layer_size, self.hidden_layer_size], stddev= 0.01), dtype=tf.float32, name= "weights"+str(i)))
                self.B.append(tf.Variable(tf.truncated_normal(shape=[self.hidden_layer_size], stddev= 0.01), dtype=tf.float32, name= "Bias"+str(i)))
            self.l2_loss += tf.nn.l2_loss(self.W[i])
            self.l2_loss += tf.nn.l2_loss(self.B[i])
            self.out.append(tf.nn.xw_plus_b(self.in_[i], self.W[i], self.B[i], name= "Prediction"+str(i)))
            if i == 0 or i == self.no_layers - 1:
                self.dropout_t = tf.constant(1.0)
            else:
                self.dropout_t = self.dropout
            self.out_activated.append(tf.nn.relu(tf.nn.dropout(self.out[i], self.dropout_t)))
            self.in_.append(self.out_activated[i])

        self.prediction = self.out_activated[self.no_layers - 1]
        self.loss = tf.reduce_sum(tf.square(tf.sub(self.labels, self.prediction)) + self.reg_L2 * self.l2_loss)
