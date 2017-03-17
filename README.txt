# MultiLayer_Perceptron
Multi Layer Perceptron in Tensorflow

Class : multilayer_perceptron
-------------------------------
Input to  __init__(): 
input_size = No of features
output_size = No of output classes
no_layers = No of Hidden Layers
hidden_layer_size = hidden layer size
reg_L2 = regularization parameter

Train Procedure:
----------------
# To Build Multi-Layer Perceptron with 10 Input Feautures and 2 output classes
no_of_epochs = 1000
mlp = multilayer_perceptron(input_size = 10,
                            output_size = 2,
                            no_layers= 1,
                            hidden_layer_size= 128,
                            reg_L2 = 0.01)
with tf.Session() as session:
   train_step = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(mlp.loss)
   session.run(tf.global_variables_initializer())
   
   for epoch in range(no_of_epochs):
      '''
        inputs, labels, dropout are numpy arrays.
      '''
      input_dictionary = {
        mlp.inputs : input,
        mlp.labels : labels,
        mlp.dropout : keep_prob
      }
      _, loss = session.run([train_step, mlp.loss], feed_dict = input_dictionary)
      print "Loss: "+str(loss)



