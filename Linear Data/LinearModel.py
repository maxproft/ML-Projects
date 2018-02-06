
#This assumes that the input data is of length 4

#my packages
import readdata

#other packages
import numpy as np
import tensorflow as tf


#Get Data
trainx,trainy=readdata.readdata(999)

testx,testy=readdata.readdata(100)

numtrain = len(trainx)
numtest = len(testx)

#hyperparameters
learning_rate = 0.5
epochs = 30

#placeholders
X = tf.placeholder("float32")
Y = tf.placeholder("float32")

#model weights
#tf.variables are trainable by default
W = tf.Variable(np.array(np.random.rand(1,4),dtype='float32'), name="weight")

#pred = tf.reduce_sum(tf.multiply(X, W))
pred = tf.matmul(W,X)

cost = tf.reduce_mean(tf.pow(tf.subtract(pred,Y), 2))

optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    print("Initial W: ", sess.run(W))
    print("Initial Cost:", sess.run(cost,feed_dict={X:trainx, Y:trainy}))    

    #looping through epochs
    for ep in range(epochs):
        if ep%5==0:
            print(ep)
            train_cost = sess.run(cost,feed_dict={X:trainx, Y:trainy})
            print("training cost=", train_cost)
        sess.run(optimiser, feed_dict={X:trainx, Y:trainy})

        
        
    
    print("optimiser finished\n")
    train_cost = sess.run(cost,feed_dict={X:trainx, Y:trainy})
    print("training cost=", train_cost)

    test_cost = sess.run(cost,feed_dict={X:testx,Y:testy})
    print("test cost=", test_cost)
    print("Final W:", sess.run(W))
    print("It should show [1.1, 2.2, 3.3, 4.4]")













    
    
