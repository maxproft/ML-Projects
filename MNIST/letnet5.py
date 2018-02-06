#Structure: LeNet-5 (*inspired)


#(N,28,28,1)
#--1--> Conv(filter_dim=5,step=1,pad=2)x6
#(N,28,28,6)---->
#--2--> max pooling (s=2,f=2)
#(N,(14,14,6))
#--3--> Conv(f=5,s=1)x16
#(N,10,10,16)
#--4--> max pooling (s=2,f=2)
# (N,5,5,16)
#--5--> FC, relu w/dropout
#(N,120)
#--6--> FC, relu w/dropout
#(N,84)
#--7--> FC, relu w/dropout *
#(N,40)*
#--8--> softmax
#(N,10)
#----->

#Cost function is L2 norm
#minimise with adamoptimizer and minibatches


#my packages
import readdata

#other packages
import numpy as np
import tensorflow as tf
import pickle


#######change lbl to a list of vectors, not integers###########




#Get Data
trainimg,trainlbl=readdata.readdata('train')
testimg,testlbl=readdata.readdata('test')

xshape,yshape,dummy = np.shape(trainimg[0])
numtrain = len(trainimg)
numtest = len(testimg)


#hyperparameters
learning_rate = 0.0001#3e-3 #tf documentation says default = 0.001 
beta1=0.94 #0.9
beta2=0.999 #0.999
epsilon=1e-8 #1e-8


dropout_rate = 0.80
#1=> no dropout

epochs = 30
batch_size = 256


print(learning_rate,beta1,dropout_rate)



#to make minibatches. 
def rand_minibatches(X,Y,batch_size):
    num_ex = len(X)

    #this randomly sorts list
    rand_ind = np.random.random(num_ex)  #this randomly assigns an order
    sortzip = sorted([[r,x,y] for r,x,y in zip(rand_ind,X,Y)])#sort by random number
    xy = [rxy[1:] for rxy in sortzip]

    #how many minibatches to make
    if num_ex%batch_size==0:
        num_mini = int(num_ex/batch_size)
    else:
        num_mini = int(num_ex/batch_size)+1

    #making the minibatches
    minibatches = [np.array(xy[i*batch_size:(i+1)*batch_size]).T for i in range(num_mini)] 
    #To turn make it the right shape, and all arrays
    newminibatches = [[np.stack(batch[0],axis=0), np.vstack(batch[1])] for batch in minibatches]
    return newminibatches[::-1] #so the final, smaller, minibatch is not last

minibatches = rand_minibatches(trainimg,trainlbl,batch_size)
num_minibatches = len(minibatches)

#data placeholders
X = tf.placeholder("float32",shape = [None,xshape,yshape,1])
Y = tf.placeholder("float32",shape = [None,10])
DROP = tf.placeholder("float32",shape = None)

### Making NN architecture ###
#see top of document for what numbers refer to

try:#try getting weights from file
    weight_open= open('weightdict.pickle','rb')
    weightdict = pickle.load(weight_open)
    for i in weightdict: #weights need to set as initilizers
        weightdict[i]=tf.constant_initializer(weightdict[i])


    #see top of document for what numbers refer to
    pad1_width = tf.constant([[0,0],[2,2],[2,2],[0,0]])
    pad1=tf.pad(X,pad1_width,'CONSTANT')
    filter1 = tf.layers.conv2d(pad1,filters=6,kernel_size=5,strides=(1,1),padding='VALID',name='filter1',kernel_initializer=weightdict['filter1kernel'],bias_initializer=weightdict['filter1bias'])    
    pooling2 = tf.nn.max_pool(filter1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    filter3 = tf.layers.conv2d(pooling2,filters=16,kernel_size=5,strides=(1,1),padding='VALID',name='filter3',kernel_initializer=weightdict['filter3kernel'],bias_initializer=weightdict['filter3bias'])
    pooling4 = tf.nn.max_pool(filter3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    flatten5 = tf.contrib.layers.flatten(pooling4)
    dropout5 = tf.nn.dropout(flatten5,DROP)
    fc5 = tf.contrib.layers.fully_connected(dropout5, num_outputs=120,activation_fn=tf.nn.relu, scope='fc5',weights_initializer=weightdict['fc5weights'],biases_initializer=weightdict['fc5biases'])
    dropout6 = tf.nn.dropout(fc5,DROP)
    fc6 = tf.contrib.layers.fully_connected(dropout6, num_outputs=84,activation_fn=tf.nn.relu,scope='fc6',weights_initializer=weightdict['fc6weights'],biases_initializer=weightdict['fc6biases'])

    dropout7 = tf.nn.dropout(fc6,DROP)
    fc7 = tf.contrib.layers.fully_connected(dropout7, num_outputs=84,activation_fn=tf.nn.relu,scope='fc7',weights_initializer=weightdict['fc7weights'],biases_initializer=weightdict['fc7biases'])
    dropout8 = fc7#tf.nn.dropout(fc7,DROP)
    sm8 = tf.contrib.layers.fully_connected(dropout8, num_outputs=10,activation_fn=tf.nn.softmax,scope='sm8',weights_initializer=weightdict['sm8weights'],biases_initializer=weightdict['sm8biases'])

except FileNotFoundError: #if no weights exist
    #see top of document for what numbers refer to
    pad1_width = tf.constant([[0,0],[2,2],[2,2],[0,0]])
    pad1=tf.pad(X,pad1_width,'CONSTANT')
    filter1 = tf.layers.conv2d(pad1,filters=6,kernel_size=5,strides=(1,1),padding='VALID',name='filter1')
    pooling2 = tf.nn.max_pool(filter1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    filter3 = tf.layers.conv2d(pooling2,filters=16,kernel_size=5,strides=(1,1),padding='VALID',name='filter3')
    pooling4 = tf.nn.max_pool(filter3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    flatten5 = tf.contrib.layers.flatten(pooling4)
    dropout5 = tf.nn.dropout(flatten5,DROP)
    fc5 = tf.contrib.layers.fully_connected(dropout5, num_outputs=120,activation_fn=tf.nn.relu, scope='fc5')
    dropout6 = tf.nn.dropout(fc5,DROP)
    fc6 = tf.contrib.layers.fully_connected(dropout6, num_outputs=84,activation_fn=tf.nn.relu,scope='fc6')

    dropout7 = tf.nn.dropout(fc6,DROP)
    fc7 = tf.contrib.layers.fully_connected(dropout7, num_outputs=40,activation_fn=tf.nn.relu,scope='fc7')
    dropout8 = fc7#tf.nn.dropout(fc7,DROP)
    sm8 = tf.contrib.layers.fully_connected(dropout8, num_outputs=10,activation_fn=tf.nn.softmax,scope='sm8')




cost = tf.reduce_sum(tf.pow(tf.subtract(sm8,Y),2))
optimiser = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(cost)
prop_correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(sm8,1), tf.argmax(Y,1)), tf.float32))

init = tf.global_variables_initializer()



#To save weights:
def getweight(name,dtype):
    with tf.variable_scope(name, reuse=True):
        data = sess.run(tf.get_variable(dtype))
    return data



with tf.Session() as sess:
    sess.run(init)
    print("initial Cost:     ",sess.run(cost, feed_dict={X:trainimg,Y:trainlbl,DROP:dropout_rate})/numtrain)
    for ep in range(epochs):
        totalcost = 0
        for batch in minibatches:
            _, tempcost = sess.run([optimiser,cost], feed_dict={X:batch[0],Y:batch[1],DROP:dropout_rate})
            totalcost+=tempcost
        print("cost after ep", ep+1, "is", totalcost/numtrain)


    #saving weights
    weightdict = {}
    weightdict['filter1kernel']=getweight('filter1','kernel')
    weightdict['filter1bias']=getweight('filter1','bias')
    weightdict['filter3kernel']=getweight('filter3','kernel')
    weightdict['filter3bias']=getweight('filter3','bias')
    weightdict['fc5weights']=getweight('fc5','weights')
    weightdict['fc5biases']=getweight('fc5','biases')
    weightdict['fc6weights']=getweight('fc6','weights')
    weightdict['fc6biases']=getweight('fc6','biases')
    weightdict['fc7weights']=getweight('fc7','weights')
    weightdict['fc7biases']=getweight('fc7','biases')
    weightdict['sm8weights']=getweight('sm8','weights')
    weightdict['sm8biases']=getweight('sm8','biases')

    weight_save = open('weightdict.pickle','wb')
    pickle.dump(weightdict, weight_save)
    weight_save.close()



    #Final Result
    print("proportion correct of train set:", sess.run(prop_correct, feed_dict={X:trainimg,Y:trainlbl,DROP:1.}))
    print("proportion correct of test set:", sess.run(prop_correct, feed_dict={X:testimg,Y:testlbl,DROP:1.}))
    

    





    
    
