import numpy as np
#This makes N pieces of data of length 4, and generates a linear model for y)
def readdata(N):
    
    xdata = np.random.randn(4,N)

    linear_multiplier = [1.1,2.2,3.3,4.4]

    ydata = [np.dot(linear_multiplier, xdata)]

    return np.array(xdata),np.array(ydata)



if __name__ == '__main__':
    x,y = readdata(99)
    print(np.shape(x))
    print(np.shape(y))
    print("output should be\n(4,99)\n(1,99)")
    
