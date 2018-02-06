import struct
import numpy as np

def readdata(dataset):
    
    if dataset == "train":
        namelbl = "Train/train-labels.idx1-ubyte"
        nameimg = "Train/train-images.idx3-ubyte" 
    elif dataset == "test":
        namelbl = "Test/t10k-labels.idx1-ubyte"
        nameimg = "Test/t10k-images.idx3-ubyte"
    
    #this is for labels
    with open(namelbl,'rb') as file:
        magic_nr, size = struct.unpack(">II", file.read(8))
        lbl = np.fromfile(file,dtype=np.int8)
        veclbl = np.zeros((len(lbl),10))
        for i in range(len(lbl)):
            veclbl[i,lbl[i]]=1
        

  
    #this is for image
    with open(nameimg,'rb') as file:
        magic_nr, size, rows, cols = struct.unpack(">IIII", file.read(16))
        img = np.fromfile(file,dtype=np.int8).reshape(len(lbl), rows, cols)/128.
        newimg = np.swapaxes(np.swapaxes([[i] for i in img],1,2),2,3)
        

    return np.array(newimg,dtype='float32'),veclbl

if __name__ == '__main__':
    testimg,testlbl = readdata("test")
    trainimg,trainlbl = readdata("train")
    print(np.shape(testimg))
    print(np.shape(testlbl))
    print(np.shape(trainimg))
    print(np.shape(trainlbl))
    print("\nExpected Output\n(10000, 28, 28, 1)\n(10000,10)\n(60000, 28, 28, 1)\n(60000,10)\n")
    print("plotting an image")
    oneimage = trainimg[900,:,:,0]
    print('The number is', trainlbl[900])
    import matplotlib.pyplot as plt
    imgplot = plt.imshow(oneimage,cmap='gray')
    plt.show()

    
