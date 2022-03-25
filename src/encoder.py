from operator import indexOf
import numpy

def encode(labels):
    size = labels.max()
    encoded = numpy.array([[0 for i in range(size)] for j in range(len(labels))])
    for i in range(len(labels)):
        encoded[i][labels[i]-1] = 1
    return encoded

def decode(encoded):
    ret = []
    for i in range(len(encoded)):
        ret.append(numpy.where(encoded[i] == 1)[0][0] + 1) 
    return numpy.array(ret)