from cProfile import label
from operator import indexOf
import numpy


def encode(values):
    """
        [DESC]
                function to encode array of values to one hot encoding
                classes encoded alphabetically
        [PARAMS]
                labels : array of values to be encoded
        [RETURN]
                2D np.ndarray
        """
    unique = numpy.unique(values)
    labels = []
    for i in range(len(values)):
        labels.append(numpy.where(unique == values[i])[0][0] + 1)
    size = max(labels)
    encoded = numpy.array([[0 for i in range(size)] for j in range(len(labels))])
    for i in range(len(labels)):
        encoded[i][labels[i]-1] = 1
    return encoded

def getUniqueLabels(labels):
    """
        literally just calls numpy unique to get existing classes lol
        returned a unique string of values in 'labels' alphabetically
    """
    return numpy.unique(labels)

def decode(encoded, labels = []):
    """
        [DESC]
                Method to decode a 2D numpy array to a string of ints
                or if specified, its corresponding string labels
        [PARAMS]
                encoded : 2D array to decode
                labels : labels for each index
        [RETURN]
                np.ndarray
    """
    ret = []
    for i in range(len(encoded)):
        ret.append(numpy.where(encoded[i] == 1)[0][0] + 1)
    if(len(labels) != 0):
        for i in range(len(encoded)):
            ret[i] = labels[ret[i] - 1]
    return numpy.array(ret)