import numpy

def confusion_matrix(pred_y, validation_y):
    """
    [DESC]
            function to generate confusion matrix numbers
    [PARAMS]
            pred_Y : model prediction results
            column : validation set y
    [RETURN]
            2d matrix, column is predicted, row is actual class
    """
    # get max length of either pred y or val y
    # to handle the case of when either one has no amount of a certain value
    val_length = len(numpy.unique(validation_y))
    set_used = validation_y
    if(len(numpy.unique(pred_y)) > len(numpy.unique(validation_y))):
        val_length = len(numpy.unique(pred_y))
        set_used = pred_y
    conf_matrix = numpy.array([[0 for i in range(val_length)] for j in range(val_length)])
    
    # dictionary to translate value strings to indexes
    # google said lookups are faster with dictionaries
    val_dict = dict(enumerate(numpy.unique(set_used), 0))
    # reverse key and values
    val_dict = dict((v, k) for k, v in val_dict.items())
    print(val_dict)

    # fill in the confusion matrix
    for i in range(len(pred_y)):
        conf_matrix[val_dict[pred_y[i]]] [val_dict[validation_y[i]]] += 1

#     print(conf_matrix)
    return conf_matrix

def f1(precison, recall):
	return 2*(precison*recall/(precison + recall))

def print_scores(conf_matrix, att_names):
	n = len(att_names)
	for i in range(n):
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		for j in range(n):
			for k in range(n):
				if(j == k):
					tp += conf_matrix[j][k]
				elif(j == k and j != i):
					tn += conf_matrix[j][k]
				elif(j == i):
					fn += conf_matrix[j][k]
				elif(k == i):
					fp += conf_matrix[j][k]
		print("attribute: ", att_names[i])
		print("accuracy:", (tp+tn)/(tp+tn+fp+fn))
		print("precision:", (tp)/(tp+fp))
		print("recall:", (tp)/(tp+fn))
		print("f1:", f1((tp)/(tp+fp), (tp)/(tp+fn)))
		print()

