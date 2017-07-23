import numpy

def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)

	return 1/(1+numpy.exp(-x))	

#input data
x = numpy.array([[0,0,1],
			  [0,1,1],
			  [1,0,1],
			  [1,1,1]])

#output data
y = numpy.array([[0],
			  [1],
			  [1],
			  [0]])

numpy.random.seed(1)

#synapses
syn0 = 2*numpy.random.random((3,4)) - 1
syn1 = 2*numpy.random.random((4,1)) - 1

#training step
for j in xrange(2000000):

	l0 = x
	l1 = nonlin(numpy.dot(l0, syn0))
	l2 = nonlin(numpy.dot(l1, syn1))

	#Back propagation of errors using the chain rule
	l2_error = y - l2
	if(j % 10) == 0:
		print("Error: " + str(numpy.mean(numpy.abs(l2_error))))

	l2_delta = l2_error*nonlin(l2, deriv=True)

	l1_error = l2_delta.dot(syn1.T)

	l1_delta = l1_error*nonlin(l1, deriv=True)

	#update weights (no learning rate term)
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

print("Output after training")
print(l2)
