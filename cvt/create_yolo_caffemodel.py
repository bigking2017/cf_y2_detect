import caffe
import numpy as np
import sys, getopt

def transpose_matrix(inputWeight, rows, cols):
	inputWeight_t = np.zeros((rows*cols,1))
	for x in xrange(rows):
		for y in xrange(cols):
			inputWeight_t[y*rows + x] = inputWeight[x*cols + y]
	return inputWeight_t

def main(argv):
	model_filename = ''
	yoloweight_filename = ''
	caffemodel_filename = ''
	try:
		opts, args = getopt.getopt(argv, "hm:w:o:")
		print opts
	except getopt.GetoptError:
		print 'create_yolo_caffemodel.py -m <model_file> -w <yoloweight_filename> -o <caffemodel_output>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'create_yolo_caffemodel.py -m <model_file> -w <yoloweight_filename> -o <caffemodel_output>'
			sys.exit()
		elif opt == "-m":
			model_filename = arg
		elif opt == "-w":
			yoloweight_filename = arg
		elif opt == "-o":
			caffemodel_filename = arg
			
	print 'model file is ', model_filename
	print 'weight file is ', yoloweight_filename
	print 'output caffemodel file is ', caffemodel_filename
	print 'Note: This script is only for yolo small with out BN layer'
	net = caffe.Net(model_filename, caffe.TEST)
	params = net.params.keys()

	# read weights from file and assign to the network
	netWeightsInt = np.fromfile(yoloweight_filename, dtype=np.int32)
	transFlag = (netWeightsInt[0]>1000 or netWeightsInt[1]>1000) # transpose flag, the first 4 entries are major, minor, revision and net.seen
	print transFlag

	netWeightsFloat = np.fromfile(yoloweight_filename, dtype=np.float32)
	netWeights = netWeightsFloat[4:] # start from the 5th entry, the first 4 entries are major, minor, revision and net.seen
	print netWeights.shape
	count = 0
	print "Total ", params,  "params"
	#for pr in params:
	for idx in range(len(params)):
		pr = params[idx]
		print "Current params: ", pr
		print "net.params[pr] list len: ", len(net.params[pr])
		
		if pr[0:2]=='co': # conv with batchNorm
			weightSize = np.prod(net.params[pr][0].data.shape)
			print "  ||conv shape = ", net.params[pr][0].data.shape
			print "  ||weight size = ", weightSize
			biasSize = np.prod(net.params[pr][1].data.shape)
			print "  ||biasSize shape = ", net.params[pr][1].data.shape
			print "  ||biasSize size = ", biasSize
			
			# store conv bias
			begin = count 
			end   = begin + biasSize
			net.params[pr][1].data[...]	= np.reshape(netWeights[begin:end], net.params[pr][1].data.shape)
			
			# Store conv weight
			begin = count + biasSize
			end   = begin + weightSize
			net.params[pr][0].data[...]	= np.reshape(netWeights[begin:end], net.params[pr][0].data.shape)
			count = count + biasSize + weightSize

		elif pr[0:2]=='sc': # scale 
			print "Scale layer, do nothing"
			count = count + 0
		elif pr[0:2]=='bn': # batchNorm 
			print "bn layer, do nothing"
			count = count + 0
		elif pr[0:2]=='dr': # dropout 
			print "bn layer, do nothing"
			count = count + 0
		elif pr[0:2]=='fc':
			# bias
			biasSize = np.prod(net.params[pr][1].data.shape)
			print "  ||bias shape = ", net.params[pr][1].data.shape
			print "  ||bias size = ", biasSize
			net.params[pr][1].data[...] = np.reshape(netWeights[count:count+biasSize], net.params[pr][1].data.shape)
			count = count + biasSize
			# weights
			dims = net.params[pr][0].data.shape
			weightSize = np.prod(dims)
			print "  ||weight shape = ", net.params[pr][0].data.shape
			print "  ||weight size = ", weightSize
			if transFlag: # need transpose for fc layers
				net.params[pr][0].data[...] = np.reshape(transpose_matrix(netWeights[count:count+weightSize], dims[1],dims[0]), dims)
			else:
				net.params[pr][0].data[...] = np.reshape(netWeights[count:count+weightSize], dims)
			count = count + weightSize
		else:
			print "unknow layer!!!!!!!!!!!!!!!!!!!!!!!!!"
		print "Current count = ", count
		print ""
	net.save(caffemodel_filename)		
		
if __name__=='__main__':	
	main(sys.argv[1:])
