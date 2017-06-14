import caffe
import numpy as np
import os 

root = '/home/ebadawy/git/EmotiW2017/'
train_root = root + 'Train_AFEW/AlignedFaces_LBPTOP_Points/Faces/'

model = root + 'vgg_face_caffe/VGG_FACE_deploy.prototxt'
weights = root + 'vgg_face_caffe/VGG_FACE.caffemodel'

caffe.set_mode_cpu()
net = caffe.Net(model, weights, caffe.TEST)

mu = np.asarray([129.1863,104.7624,93.5940])

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)      
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

feat = {}

for file in os.listdir(train_root):
	feat[file] = {}
	for frame in os.listdir(train_root+file):
		image = caffe.io.load_image(train_root+file+'/'+frame)
		net.blobs['data'].data[...] = transformed_image
		net.forward()
		feat[file][frame[:-4]] = net.blobs['fc6'].data
	break

np.save('feat.npy', feat)