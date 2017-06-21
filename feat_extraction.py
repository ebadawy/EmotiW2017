import caffe
import numpy as np
import os 

root = os.getcwd() + '/'
train_root = root + 'Train_AFEW/AlignedFaces_LBPTOP_Points/Faces/'
# train_root = root + 'Val_AFEW/AlignedFaces_LBPTOP_Points_Val/Faces/'

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
len1 = len(os.listdir(train_root))
c1, c2 = 1, 1
for file in sorted(os.listdir(train_root)):
	feat[file] = {}
	len2 = len(os.listdir(train_root+file))
	for frame in sorted(os.listdir(train_root+file)):
		print('%d/%d %d/%d %s/%s' % (c1, len1, c2, len2, file, frame))
		image = caffe.io.load_image(train_root+file+'/'+frame)
		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[...] = transformed_image
		net.forward()
		feat[file][frame[:-4]] = net.blobs['fc6'].data[0].copy()
		c2 += 1
	c1 += 1
	c2 = 1
	if c1 % 100 == 0:
		np.save('feat_train_%d.npy'%c1, feat)

np.save('feat_train.npy', feat)