import numpy as np
import os
import matplotlib.image as mpimg

## floyd run --data bic2ggTB3rbGMTAvryPP5S:d1 --data chdojajSk9EFvbmZqDfohn:d2 "ls /d1; echo '******'; ls /d2"
## d1 train , d2 val
subset = 'val'
root = os.getcwd() + '/'

if subset == 'train':
    data_root = root + 'Train_AFEW/AlignedFaces_LBPTOP_Points/Faces/'
else:
    data_root = root + 'Val_AFEW/AlignedFaces_LBPTOP_Points_Val/Faces/'

data = {}
len1 = len(os.listdir(data_root))
c1, c2 = 1, 1
for file in sorted(os.listdir(data_root)):
    data[file] = {}
    len2 = len(os.listdir(data_root+file))
    for frame in sorted(os.listdir(data_root+file)):
        print('%d/%d %d/%d %s/%s' % (c1, len1, c2, len2, file, frame))
        image = mpimg.imread(data_root+file+'/'+frame)
        data[file][frame[:-4]] = image.copy()
        c2 += 1
    c1 += 1
    c2 = 1

np.save('image_data_%s_%d.npy'%(subset,c1), data)