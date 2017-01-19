import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

#decode pickle files
def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

#original images are stored as one-dimensial array with 3072 elements
#to process it using matplotlib, we need to convert it to three dimension array of size 32 * 32 * 3
#The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue, in original image representation

def convertImg(img):
	newImg = np.zeros((32, 32, 3))
 	for i in range(len(img)):
 		newImg[i % 1024 / 32, i % 32, i / 1024] = img[i]
 	return newImg

"""
img1 = imread('testPic.jpg')
plt.imshow(np.uint8(img1))
plt.show()
"""

dict1 = unpickle("cifar-10-batches-py/data_batch_1")
img1 = dict1['data'][0] #the first picture

"""
print dict1['labels'][0] #the class to which the picture belongs to, represented as numbers
print dict1['batch_label'][0] #five train batches and one test batch
print dict1['filenames'][0]
"""
img2 = convertImg(img1)
"""
print img1.dtype, img1.shape
print img2.dtype, img2.shape
"""
plt.imshow(np.uint8(img2)) #Convert float64 to uint8, otherwise the picture will look strange
plt.show()
