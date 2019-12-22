# coding:utf-8

from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import numpy as np

def postprocess(pred):
	pred = np.array(pred)
	pred = pred / np.max(pred) * 255.
	pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
	pred = gaussian_filter(pred, sigma=0.015*min(pred.shape[0], pred.shape[1]))
	pred = 255. * (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
	pred = Image.fromarray(pred.astype('uint8'))
	return pred