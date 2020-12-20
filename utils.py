import os
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)