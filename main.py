import os
import time
import random
import shutil
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim

import torch.utils.data
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

from utils import *
from dataset.datasets_csv import Imagefolder_csv
from model import CrossAttention


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='/home/krish/krish/Datasets/FewShotDatasetSplits/miniImageNet/', help='base directory containing csv and images folder')
parser.add_argument('--iter_name', default='miniImageNet-CTX')
parser.add_argument('--mode', default='train', help='train|val|test')
parser.add_argument('--outf', default='./results/')
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--workers', type=int, default=6)

parser.add_argument('--imageSize', type=int, default=224)
parser.add_argument('--episodeSize', type=int, default=1, help='the mini-batch size of training')
parser.add_argument('--testepisodeSize', type=int, default=1, help='one episode is taken as a mini-batch')
parser.add_argument('--epochs', type=int, default=30, help='the total number of training epoch')

parser.add_argument('--episode_train_num', type=int, default=10000, help='the total number of training episodes')
parser.add_argument('--episode_val_num', type=int, default=1000, help='the total number of evaluation episodes')
parser.add_argument('--episode_test_num', type=int, default=1000, help='the total number of testing episodes')

parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--shot_num', type=int, default=1, help='the number of shot')
parser.add_argument('--query_num', type=int, default=15, help='the number of queries')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--lr_decay', type=int, default=5, help='learning rate decay epochs, default=5')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--print_freq', '-p', default=1000, type=int, metavar='N', help='print frequency (default: 100)')

opt = parser.parse_args()



# ======================================= Define functions =============================================

def adjust_learning_rate(optimizer, epoch_num):
	"""Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
	lr = opt.lr * (0.05 ** (epoch_num // opt.lr_decay))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch_index, F_txt):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	scaler = GradScaler()


	end = time.time()
	for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(train_loader):

		# Measure data loading time
		data_time.update(time.time() - end)

		# Convert query and support images
		query_images = torch.cat(query_images, 0)
		input_var1 = query_images.cuda()

		input_var2 = []
		for i in range(len(support_images)):
			temp_support = support_images[i]
			temp_support = torch.cat(temp_support, 0)
			temp_support = temp_support.cuda()
			input_var2.append(temp_support)
		input_var2 = torch.stack(input_var2)

		# Deal with the targets
		target = torch.cat(query_targets, 0)
		target = target.cuda()

		# Calculate the output
		with autocast():
			output = model(input_var1, input_var2)
			loss = criterion(output, target)

		# Compute gradient and do SGD step
		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	  
		# Measure accuracy and record loss
		prec1, _ = accuracy(output, target, topk=(1,3))
		losses.update(loss.item(), query_images.size(0))
		top1.update(prec1[0], query_images.size(0))


		# Measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		#============== print the intermediate results ==============#
		if episode_index % opt.print_freq == 0 and episode_index != 0:

			print('Eposide-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

			print('Eposide-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1), file=F_txt)

def validate(val_loader, model, criterion, epoch_index, best_prec1, F_txt):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
  

	# switch to evaluate mode
	model.eval()
	accuracies = []


	end = time.time()
	for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):

		# Convert query and support images
		query_images = torch.cat(query_images, 0)
		input_var1 = query_images.cuda()

		input_var2 = []
		for i in range(len(support_images)):
			temp_support = support_images[i]
			temp_support = torch.cat(temp_support, 0)
			temp_support = temp_support.cuda()
			input_var2.append(temp_support)
		input_var2 = torch.stack(input_var2)

		# Deal with the targets
		target = torch.cat(query_targets, 0)
		target = target.cuda()

		# Calculate the output 
		with autocast():
			with torch.no_grad():
				output = model(input_var1, input_var2)
				loss = criterion(output, target)


		# measure accuracy and record loss
		prec1, _ = accuracy(output, target, topk=(1, 3))
		losses.update(loss.item(), query_images.size(0))
		top1.update(prec1[0], query_images.size(0))
		accuracies.append(prec1)


		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		#============== print the intermediate results ==============#
		if episode_index % opt.print_freq == 0 and episode_index != 0:

			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1), file=F_txt)

		
	print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1))
	print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1), file=F_txt)

	return top1.avg, accuracies


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res



# ======================================== Settings of path ============================================
# saving path
opt.outf = "{}_{}_{}Way_{}Shot".format(opt.outf, opt.iter_name, opt.way_num, opt.shot_num)
make_dir(opt.outf)

# save the opt and results to a txt file
txt_save_path = os.path.join(opt.outf, 'opt_resutls.txt')
F_txt = open(txt_save_path, 'a+')
print(opt)
print(opt, file=F_txt)



# ========================================== Model Config ===============================================
global best_prec1, epoch_index
best_prec1 = 0
epoch_index = 0

model = CrossAttention()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

# optionally resume from a checkpoint
if opt.resume:
	if os.path.isfile(opt.resume):
		print("=> loading checkpoint '{}'".format(opt.resume))
		checkpoint = torch.load(opt.resume)
		epoch_index = checkpoint['epoch_index']
		best_prec1 = checkpoint['best_prec1']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch_index']))
		print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch_index']), file=F_txt)
	else:
		print("=> no checkpoint found at '{}'".format(opt.resume))
		print("=> no checkpoint found at '{}'".format(opt.resume), file=F_txt)

# print the architecture of the network
print(model) 
print(model, file=F_txt) 


# ======================================== Training phase ===============================================
print('\n............Start training............\n')
start_time = time.time()


for epoch_item in range(opt.epochs):
	print('===================================== Epoch %d =====================================' %epoch_item)
	print('===================================== Epoch %d =====================================' %epoch_item, file=F_txt)
	adjust_learning_rate(optimizer, epoch_item) 
	

	# ======================================= Folder of Datasets =======================================
	# image transform & normalization
	ImgTransform = transforms.Compose([
			transforms.Resize((opt.imageSize, opt.imageSize)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			])

	trainset = Imagefolder_csv(
		data_dir=opt.base_dir, mode=opt.mode, image_size=opt.imageSize, transform=ImgTransform,
		episode_num=opt.episode_train_num, way_num=opt.way_num, shot_num=opt.shot_num, query_num=opt.query_num
	)
	valset = Imagefolder_csv(
		data_dir=opt.base_dir, mode='val', image_size=opt.imageSize, transform=ImgTransform,
		episode_num=opt.episode_val_num, way_num=opt.way_num, shot_num=opt.shot_num, query_num=opt.query_num
	)
	testset = Imagefolder_csv(
		data_dir=opt.base_dir, mode='test', image_size=opt.imageSize, transform=ImgTransform,
		episode_num=opt.episode_test_num, way_num=opt.way_num, shot_num=opt.shot_num, query_num=opt.query_num
	)

	print('Trainset: %d' %len(trainset))
	print('Valset: %d' %len(valset))
	print('Testset: %d' %len(testset))
	print('Trainset: %d' %len(trainset), file=F_txt)
	print('Valset: %d' %len(valset), file=F_txt)
	print('Testset: %d' %len(testset), file=F_txt)



	# ========================================== Load Datasets =========================================
	train_loader = torch.utils.data.DataLoader(
		trainset, batch_size=opt.episodeSize, shuffle=True, 
		num_workers=int(opt.workers), drop_last=True, pin_memory=True
		)
	val_loader = torch.utils.data.DataLoader(
		valset, batch_size=opt.testepisodeSize, shuffle=True, 
		num_workers=int(opt.workers), drop_last=True, pin_memory=True
		) 
	test_loader = torch.utils.data.DataLoader(
		testset, batch_size=opt.testepisodeSize, shuffle=True, 
		num_workers=int(opt.workers), drop_last=True, pin_memory=True
		) 


	# ============================================ Training ===========================================
	# Fix the parameters of Batch Normalization after 10000 episodes (1 epoch)
	if epoch_item < 1:
		model.train()
	else:
		model.eval()
		
	# Train for 10000 episodes in each epoch
	train(train_loader, model, criterion, optimizer, epoch_item, F_txt)


	# =========================================== Evaluation ==========================================
	print('============ Validation on the val set ============')
	print('============ validation on the val set ============', file=F_txt)
	prec1, _ = validate(val_loader, model, criterion, epoch_item, best_prec1, F_txt)


	# record the best prec@1 and save checkpoint
	is_best = prec1 > best_prec1
	best_prec1 = max(prec1, best_prec1)

	# save the checkpoint
	if is_best:
		save_checkpoint(
			{
				'epoch_index': epoch_item,
				'arch': opt.basemodel,
				'state_dict': model.state_dict(),
				'best_prec1': best_prec1,
				'optimizer' : optimizer.state_dict(),
			}, os.path.join(opt.outf, 'model_best.pth.tar'))


	if epoch_item % 10 == 0:
		filename = os.path.join(opt.outf, 'epoch_%d.pth.tar' %epoch_item)
		save_checkpoint(
		{
			'epoch_index': epoch_item,
			'arch': opt.basemodel,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
			'optimizer' : optimizer.state_dict(),
		}, filename)

	
	# Testing Prase
	print('============ Testing on the test set ============')
	print('============ Testing on the test set ============', file=F_txt)
	prec1, _ = validate(test_loader, model, criterion, epoch_item, best_prec1, F_txt)


F_txt.close()
print('............Training is end............')

# ============================================ Training End ==============================================================
