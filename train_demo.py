import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import math
import torch.nn.functional as F

import model
import config

torch.backends.cudnn.benchmark = True

def setup_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def calculate_sensitivity_specificity(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	sensitivity = tp / (tp + fn)
	specificity = tn / (tn + fp)
	return sensitivity, specificity

setup_seed(1)

cuda = config.cuda
print('cuda:',cuda)
if cuda != '-1':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = cuda

# data preparation phase and should be replaced by real-world training and testing data
train_total = 64
test_total = 16
slice_num = 10

train_data = torch.rand(train_total, slice_num, 116, 116)  # subjects * slices * node * node
train_data = train_data.view(-1, 116, 116)
train_data = train_data[torch.randperm(train_data.size(0))]  # shuffle train data
train_data = train_data.view(10, 64, 1, 116, 116)  # batch number * batch size * 1 * node * node

train_label = torch.randint(0, 2, (train_total,1))		# train label
train_label = train_label.expand(train_total, slice_num).reshape(10, 64)

test_data = torch.rand(test_total, slice_num, 116, 116)  # subjects * slices * node * node
test_data = test_data.unsqueeze(2)

test_label_s = torch.randint(0, 2, (test_total,1))		# test label
test_label_s = test_label_s.expand(test_total, slice_num)

criterion = nn.CrossEntropyLoss()

net = model.HM_AGPC(16, 16, 16, 64, 256)

net.apply(model.weights_init)
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.3fM" % (total/1e6))
if config.cuda != '-1':
	net = nn.DataParallel(net)
	net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00001,weight_decay=0)

acc_best= 0
starttime = time.time()

for epoch in range(60):
	train_correct = 0
	running_loss = 0.0

	net.train()
	for i in range(train_label_s.size(0)):
		inputs = train_data[i]
		labels = train_label[i]
		if para.cuda != '-1':
			inputs = inputs.cuda()
			labels = labels.cuda()
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()

		optimizer.step()
		# scheduler.step()
		running_loss += loss.item() / L_train * scale
		_, predicted = torch.max(outputs.data, 1)  # 这里是每一组取最大值之后，然后取最大值的下标
		train_total += labels.size(0)
		train_correct += (predicted == labels).sum().item()

	net.eval()
	correct_s, total_s = 0, 0
	prelist = []
	truelist = []
	test_loss = 0.0
	with torch.no_grad():
		for i in range(train_label_s.size(0)):
			inputs = train_data[i]
			labels = train_label[i]
			if config.cuda != '-1':
				inputs = inputs.cuda()
				labels = labels.cuda()
			output = net(inputs)
			loss = criterion(output, labels)
			test_loss += loss.item() / inputs.size()[0]
			
			output = F.softmax(output, dim=1)
			output = torch.mean(output, 0)
			_, predicted = torch.max(output.data, 0)
			prelist.append(predicted.cpu())
			truelist.append(labels_s.cpu().numpy()[0][0])
			if predicted == labels[0][0].item():
				correct_s += 1

	ltime = time.time()-starttime
	truelist, prelist = np.array(truelist).squeeze(), np.array(prelist).squeeze()
	val_f1 = f1_score(truelist, prelist, average='weighted')
	sensitivity, specificity = calculate_sensitivity_specificity(truelist, prelist)
	if (acc_best < correct_s / test_total):
		acc_best = correct_s / test_total
		print('Best acc')
		# torch.save(net.state_dict(), './save/' + config.modelname +'.pkl')
	print('[%d,%d,%d]loss:%.3f train acc:%.4f test acc:%.4f testacc_s:%.4f testf1_s:%.4f sen:%.4f spe:%.4f time:%.2fm' %
		(fold,I+1, epoch + 1, running_loss,train_correct/train_total,correct/total,correct_s/total_s,val_f1,sensitivity,specificity,ltime/60))

	if math.isnan(running_loss):
		print('break')
		break

print('Best_Acc:%.4f'%(acc_best))
