import math
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as fdly
import datapro
import numpy as np
import random
from collections import namedtuple
from collections import Counter
from math import sqrt
import time
import six

class WiAU():
	def __init__(self, is_train=True):
		self.is_train = is_train

	def conv_bn_layer(self,
		input,
		num_filters,
		filter_size=5,
		stride=2,
		bias_attr=fluid.initializer.Normal(loc=0.0, scale=0.2),
		padding='SAME',
		act=None,
		name='CONV'):
#paddle.fdly.conv2d(input, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, use_mkldnn=False, act=None, name=None)
			#W=[,num_filters,filter_size, filter_size]
			if padding=='SAME':
				padding_size = num_filters//stride+1
			elif padding=='VALID':
				padding_size = (num_filters-filter_size+1)//stride+1
			conv = fdly.conv2d(
				input=input,
				num_filters=num_filters,
				filter_size=filter_size,
				stride=stride,
				padding=padding_size,
				groups=None,
				act=act,
				param_attr=None,
				bias_attr=bias_attr,
				name=name)
			return fdly.batch_norm(
				input=conv, act=act, is_test=not self.is_train)
	
	#class_dim: number of users in train
	def wiau_net(self,input,class_dim):
	
		LayerBlock = namedtuple('LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
		blocks = [LayerBlock(3, 128, 32),
				LayerBlock(3, 128, 32),
				LayerBlock(3, 256, 64),
				LayerBlock(3, 256, 64),
				LayerBlock(3, 512, 128),
				LayerBlock(3, 512, 128),
				LayerBlock(3, 1024, 256)]
		
		input_shape=fdly.shape(input)
		if fdly.array_length(input_shape) == 2:
			ndim=int(sqrt(input.shape[1]))
			if (ndim*ndim!=input.shape[1]):
				input,other=fdly.split(input,num_or_sections=[ndim*ndim,input.shape[1]-ndim*ndim],dim=1)
			input = fdly.reshape(input, [-1, ndim, ndim, 1],inplace=False)
              	
		#conv2.7×7,64
		conv=self.conv_bn_layer(
			input=input, 
			num_filters=64, 
			filter_size=7, 
			stride=2,
			bias_attr=fluid.initializer.Normal(loc=0.0, scale=0.2),
			padding='SAME',
			act='relu',
			name='conv2')

		# Max pool and downsampling 3*3
		pool_max=fdly.pool2d(
			input=conv,
			pool_size=3,
			pool_stride=2,
			pool_padding=1,
			pool_type='max')

		#conv3 first chain of resnets 1*1
		net=self.conv_bn_layer(
			input=pool_max,
			num_filters=blocks[0].num_filters,
			filter_size=1,
			stride=1,
			padding='VALID',
			name='conv3')
		
		for block_i, block in enumerate(blocks):
			for repeat_i in range(block.num_repeats):   #每一种卷积层的结构中进行循环，不同结构如上述blocks
				name = 'block_%d/repeat_%d' % (block_i, repeat_i)
				conv_in = self.conv_bn_layer(
					net, 
					block.bottleneck_size, 
					filter_size=1,
					padding='VALID', 
					stride=1,
					act='relu',
					name=name+'/conv_in')

				conv_bottleneck = self.conv_bn_layer(
					conv_in, 
					block.bottleneck_size, 
					filter_size=3,
					padding='SAME', 
					stride=1,
					act='relu',
					name=name+'/conv_bottleneck')

				conv_out = self.conv_bn_layer(
					conv_bottleneck, 
					block.num_filters, 
					filter_size=1,
					padding='VALID', 
					stride=1,
					act='relu',
					name=name+'/conv_out')

				#以上为在同种结构的卷积层中卷积三次	
				net=conv_out+net			#该步骤是将经历了以上同种结构卷积三次的结果，和以下跨不同结构卷积层的结构，以1：1的权重结合。
				#以下为在不同结构的卷积层中卷积一次
				
			try:
		        # upscale to the next block size
				next_block = blocks[block_i + 1]
				net = self.conv_bn_layer(
					net, 
					next_block.num_filters, 
					filter_size=1,
		            padding='SAME', 
		            stride=1, 
		            bias_attr=None,
		            name='block_%d/conv_upscale' % block_i)
			except IndexError:
				pass
		
		pool_avg=fdly.pool2d(
		    	input=net, 						
		    	pool_size=net.shape[1],
		    	pool_stride=1,
		    	pool_type='avg',
		    	global_pooling=True,
			name='pool_avg')

		reshaped=fluid.layers.reshape(
    			pool_avg, 
			shape=[-1, 
				pool_avg.shape[1] *
         			pool_avg.shape[2] *
         			pool_avg.shape[3]],
			inplace=1,
			name='reshape'
			)
            
		out = fdly.fc(input=reshaped,
			size=class_dim,
			act='softmax')
		
		return out

def train(use_cuda,user_ID,is_local=1):
	all_user_path="/home/elliott/eudemon_server/user/"
	self_save_path="/home/elliott/eudemon_server/user/"+str(user_ID)+"/model"

	place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
	
	data,label,labelOrder=datapro.train_data(all_user_path,user_ID)
	n_classes=label.shape[1]
	n_input=data.shape[1]

	#csi=fdly.data(name='csi_data',shape=data.shape,dtype='float32')
	#usr=fdly.data(name='user_ID',shape=label.shape,dtype='float32')

	csi=fdly.data(name='csi_data',shape=[-1,n_input],dtype='float32')
	usr=fdly.data(name='user_ID',shape=[-1,n_classes],dtype='float32')
	print(csi.shape)
	
	eudemon=WiAU()
	usr_pred=eudemon.wiau_net(input=csi,class_dim=n_classes)
	usr_pred=fdly.l2_normalize(usr_pred, axis=0, epsilon=1e-12, name=None)
	
	correct_prediction=fdly.equal(fdly.argmax(usr_pred, 1), fdly.argmax(usr, 1))
	accuracy=fdly.reduce_mean(fdly.cast(correct_prediction, 'float32'))

	cross_entropy=fdly.reduce_sum(usr*fdly.log(fdly.reciprocal(fdly.abs(usr_pred))))
	optimizer=fluid.optimizer.AdamOptimizer().minimize(cross_entropy)

	BATCH_SIZE=50
	num_passes=2000
	accuracy_all=[]
	valid_all=[]
	num=0
	num1=0
	
	exe=fluid.Executor(place)

	def train_valid_reader(user_ID,train_data=1):
		def reader():
			n_num=label.shape[0]

			listNum=list(range(n_num))
			random.shuffle(listNum)
			endNum=int(n_num*0.8)

			traindata=data[listNum[0:endNum]]
			validdata=data[listNum[endNum:n_num]]
			trainlabel=label[listNum[0:endNum]]
			validlabel=label[listNum[endNum:n_num]]
			maxlen = traindata.shape[1]
			if train_data==1:
				yield traindata,trainlabel
			elif train_data==0:
				yield validdata,validlabel
		return reader

	train_batch=paddle.batch(paddle.reader.shuffle(train_valid_reader(user_ID=user_ID,train_data=1), buf_size=500),batch_size=BATCH_SIZE)
	valid_batch=paddle.batch(paddle.reader.shuffle(train_valid_reader(user_ID=user_ID,train_data=0), buf_size=500),batch_size=BATCH_SIZE)
	print(train_batch)
	
	feeder = fluid.DataFeeder(feed_list=['csi_data','user_ID'],place=place)
	
	def train_loop(main_program):
		exe.run(fluid.default_startup_program())
		batch_id=0
		for pass_id in six.moves.xrange(num_passes):

			for datas in train_batch():
				print(datas)
				loss=exe.run(feed=feeder.feed(datas),fetch_list=['cross_entropy'])
				train_accuracy=exe.run(fetch_list=['optimizer','accuracy'],feed=feeder.feed(datas))[1]
				
				accuracy_all.append(train_accuracy)

			if pass_id%100==0:
				for datas in valid_batch:
					valid_accuracy=exe.run(fetch_list=['optimizer','accuracy'],feed=feeder.feed(datas))[1]
					valid_loss=exe.run(fetch_list=['cross_entropy'],feed=feeder.feed(datas))
					valid_all.append(valid_loss)

				print('the epoch',pass_id,'accuracy is',train_accuracy,'loss is',loss)
				print('the epoch',pass_id,'valid_accuracy is',valid_accuracy,'valid_loss is',valid_loss)
				
				if num1>0 and valid_loss < valid_all[num1-1]:
					fluid.io.save_inference_model(self_save_path+"/infer_model", ['n_input','csi_data','user_ID'], labelOrder, exe)										
				num1=num1+1
			if pass_id*batch_size>= len(traindata) and(train_accuracy)>=0.90 and (np.mean(valid_all[num-2:num])>=0.90):
				break
			num=num+1

	train_loop(fluid.default_main_program())

def infer(use_cuda, user_ID):
	user_test_path="/home/elliott/eudemon_server/user/"+str(user_ID)+"/use_dataset"
	self_save_path="/home/elliott/eudemon_server/user/"+str(user_ID)+"/model"+"/infer_model"
	place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
	exe = fluid.Executor(place)

	x_test=datapro.test(user_test_path, 4000)
	
	[inference_program, feed_target_names, fetch_targets]=fluid.io.load_inference_model(dirname=self_save_path, executor=exe)
	results = exe.run(inference_program,
				feed={
					feed_target_names[0]: x_test
				
				},fetch_list=fetch_targets)
	print(results[0].lod())
	np_data = np.array(results[0])
	print("Inference Shape: ", np_data.shape)


def main(use_cuda,user_ID,isTrain):
	all_user_path="/home/elliott/eudemon_server/user/"
	self_save_path="/home/elliott/eudemon_server/user/"+str(user_ID)+"/model"

	if isTrain==1:
		train(use_cuda,user_ID)
	else:
		infer(use_cuda,user_ID)

if __name__=='__main__'	:
	main(use_cuda=1,user_ID=4,isTrain=1)
