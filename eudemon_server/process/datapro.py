#!usr/bin/env python3
# -*- coding: UTF-8 -*-
#代码用来切分数据和进行预处理

import numpy as np
import os
import pickle
import cmath
import math
import scipy.stats as scita
import scipy.signal as signal
import scipy.io as scio
from scipy.spatial.distance import euclidean
from numpy import linalg as la

def db(x):
#变换到log空间
	if (x==0):
		ans=0
	else:
		ans=20*(np.log10(x))
	return ans

def butterworth_II(file,Fc):
#去噪
	N  = 2    # Filter order
	Wn = 2*(np.pi)*Fc # Cutoff frequency
	B, A = signal.butter(N, Wn, output='ba')
	ret = signal.filtfilt(B,A,file)
	return ret

def relative_phase(tmp1,tmp2):
#计算相对相位
	tmp=tmp1*np.conjugate(tmp2)
	tmp_1=(tmp.real)/(abs(tmp))
	ret=np.arccos(tmp_1)

	return (ret)


def file_data(filename):
#解析原始数据包
	l=[]
	#print (filename)
	with open(filename, 'rb+') as f:
		while 1:
			try:
				flag = 1
				k = pickle.load(f)
				for i in range(90):
					if(abs(k[i]) <= 0):
						flag = 0
				if flag == 0:
					continue
				l.append(k[0:90])
			except Exception as e:
				#print('read end')
				break
	a = np.array(l).T
	f.close()
	return(a)

def csi_amplitude(file):
	#print(file.shape)
	[row,col]=file.shape
	newFile = np.zeros((row,col))
	for i in range(row):
		for j in range(col):
			newFile[i,j]=db(abs(file[i,j]))
			#print (file[i,j])
	ret=np.array(newFile)
	return (ret.real)

def csi_relative_phase(file):
#计算数据的相对相位
	file=file.reshape(3,30,-1)
	[row,col,other]=file.shape
	csi_ant1=file[0]
	csi_ant2=file[1]
	csi_ant3=file[2]
	rephase1_2=[]
	rephase1_3=[]
	rephase3_2=[]
	rephase_all=[]

	for i in range(col):
		tmp1_2=relative_phase(csi_ant1[i],csi_ant2[i])

		tmp1_3=relative_phase(csi_ant1[i],csi_ant3[i])
		tmp3_2=relative_phase(csi_ant3[i],csi_ant2[i])
		rephase1_2.append(tmp1_2)
		rephase1_3.append(tmp1_3)
		rephase3_2.append(tmp3_2)

	for j in range(30):
		rephase_all.append(rephase1_2[j])
	for j in range(30):
		rephase_all.append(rephase1_3[j])
	for j in range(30):
		rephase_all.append(rephase3_2[j])

	ret=np.array(rephase_all)
	return ret.real

def get_characters(matrix, num, label):
#计算出相应特征值
	max = []
	min = []
	mean = []
	skewness = []
	kurtosis = []
	std = []
	i = 1
	col = matrix.shape[1]
	chunk = int(col / num)
	while (i) * chunk <= col and i <= num:
		tmp = matrix[:, chunk * (i-1):chunk * (i)]
		i = i + 1
		cnt = 0
		max_t = []
		min_t = []
		mean_t = []
		skewness_t = []
		kurtosis_t = []
		std_t = []
		while cnt < 90:
			t = tmp[cnt:]
			max_t.append(np.max(t))
			min_t.append(np.min(t))
			mean_t.append(np.mean(t))
			skewness_t.append(np.mean(scita.skew(t,axis=1, bias=True)))
			kurtosis_t.append(np.mean(scita.kurtosis(t,axis=1, bias=True)))
			std_t.append(np.std(t))
			cnt = cnt + 1
		max.append(max_t)
		min.append(min_t)
		mean.append(mean_t)
		skewness.append(skewness_t)
		kurtosis.append(kurtosis_t)
		std.append(std_t)
	l = []
	for i in range(0, 540):
		l.append(label)
	max = np.array(max).T
	min = np.array(min).T
	mean = np.array(mean).T
	skewness = np.array(skewness).T
	kurtosis = np.array(kurtosis).T
	std = np.array(std).T
	result = np.append(max, min, axis=0)
	result = np.append(result, mean, axis=0)
	result = np.append(result, skewness, axis=0)
	result = np.append(result, kurtosis, axis=0)
	result = np.append(result, std, axis=0)
	return result, l
	
def frontMad(input):
	csiMatrix=[]
	ans=input
	for i in range(input.shape[0]):
		csiList=[]
		for j in range(0,input.shape[1],3):	
			csiStd=input[i][j]-np.mean(input[i][j:j+2])
			csiList.append(csiStd)
		csiMatrix.append(csiList)
	ans=np.array(csiMatrix)
	return ans
	
def densityDetect(input,step):
	input=np.abs(input)
	ansLen=[]
	ansStd=[]
	tmp1=[]
	tmp=[]
	orderList=[]
	tmpList=np.argsort(input)
	tt=0
	j=0
	while((np.min(input)+(tt+1)*step)<=np.max(input)):	
		while ((input[tmpList[j]]>(np.min(input)+(tt)*step)) and (input[tmpList[j]]<=(np.min(input)+(tt+1)*step)) or input[tmpList[j]]==np.min(input) ) :
			tmp.append(input[tmpList[j]])
			tmp1.append(tmpList[j])
			j=j+1
		ansLen.append(len(tmp))
		orderList.append(tmp1)
		ansStd.append(np.std(tmp))
		tmp=[]
		tmp1=[]
		tt=tt+1
	return ansLen,orderList,ansStd


def classification(input):
#根据标签list转化为标签矩阵（one-hot编码）
	labelOrder=set()
	labelList=[]
	i=0
	for label in input:
		label=label.split('_')[0]
		labelList.append(label)
		labelOrder.add(label)
	labelOrder=list(labelOrder)
	labelCol=len(labelOrder)
	labelMatrix = []
	for label in labelList:
		l = [0] * labelCol
		labelNum = labelOrder.index(label)
		l[labelNum] = 1
		labelMatrix.append(l)
	labelMatrix = np.array(labelMatrix, dtype='float32')
	return labelMatrix,labelOrder
					
def srAlgorithm(input):
## 循环找动态变量和面积最大量
	step=0.05
	step1=0.01
	finaList=[]
	ansList,orderList,ansStd=densityDetect(input,step)
	if (len(ansList)>1):
		while (((ansList[0])!=np.max(ansList)) or (ansStd[0])!=np.min(ansStd) ):
			step=step+step1
			ansList,orderList,ansStd=densityDetect(input,step)
	else:
		while ((ansStd[0])!=np.min(ansStd)):
			step=step-step1
			ansList,orderList,ansStd=densityDetect(input,step)
	if len(orderList)>=1:
		finaNum=(orderList[0])
	else:
		finaNum=orderList
	return finaNum,step
	
def readmat(path):
	anslabel=[]
	ansdata=[]
	a=scio.loadmat(path)
	data=a['csi']
	return data

def active(path,plotNum):
	file= os.listdir(path)
	labelList=[]
	labelFina=[]
	choseMatrix=[]
	for files in file:
		if files:	
			# print(files)	
			if (files.split('.'))[1]=='mat':
				csidata=file_data(path+'/'+files)
				csiAmplitude=butterworth_II(csi_amplitude(csidata),0.03)	
			csiStdmatrix=frontMad(csiAmplitude)
			numList=[]
			for jj in plotNum:
				finaNum,step=srAlgorithm(csiStdmatrix[jj])
				numList=list(set(numList)^set(finaNum))
			numList=3*np.sort(numList)
			csiAmplitude[:,numList]=0
			choseMatrix.extend((csiAmplitude).tolist())
			label=90*[files]
			labelFina.extend(label)	
	labelMatrix,labelOrder=classification(labelFina)
	choseMatrix=np.array(choseMatrix, dtype=object)
	return choseMatrix,labelMatrix,labelOrder
	
def pad_sequences(sequences, maxlen=None, dtype='int32',padding='pre', truncating='pre', value=0.):
    lengths = []
    for x in sequences:
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def train_data(path,ID):
	plotNum=[1,8,20]
	files= os.listdir(path)
	labelList=[]
	labelFina=[]
	choseMatrix=[]
	for file in files:
		if int(file) not in get_related_ID(path,ID).tolist():
			continue
		filepath=os.path.join(path,file,'train_dataset')
		print(filepath)
		t=os.listdir(filepath)
		for files in t:
			if files:	
				print(files)	
				if len(files.split('.'))>1 and (files.split('.'))[1]=='mat':
					csidata=readmat(filepath+'/'+files)
					print(csidata.shape)
					csiAmplitude=butterworth_II(csi_amplitude(csidata),0.03)
				else:
					csidata=file_data(filepath+'/'+files)
					csiAmplitude=butterworth_II(csi_amplitude(csidata),0.03)	
				csiStdmatrix=frontMad(csiAmplitude)
				numList=[]
				for jj in plotNum:
					finaNum,step=srAlgorithm(csiStdmatrix[jj])
					numList=list(set(numList)^set(finaNum))
				numList=3*np.sort(numList)
				csiAmplitude[:,numList]=0
				choseMatrix.extend((csiAmplitude).tolist())
				label=90*[file]
				labelFina.extend(label)	
	labelMatrix,labelOrder=classification(labelFina)
	result=pad_sequences(choseMatrix, maxlen=None, dtype='float32',padding='post', truncating='pre', value=0.)
	return result,labelMatrix,labelOrder

def test(path,maxlen):
	'the input is the path of file and the maxlen denotes the maximize length of the traindata'
	plotNum=[1,8,20]
	files= os.listdir(path)
	choseMatrix=[]
	for file in files:
		print(file)
		csidata=file_data(path+'/'+file)
		csiAmplitude=butterworth_II(csi_amplitude(csidata),0.03)	
		csiStdmatrix=frontMad(csiAmplitude)
		numList=[]
		for jj in plotNum:
			finaNum,step=srAlgorithm(csiStdmatrix[jj])
			numList=list(set(numList)^set(finaNum))
		numList=3*np.sort(numList)
		csiAmplitude[:,numList]=0
		choseMatrix.extend((csiAmplitude).tolist())
	result=pad_sequences(choseMatrix, maxlen=maxlen, dtype='float32',padding='post', truncating='pre', value=0.)
	return result
	
def get_related_ID(path,ID):
	filepath=os.path.join(path , str(ID)+'/'+str(ID)+'-info.npz')
	usr_data=np.load(filepath)
	#print(usr_data["arr_3"])
	return usr_data["arr_3"]

if __name__=='__main__'	:
	path="/home/elliott/eudemon_server/user"
	data,label,labelOrder=train_data(path,4)
	#print(data.shape)

		



	
	

	
	
	
	
	
	
