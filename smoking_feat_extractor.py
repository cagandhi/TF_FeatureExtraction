# Usage: python smoking_feat_extractor --network NETWORK_NAME --folder FOLDER_NAME

# import libraries
import h5py
import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
from yellowbrick.classifier import ClassPredictionError, ClassificationReport, ROCAUC, PrecisionRecallCurve

# start timing
start = time.time()

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--network", dest="network_name", type=str, required=True, help='network name can be one of inception_v4, inception_resnet_v2, resnet_v1_152, resnet_v2_152')
parser.add_argument('--folder', dest='folder_name', type=str, required=True, help='folder name prefix: eg. smoking, smoking_b4_hmdb_added. Please enter exact folder name.')
args = parser.parse_args()

# store network name provided while running command
network_name = args.network_name
folder_name = args.folder_name

# create a dictionary of checkpoints for dynamic selection
checkpointDict = {'inception_v4': 'inception_v4.ckpt', 'inception_resnet_v2': 'inception_resnet_v2.ckpt', 'resnet_v1_152': 'resnet_v1_152.ckpt', 'resnet_v2_152': 'resnet_v2_152.ckpt'}

# this string will be appended to command later on
append = ''

# specify from which layer you want the output; change only after looking at code on https://github.com/tensorflow/models/tree/master/research/slim
if network_name == 'inception_v4':
    append = '--layer_names PreLogitsFlatten'

elif network_name == 'inception_resnet_v2': 
    append = '--layer_names PreLogitsFlatten --preproc_func inception'

elif network_name == 'resnet_v1_152':
    append = '--layer_names global_pool'

elif network_name == 'resnet_v2_152':
    append = '--layer_names global_pool --preproc_func inception'

# create folder to store features
cmd = 'mkdir smoking_features'
os.system(cmd)
cmd = 'mkdir smoking_features/' + network_name
os.system(cmd)

# initialize train and test feature matrices
x_train = []
x_test = []

y_train = []
y_test = []

# loop over class folders; change this when you change dataset
for i in range(2):
    # print(classNm)

    # --- training part --- command to create a h5py file of features of training images
    cmd = 'python3 example_feat_extract.py --network '+network_name+' --checkpoint ./checkpoints/'+checkpointDict[network_name]+' --image_path ./'+folder_name+'/train/'+str(i)+'/ --out_file ./smoking_features/'+network_name+'/train_'+str(i)+'.h5 '
    cmd += append

    # os.system(cmd)
    
    # open features file for reading
    filename = './smoking_features/'+network_name+'/train_'+str(i)+'.h5'
    data = h5py.File(filename, 'r')

    # extract feature vector and append to training feature matrix; according to code of these networks provided in TF-Slim library
    if 'inception' in network_name:    
        for key in data.keys():
            if key == 'PreLogitsFlatten':
                group = data[key]
                samples = group[:,:].shape[0]
                labels = np.array([i for j in range(samples)])

                x_train.extend(group[:,:])
                y_train.extend(labels)
    
    elif network_name == 'resnet_v1_152' or network_name == 'resnet_v2_152':
        for key in data.keys():
            if key == 'global_pool':
                group = data[key]
                samples = group[:,:].shape[0]
                labels = np.array([i for j in range(samples)])

                x_train.extend(group[:,:])
                y_train.extend(labels)
    data.close()

    # ------------------------------------------------------------------------------------------------------
    
    # --- testing part --- command to create a h5py file of features of testing images
    cmd = 'python3 example_feat_extract.py --network '+network_name+' --checkpoint ./checkpoints/'+checkpointDict[network_name]+' --image_path ./'+folder_name+'/test/'+str(i)+'/ --out_file ./smoking_features/'+network_name+'/test_'+str(i)+'.h5 '
    cmd += append
    
    # os.system(cmd)

    # construct test feature matrix
    # open features file for reading
    filename = './smoking_features/'+network_name+'/test_'+str(i)+'.h5'
    data = h5py.File(filename, 'r')

    # extract feature vector and append to training feature matrix; according to code of these networks provided in TF-Slim library
    if 'inception' in network_name:    
        for key in data.keys():
            if key == 'PreLogitsFlatten':
                group = data[key]
                samples = group[:,:].shape[0]
                labels = np.array([i for j in range(samples)])

                x_test.extend(group[:,:])
                y_test.extend(labels)
    
    elif network_name == 'resnet_v1_152' or network_name == 'resnet_v2_152':
        for key in data.keys():
            if key == 'global_pool':
                group = data[key]
                samples = group[:,:].shape[0]
                labels = np.array([i for j in range(samples)])

                x_test.extend(group[:,:])
                y_test.extend(labels)
    data.close()

# transform into numpy arrays for better usage
x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

print('printing shapes')
print('X test shape:')
print(x_test.shape)
print('y test shape: ')
print(y_test.shape)

# create LinearSVC classifier, fit to training data and predict labels for test data
C = -1; # for C in range(-3,3):
clf = svm.LinearSVC(C=0.1) # clf = svm.SVC(C=10000,kernel='poly',degree=5) #
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
y_pred_train = clf.predict(x_train)

print('y pred shape:')
print(y_pred.shape)

# score the classifier
print('Classifier score: ', clf.score(x_test,y_test))
# precision
print('Precision: ', precision_score(y_test,y_pred))
# recall
print('Recall: ', recall_score(y_test,y_pred))

# confusion matrix to check for misclassifications
#          Predicted
#            0   1
# Actual 0  TN  FP
#        1  FN  TP
cm = confusion_matrix(y_test, y_pred) 
print(cm)
tn,fp,fn,tp = cm.ravel()
print('TN-',tn,' ; FP-',fp,'\nFN-',fn,' ; TP-',tp)

# end timing
end = time.time()
print('\nExecution time in minutes: ', (end-start)/60)

# L2 loss - test
loss = 0
lossvec = []
for i in range(len(y_test)):
    val=(y_test[i]-y_pred[i])**2
    loss+=val
    lossvec.append(loss/(i+1))

print('Test Error: '+str(loss))
print('Test Error vector: ')
print(len(lossvec))
# print(lossvec)

# print test loss curve
xaxis = [i for i in range(len(lossvec))]
plt.plot(xaxis, lossvec) # plt.plot(xaxis, lossvec, '.')
plt.xlabel('Testing examples')
plt.ylabel('Classifier Loss')
plt.title('Test loss vs no. of testing examples')
# plt.legend(['Train score','Test score'])
plt.savefig('test_loss.png')
plt.close()

# L2 loss - train
loss = 0
lossvec = []
for i in range(len(y_train)):
    val=(y_train[i]-y_pred_train[i])**2
    loss+=val
    lossvec.append(loss/(i+1))

print('Train Error: '+str(loss))
print('Train Error vector: ')
print(len(lossvec))
# print(lossvec)

# print train loss curve
xaxis = [i for i in range(len(lossvec))]
plt.plot(xaxis, lossvec, '.')
plt.xlabel('Training examples')
plt.ylabel('Classifier Loss')
plt.title('Training loss vs no. of training examples')
# plt.legend(['Train score','Test score'])
plt.savefig('training_loss.png')
plt.close()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test,y_pred)
print('\nfpr shape:')
print(fpr)
print('tpr shape:')
print(tpr)
print('AUC:')
print(auc)
print()

# TRAIN TEST ERROR CURVE
train_sizes = [10,100,250,450,575]
testvec=[]
trainvec=[]
x_train, y_train = shuffle(x_train, y_train)

for ts in train_sizes:
    # x_new, y_new = shuffle(x_train, y_train)
    
    x_new = x_train[:ts,:] 
    # x_new = x_new[:ts,:]
    y_new = y_train[:ts] 
    # y_new = y_new[:ts]

    clf1 = svm.LinearSVC(C=0.1)
    clf1.fit(x_new,y_new)
    
    # test error
    y_ptest = clf1.predict(x_test)
    loss = 0
    for i in range(len(y_test)):
        val=(y_ptest[i]-y_test[i])**2
        loss+=val
    loss/=len(y_test)
    testvec.append(loss)

    # train error
    y_ptrain = clf1.predict(x_new)
    loss=0
    for i in range(len(y_new)):
        val=(y_ptrain[i]-y_new[i])**2
        loss+=val
    loss/=len(y_new)
    trainvec.append(loss)

print(testvec)
print(trainvec)

plt.plot(train_sizes, testvec)
plt.plot(train_sizes, trainvec, 'r')
plt.xlabel('No. of training examples')
plt.ylabel('Error')
plt.title('Train-Test error vs no. of training examples')
plt.legend(['Test error','Train error'])
plt.savefig('train_test_error.png')
plt.close()

# TRAIN TEST SCORE METRIC
testvec=[]
trainvec=[]
for ts in train_sizes:
    # x_new, y_new = shuffle(x_train, y_train)
    
    x_new = x_train[:ts,:] 
    # x_new = x_new[:ts,:]
    y_new = y_train[:ts] 
    # y_new = y_new[:ts]

    clf1 = svm.LinearSVC(C=0.1)
    clf1.fit(x_new,y_new)
    
    testvec.append(clf1.score(x_test,y_test))
    trainvec.append(clf1.score(x_new,y_new))

print(testvec)
print(trainvec)

plt.plot(train_sizes, testvec)
plt.plot(train_sizes, trainvec, 'r')
plt.xlabel('No. of training examples')
plt.ylabel('Score')
plt.title('Train-Test score vs no. of training examples')
plt.legend(['Test score','Train score'])
plt.savefig('train_test_score.png')
plt.close()

'''
# CHECKING FOR OUTPUT ; MAP FILENAME TO PREDICTED LABEL
# write y_pred to file

fin = open('testvec.txt','r')
filelist = fin.readlines()
newl=[]
for f in filelist:
    if f not in newl:
        newl.append(f)

print(y_pred)
fout = open('pred.txt','w')
i=0
fout.write('Filename ; Predicted ; Actual\n')
for line in newl:
    str1 = line.split('/')[-1][:-1]+' ; '+str(y_pred[i])+' ; '+str(y_test[i])
    fout.write(str1+'\n')
    i+=1

fout.close()
fin.close()
'''