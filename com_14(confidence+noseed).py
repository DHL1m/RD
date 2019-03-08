from binary_14 import *
import scipy.io
# tf.set_random_seed(222)  #777
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("./data", one_hot=True)
data.test.cls = np.array([label.argmax() for label in data.test.labels])

myinit=0; mybin=1; mystochastic=0; mybias=0; mydropout=1; myupdate=0; myminus=1; myvar=1; myclip=0; mystd=1; mywrite=1


#최종 원하는 폴더 및 파일명: E:/TF_storage/20181117/Drift/workingtest/
#                         E:/TF_storage/20181117/Drift/work/
#                                                             .....N번 반복해서 평균편차 이용.......01~??/
#      event 파일(accuracy, histogram), accuracy, test output, test class, correctness, confusion matrix,
folder="E:/TF_Confidence/20190306/004_test/25/"
mycount=1
mytry = folder+'%02d' % mycount + '/'
for total_cycle in (1,):

    b={}
    for mybatch in (100,):  #100,50,24
        for mylr in (0.001,):   #0.0001,0.001,0.00001
            for myepochs in (45,):  #30,60
                count_drift=1
                for mydvalue in (1,):
                    if count_drift==1:
                        myfilename = mytry + 'conv.xlsx'
                    if count_drift==2:
                        myfilename = mytry + 'conf.xlsx'

                    for myt in (1,):
                        # b['correct_TF_vector_' + str(mycount)], b['cls_pred_'+str(mycount)], b['total_cm_'+str(mycount)] = BinaryNet('49com_seed232_repeat02' + mytry + '_', mycount, mybatch, mylr,
                        #                                          myepochs, myt, mydvalue, myinit, mybin, mystochastic,
                        #                                          mybias, mydropout, myupdate, myminus, myvar, myclip, mystd,
                        #                                          mywrite)

                        cls_true, cls_pred = BinaryNet(mytry, mycount, mybatch, mylr,
                                                                 myepochs, myt, mydvalue, myinit, mybin, mystochastic,
                                                                 mybias, mydropout, myupdate, myminus, myvar, myclip, mystd, mywrite, myfilename)

                        mycorrectness = [mytry + 'cls_true.xlsx', mytry + 'cls_pred.xlsx']
                        dfa = pd.DataFrame(cls_true)
                        dfa.to_excel(mycorrectness[0], index=False, header=False)
                        dfb = pd.DataFrame(cls_pred)
                        dfb.to_excel(mycorrectness[1], index=False, header=False)

                        mycount = mycount +1
                        mytry = folder + '%02d' % mycount + '/'
                    count_drift=count_drift+1


# dri0_morX
# c_1 = b['correct_TF_vector_1']
# c_2 = b['cls_pred_1']
# c_3 = b['total_cm_1']
# scipy.io.savemat('D:/BNNproject/BNN01/logs/total_cm_dri(conw).mat', {'c_3':c_3})
#
# d_1 = b['correct_TF_vector_2']
# d_2 = b['cls_pred_2']
# d_3 = b['total_cm_2']
# scipy.io.savemat('D:/BNNproject/BNN01/logs/total_cm_nor(seed).mat', {'d_3':d_3})



# images = data.test.images[c_1 & (~d_1)]
# # driX_nor0 = data.test.images[~correct_1&correct_2]


# cls_true = data.test.cls[c_1 & (~d_1)]
# logits_pred = d_2[c_1 & (~d_1)]
#
# cls_FF = data.test.cls[(~c_1) & (~d_1)]
#
# print('num of F(DRI)-F(NOR)=%d'%len(cls_FF))
# print('num of T(DRI)-F(NOR)=%d'%len(cls_true))

# plot_images(images=images[0:64],
#             cls_true=cls_true[0:64],
#             cls_pred=logits_pred[0:64])
#
# plt.show()
