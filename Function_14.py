
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
tf.set_random_seed(3003)
class Model():
    def __init__(self, x, y_true, keep_prob, pre_Wbin, pre_Wfluc, layer_sizes, control,name):
        self.x = x
        self.y_true = y_true
        self.keep_prob = keep_prob
        self.pre_Wbin = pre_Wbin
        self.pre_Wfluc = pre_Wfluc
        self.layer_sizes = layer_sizes
        #    control=[mode_init,mode_bin,mode_stochastic,mode_bias,mode_dropout,mode_update]
        self.mode_init = control[0]  # 0:general initialization with xavier    0<x<=1:binary init with probs=x
        self.mode_bin, self.mode_stochastic = control[1], control[2]
        self.mode_bias = control[3]  # 0:no bias   1:use bias
        self.mode_dropout = control[4]  # equal the dropout rate in training phase, 1 means we don't use dropout
        self.mode_update = control[5]  # 0:general update with accumulating   1:binary update without accumulating
        self.mode_minus = control[6]
        self.mode_var = control[7]
        self.mode_clip = control[8]
        self.mode_std = control[9]
        self.mode_t = control[10]
        self.mode_dvalue = control[11]

        self.name = name

    def hard_sigmoid(self, x):
        return tf.clip_by_value((x + 1.) / 2., 0., 1.)
        # return tf.clip_by_value((x + 0.05)/0.1, 0., 1.)

    def binarization(self, W):
        num_inputs, num_outputs = [dim.value for dim in W.get_shape()]
        Wb = self.hard_sigmoid(W)  # 이거해도 텐서로 변하긴 하는데 실행은되네..?
        if self.mode_stochastic == 0:
            threshold = 0.5 if self.mode_minus else 0.75
            Wb = tf.cast(Wb > threshold, tf.float32)  # 부뜽호 없어야지 0->-1이 된다, 안그러면 0도 전부다 1이 되잖아..  tf.cast는 코스트함수를 만들기가 애매하긴하다,
            # Wb =tf.div(tf.subtract(Wb,0.501),tf.add(tf.subtract(Wb,0.5),0.0000001))
        else:
            Wb = tf.reshape(tf.contrib.distributions.Bernoulli(probs=Wb*0.5).sample(1), [num_inputs, num_outputs])       #probs=Wb
            Wb = tf.cast(Wb, tf.float32)
            # 근데 멀쩡한 웨이트를 매 턴마다 랜덤하게 이진화시키면..이거 핵 안좋은거 아닌가?
        Wb = ((Wb - 0.5) * 2.) if self.mode_minus else Wb
        #We can simplify this part through tf.sign()
        return Wb

    # def binarization(self, W):
    #     num_inputs, num_outputs = [dim.value for dim in W.get_shape()]
    #     Wa = self.hard_sigmoid(W)  # 이거해도 텐서로 변하긴 하는데 실행은되네..?
    #     Wb=Wa*0.0
    #     if self.mode_stochastic == 0:
    #         if self.mode_minus == 1:
    #             Wb = Wb + tf.cast(0.875 < Wa, tf.float32) * (0.25)
    #             Wb = Wb + tf.cast(0.75 < Wa, tf.float32) * (0.25)
    #             Wb = Wb + tf.cast(0.625 < Wa, tf.float32) * (0.25)
    #             Wb = Wb + tf.cast(0.5 < Wa, tf.float32) * (0.25)
    #             Wb = Wb + tf.cast(0.5 >= Wa, tf.float32) * (-0.25)
    #             Wb = Wb + tf.cast(0.375 > Wa, tf.float32) * (-0.25)
    #             Wb = Wb + tf.cast(0.25 > Wa, tf.float32) * (-0.25)
    #             Wb = Wb + tf.cast(0.125 >= Wa, tf.float32) * (-0.25)

                # Wb = Wb + tf.cast(0.75 < Wa, tf.float32) * (0.5)
                # Wb = Wb + tf.cast(0.5 < Wa, tf.float32) * (0.5)
                # Wb = Wb + tf.cast(0.5 >= Wa, tf.float32) * (-0.5)
                # Wb = Wb + tf.cast(0.25 > Wa, tf.float32) * (-0.5)


                # if 0.5 < Wb:
                #     if  Wb <= 0.75:
                #         Wb=tf.Constant(1.0)
                #     if  0.75 <= Wb:
                #         Wb=tf.Constant(2.0)
                # else:
                #     if 0.25 < Wb:
                #         Wb=tf.Constant(-1.0)
                #     if Wb <=0.25:
                #         Wb=tf.Constant(-2.0)


            # Wb = tf.cast(Wb > threshold, tf.float32)  # 부뜽호 없어야지 0->-1이 된다, 안그러면 0도 전부다 1이 되잖아..  tf.cast는 코스트함수를 만들기가 애매하긴하다,


            # Wb =tf.div(tf.subtract(Wb,0.501),tf.add(tf.subtract(Wb,0.5),0.0000001))
        # else:
        #     Wb = tf.reshape(tf.contrib.distributions.Bernoulli(probs=Wb*0.5).sample(1), [num_inputs, num_outputs])       #probs=Wb
        #     Wb = tf.cast(Wb, tf.float32)
        #     # 근데 멀쩡한 웨이트를 매 턴마다 랜덤하게 이진화시키면..이거 핵 안좋은거 아닌가?
        # Wb = ((Wb - 0.5) * 2.) if self.mode_minus else Wb
        # #We can simplify this part through tf.sign()
        # return Wb


    def makelayer(self, pre_Wfluc, pre_Wbin, input, num_outputs, uid, lastlayer):
        tf.set_random_seed(3003)
        np.random.seed(3003)
        batch_size, num_inputs = [dim.value for dim in input.get_shape()]
        with tf.variable_scope('binary_layer_%d' % uid):
            if self.mode_init == 0:
                W = tf.get_variable('w', shape=(num_inputs, num_outputs),
                                    initializer=tf.contrib.layers.xavier_initializer(seed=3003, dtype=tf.float32),
                                    trainable=True)
                tf.add_to_collection('mycheck0', W)
            else:
                Bernoulliset = tf.cast(tf.contrib.distributions.Bernoulli(probs=self.mode_init).sample([num_inputs, num_outputs]),
                                       tf.float32)
                Bernoulliset = ((Bernoulliset - 0.5) * 2) if self.mode_minus else Bernoulliset
                W = tf.Variable(Bernoulliset)

            # Reset_Meanmean, Reset_Meanstd = 0.*self.mode_std, 0.1707*self.mode_std
            # Reset_Stdmean, Reset_Stdstd = 0.0942*self.mode_std, 0.01884*self.mode_std
            # Set_Meanmean, Set_Meanstd = 0.*self.mode_std, 0.1538*self.mode_std
            # Set_Stdmean, Set_Stdstd = 0.1311*self.mode_std, 0.06894*self.mode_std

            Reset_Meanmean, Reset_Meanstd = 0.000001 * self.mode_std, 0.000001 * self.mode_std
            Reset_Stdmean, Reset_Stdstd = 0.000001 * self.mode_std, 0.000001 * self.mode_std
            Set_Meanmean, Set_Meanstd = 0.000001 * self.mode_std, 0.000001 * self.mode_std
            Set_Stdmean, Set_Stdstd = 0.000001 * self.mode_std, 0.000001 * self.mode_std
            #드리프트에 var 넣기1####################################################################
            drift_Meanmean, drift_Meanstd=0.09, 0.001
            drift_Stdmean, drift_Stdstd=0, 0.003
            ######################################################################################
            # Reset_Meanmean, Reset_Meanstd = 0.10000 * self.mode_std, 0.1 * self.mode_std
            # Reset_Stdmean, Reset_Stdstd = 0.00000 * self.mode_std, 0.05 * self.mode_std
            # Set_Meanmean, Set_Meanstd = 0.10000 * self.mode_std, 0.1 * self.mode_std
            # Set_Stdmean, Set_Stdstd = 0.000000 * self.mode_std, 0.05 * self.mode_std

            Reset_Meanvalue = tf.Variable(tf.random_normal(shape=[num_inputs, num_outputs],
                                                           mean=Reset_Meanmean,stddev=Reset_Meanstd,dtype=tf.float32,
                                                           name="Mean_Value_Reset"),trainable=False,name='Im1')
            # Reset_Meanvalue = tf.cast(np.random.normal(loc=Reset_Meanmean,scale=Reset_Meanstd,size=[num_inputs, num_outputs]),dtype=tf.float32)
            #Reset_Meanvalue
            Reset_Stdvalue = tf.cast(tf.Variable(((Reset_Meanvalue - Reset_Meanmean) / Reset_Meanstd) * Reset_Stdstd + Reset_Stdmean,trainable=False,name='Im2'),dtype=tf.float32)
            #Reset_Stdvalue=tf.random_normal(shape=[num_inputs, num_outputs],
                                                           # mean=Reset_Meanmean,stddev=Reset_Meanstd,dtype=tf.float32,
                                                           # name="Mean_Value_Reset")

            Set_Meanvalue = tf.Variable(tf.random_normal(shape=[num_inputs, num_outputs],
                                                         mean=Set_Meanmean, stddev=Set_Meanstd,dtype=tf.float32,
                                                         name="Mean_Value_Set"),trainable=False,name='Im3')
            # Set_Meanvalue = tf.cast(np.random.normal(loc=Set_Meanmean, scale=Set_Meanstd, size=[num_inputs, num_outputs]),dtype=tf.float32)
            Set_Stdvalue = tf.cast(tf.Variable(((Set_Meanvalue - Set_Meanmean) / Set_Meanstd) * Set_Stdstd + Set_Stdmean,trainable=False,name='Im4'),dtype=tf.float32)

            #Set_Stdvalue = tf.random_normal(shape=[num_inputs, num_outputs],
            #                                  mean=Reset_Meanmean, stddev=Reset_Meanstd, dtype=tf.float32,)
            # 드리프트에 var 넣기2##################################################################
            drift_Meanvalue = tf.Variable(tf.random_normal(shape=[num_inputs, num_outputs],
                                                           mean=drift_Meanmean, stddev=drift_Meanstd, dtype=tf.float32,
                                                           name="Mean_Value_drift"), trainable=False, name='Im5')
            drift_Stdvalue = tf.cast(tf.Variable(((drift_Meanvalue - drift_Meanmean) / drift_Meanstd) * drift_Stdstd +
                                                 drift_Stdmean, trainable=False, name='Im6'), dtype=tf.float32)
            drift_value = tf.Variable(tf.ones(shape=[num_inputs, num_outputs]) * 0.09)
            ######################################################################################
            tf.add_to_collection('mycheck2', drift_value)
        if self.mode_clip==1:
            K=tf.assign(W,tf.clip_by_value(W,-1.,1.))

            with tf.control_dependencies([K]):
                Wbin = self.binarization(W) if self.mode_bin else W
        else:
            Wbin = self.binarization(W) if self.mode_bin else W

        if self.mode_var==1:

            keep_element = tf.cast(tf.equal(pre_Wbin, Wbin), tf.float32)  # 전파트와 같은 원소들이 1
            update_element = tf.cast(~tf.equal(pre_Wbin, Wbin), tf.float32)  # 전파트와 다른 원소들이 1

            fluc_Reset = tf.reshape(tf.contrib.distributions.Normal(loc=Reset_Meanvalue, scale=Reset_Stdvalue).sample(1),
                                    [num_inputs, num_outputs])
            fluc_Set = tf.reshape(tf.contrib.distributions.Normal(loc=Set_Meanvalue, scale=Set_Stdvalue).sample(1),
                                    [num_inputs, num_outputs])
            Wfluc_Reset = update_element*fluc_Reset* tf.cast(tf.equal(Wbin, 1), tf.float32)
            Wfluc_Set = update_element*fluc_Set* tf.cast(tf.equal(Wbin, -1), tf.float32)

            # 1,-1 fluc state가 같다는 가정
            """        
            Meanmean, Meanstd = 0., 0.1
            Stdmean, Stdstd = 0., 0.2
            Meanvalue = tf.random_normal(shape=[num_inputs, num_outputs], mean=Meanmean, stddev=Meanstd,
                                         dtype=tf.float32, name="Mean_Value")
            Stdvalue = ((Meanvalue - Meanmean) / Meanstd) * Stdstd + Stdmean
            fluc = tf.reshape(tf.distributions.Normal(loc=Meanvalue, scale=Stdvalue).sample(1),
                              [num_inputs, num_outputs])
            #)tf.cast(tf.equal(update_element, 1), tf.float32)
            Wfluc1 = tf.multiply(tf.cast(tf.equal(update_element, 1), tf.float32), fluc)
            """
            # Wfluc_1 = tf.multiply(tf.cast(tf.equal(Wbin, -1), tf.float32), fluc)               # 1.00005

            # Wfluc = tf.cast(tf.equal(pre_Wbin, 1), tf.float32)*keep_element*pre_Wfluc*1.0001+\
            #         tf.cast(tf.equal(pre_Wbin, -1), tf.float32)*keep_element*pre_Wfluc*1.00000\
            #         + tf.multiply(Wbin, update_element) + Wfluc_Reset+Wfluc_Set

            batch_num = tf.Variable(1)
            batch_num_float = tf.cast(batch_num, dtype=tf.float32)
            # time_scale = tf.cast(self.mode_t, dtype=tf.float32)
            drift_factor = (1 + batch_num_float) / batch_num_float
            # drift_scale = tf.cond(tf.equal(batch_num, 1), lambda: tf.cast(tf.equal(batch_num, 1), dtype=tf.float32),
            #                       lambda: tf.cast(tf.pow((time_scale+(batch_num_float-1))/(time_scale+(batch_num_float - 2)), self.mode_dvalue),
            #                                       dtype=tf.float32))

            # 드리프트에 var 넣기3 ###################################################################
            new_drift=tf.cast((pre_Wbin<=0),dtype=tf.float32)*tf.cast((Wbin>0),dtype=tf.float32)
            # 전사이클에서는 -1이었고 이번 사이클에 1로 업데이트 된 element 부분을 1로
            new_value=tf.reshape(tf.contrib.distributions.Normal(loc=drift_Meanvalue, scale=drift_Stdvalue).sample(1),
                                    [num_inputs, num_outputs])
            # 전체의 random 값을 generate.
            assign_drift=tf.cond(tf.equal(batch_num, 1), lambda: drift_value.assign(new_value),
                                      lambda: drift_value.assign(new_value*new_drift+drift_value*(1-new_drift)))
            # # 원래 값에 assign을 해주는데 new_drift가 1인 파트는 새로운 값으로, 0인 파트는 기존의 값으로 assign
            # 딱 첫번째 사이클에만 첫번째 조건문 실행하면되는데 매번 condition체크하는게 비효율적, 이걸 어떻게 바꿀 수 있을까
            with tf.control_dependencies([assign_drift]):
                aaa=drift_value*tf.log(drift_factor)/tf.log(tf.constant(10, dtype=tf.float32))
                aaaa= tf.cast(drift_value*tf.log(drift_factor)/tf.log(tf.constant(10, dtype=tf.float32)), dtype=tf.float32)
                if self.mode_dvalue==0:
                    drift_scale=tf.constant(0.)
                else:
                    drift_scale = tf.cond(tf.equal(batch_num, 1), lambda: tf.zeros(shape=[num_inputs,num_outputs],dtype=tf.float32),
                                          lambda: tf.cast(drift_value*tf.log(drift_factor)/tf.log(tf.constant(10, dtype=tf.float32)), dtype=tf.float32))
            #여기가 0이 맞는지 확인필요
            #######################################################################################
            # drift_scale = tf.cond(tf.equal(batch_num, 1), lambda: tf.cast(1/time_scale, dtype=tf.float32),
            #                       lambda: tf.cast(tf.pow(batch_num_float / (batch_num_float - 1.0), self.mode_dvalue),
            #                                       dtype=tf.float32))
            # drift_scale = tf.cond(tf.equal(batch_num, 1), lambda: tf.cast(1.0/time_scale, dtype=tf.float32), lambda: tf.cast(tf.pow(batch_num_float / ((time_scale) * (batch_num_float - 1)), self.mode_dvalue),dtype=tf.float32))

            with tf.control_dependencies([drift_scale]):
                # Wfluc = tf.cast(tf.equal(pre_Wbin, 1), tf.float32) * keep_element * pre_Wfluc * drift_scale \
                #         + tf.cast(tf.equal(pre_Wbin, -1), tf.float32) * keep_element * pre_Wfluc * 1.000000 \
                #         + tf.multiply(Wbin, update_element) + Wfluc_Reset + Wfluc_Set
                # Wfluc = (tf.cast(tf.equal(pre_Wbin, 1), tf.float32))*keep_element*(pre_Wfluc) \
                #         + tf.cast(tf.equal(pre_Wbin, -1), tf.float32) * keep_element * (pre_Wfluc) \
                #         + tf.multiply(Wbin, update_element) + Wfluc_Reset + Wfluc_Set
                bbb=pre_Wfluc + drift_scale
                Wfluc = (tf.cast(tf.equal(pre_Wbin, 1), tf.float32)) * keep_element * (pre_Wfluc + drift_scale) \
                        + tf.cast(tf.equal(pre_Wbin, -1), tf.float32) * keep_element * (pre_Wfluc) \
                        + tf.multiply(Wbin, update_element) + Wfluc_Reset + Wfluc_Set
                tf.add_to_collection('mycheck1', Wbin)

        else:

            if self.mode_bin == 1:
                Wfluc = Wbin
            else:
                Wfluc = W
            b = tf.Variable(tf.random_normal([num_outputs]))
        if self.mode_bias == 1:
            logit = tf.matmul(input, Wfluc) + b
        else:
            if self.mode_var == 1:
                batch_count = tf.assign(batch_num, 1 + batch_num)
                with tf.control_dependencies([batch_count]):
                    logit = tf.matmul(input, Wfluc)
            else:
                if self.mode_bin == 1:
                    logit = tf.matmul(input, Wbin)
                else:
                    logit = tf.matmul(input, W)
        if lastlayer == 1:
            return logit, (logit, input, W, Wbin, Wfluc)
        else: #tf.nn.relu(logit)
            output=tf.nn.dropout(tf.nn.relu(logit), keep_prob=self.keep_prob)
            return output, (output, input, W, Wbin, Wfluc)

    def build_model(self):
        updates = []
        hidden_layer = self.x
        for i, num_hidden in enumerate(self.layer_sizes):
            hidden_layer, update = self.makelayer(self.pre_Wfluc[i], self.pre_Wbin[i], hidden_layer, num_hidden, i,
                                                  lastlayer=(i == len(self.layer_sizes) - 1))
            updates.append(update)
        output_layer = hidden_layer
        return output_layer, updates




def makedataset(data, i, j, name):
    setname = data.test if name == 'test' else data.train
    temp = (np.argmax(setname.labels, 1) == i)  #
    zeroset, zerosetindex = setname.images[temp], setname.labels[temp]
    temp = (np.argmax(setname.labels, 1) == j)
    oneset, onesetindex = setname.images[temp], setname.labels[temp]
    trainset = np.append(zeroset, oneset, 0)
    trainindex = np.append(zerosetindex, onesetindex, 0)
    # Calculate frequency of each number in this data set:
    num = []
    for i in range(10):
        temp1 = (np.argmax(setname.labels, 1) == i)
        temp2 = setname.labels[temp1]
        frequency = len(temp2) / len(temp1)
        num.append(frequency)
        print(str(i), ":", frequency)
    return trainset, trainindex


def binary_backprop(loss, output, updates):
    backprop_updates = []
    loss_grad, = tf.gradients(loss, output)
    # (p, prev_layer, w, bin_w)  binary_backprop(loss, output_layer, updates)
    for layerout, prev_layer, w, bin_w, wfluc in updates[::-1]:
        w_grad, loss_grad = tf.gradients(layerout, [wfluc, prev_layer], loss_grad)
        backprop_updates.append((w_grad, w))

    return backprop_updates


def get_loss(hypothesis, y_true):
    #temp=tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y_true)
    #index=temp>tf.nn.softmax_cross_entropy_with_logits(logits=tf.constant([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]),labels=[1,0,0,0,0,0,0,0,0,0])
    #temp=temp*tf.cast(index,tf.float32)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y_true))


def get_accuracy(hypothesis, y_true):
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###################################################################################
# def print_confusion_matrix (hypothesis, y_true):
#     cls_pred = tf.argmax(hypothesis,1)
#     cls_true = tf.argmax(y_true,1)
#     cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
#     print(cm)
#     # Plot the confusion matrix as an image.
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#
#     # Make various adjustments to the plot.
#     plt.tight_layout()
#     plt.colorbar()
#     tick_marks = np.arange(num_classes)
#     plt.xticks(tick_marks, range(num_classes))
#     plt.yticks(tick_marks, range(num_classes))
#     plt.xlabel('Predicted')
#     plt.ylabel('True')

1

def make_feeddict(updates=0,pre_Wbin=0,pre_Wfluc=0,feed_dict=0,save=0,mode=0):
    if mode==0:
        temp = {}
        for i in range(len(pre_Wfluc)):
            num_inputs, num_outputs = [dim.value for dim in pre_Wfluc[i].get_shape()]
            temp[pre_Wbin[i]],temp[pre_Wfluc[i]]=np.zeros([num_inputs, num_outputs]),np.zeros([num_inputs, num_outputs])
        return temp
    elif mode==1:
        temp=[]
        for i in range(len(pre_Wfluc)):
            temp.append(updates[i][3])
            temp.append(updates[i][4])
        return temp
    elif mode==2:
        for i in range(len(pre_Wfluc)):
            feed_dict[pre_Wbin[i]],feed_dict[pre_Wfluc[i]]=save[i*2:(i+1)*2]
        return feed_dict


# We know that MNIST images are 28 pixels in each dimension.
img_size = 28
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 64

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(8, 8)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

