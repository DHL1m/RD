from Function_14 import *
import tensorflow as tf
import numpy as np
import pandas as pd

# import os
from tensorflow.contrib.tensorboard.plugins import projector
# binary_2.py 에서 정확도의 차이와 관련되는 image 를 추출하는 과정을 추가함

def BinaryNet(mytry,mycount, mybatch,mylr,myepochs,myt, mydvalue,myinit,mybin,mystochastic,mybias,mydropout,myupdate,myminus,myvar,myclip,mystd,mywrite,myfilename):
    tf.set_random_seed(3003)  # 777
    with tf.name_scope('Load_Data'):
        from tensorflow.examples.tutorials.mnist import input_data
        data = input_data.read_data_sets("./data", one_hot=True)

        # b=[]
        # for iii in range(0, 10000):
        #     for a in range(70):
        #         b = b + [np.random.randint(784)]
        #     for i in b:
        #         data.test.images[iii, i] = data.test.images[iii, i] + np.random.rand() * 0.5
        #     del b[:]
        # for iii in range(0, 10000):
        #     for i in range(393,784):
        #         # if data.test.images[iii, i] >:
        #         data.test.images[iii, i] = 0
        #         # else:
                #     data.train.images[iii, i] = 1

        num_classes = len(data.test.labels[2,:])
        assert len(data.train.labels[1,:])==len(data.validation.labels[0,:])==num_classes
        img_size_flat=len(data.test.images[0,:])
        assert len(data.validation.images[1,:])==len(data.train.images[2,:])==img_size_flat


####################################################################################

        # Number of classes, one class for each of 10 digits.
        num_classes = 10
        data.test.cls = np.array([label.argmax() for label in data.test.labels])

        # Get the first images from the test-set.
        # images = data.test.images[0:9]
        # # Get the true classes for those images.
        # cls_true = data.test.cls[0:9]
        # # Plot the images and labels using our helper-function above.
        # plot_images(images=images, cls_true=cls_true)
        # plt.show()
###################################################################################

    with tf.name_scope('Hyper_Parameters'):
        # learning_rate = mylr   #BEST=0.0001
        learning_rate = tf.get_variable(dtype=tf.float32,name="LR", shape=[], initializer=tf.zeros_initializer())
        time_warp = tf.get_variable(dtype=tf.float32, name="TW", shape=[], initializer=tf.zeros_initializer())
        training_epochs = myepochs     #15  // 30, 60, 120
        batch_size = mybatch #100
        layer_sizes=[256,10]     #layer_sizes=[256,10]

    with tf.name_scope('Control_Parameters'):
        mode_test, ii, jj = 0, 4, 9
        mode_init = myinit  # 0:general initialization with xavier    0<x<=1:binary init with probs=x #0.7
        mode_bin, mode_stochastic = mybin, mystochastic
        mode_bias = mybias  # 0:no bias   1:use bias
        mode_dropout = mydropout # equal the dropout rate in training phase, 1 means we don't use dropout  #0.7
        mode_update = myupdate  # 0:general update with accumulating   1:binary update without accumulating
        mode_minus = myminus
        mode_var = myvar
        mode_clip = myclip
        mode_std = mystd #1
        mode_t = myt
        mode_dvalue = mydvalue
        control = [mode_init,mode_bin,mode_stochastic,mode_bias,mode_dropout,mode_update,mode_minus,mode_var,mode_clip,mode_std,mode_t, mode_dvalue]
        mode_write = mywrite

        #We make train and test set:
        if mode_test == 0:
            trainset = data.train.images
            trainindex = data.train.labels
            testset = data.test.images
            testindex = data.test.labels
        else:
            trainset, trainindex = makedataset(data, ii, jj, 'train')
            testset, testindex = makedataset(data, ii, jj, 'test')

    with tf.name_scope('Input_Data'):   #Model클래스에 인풋으로 들어가게 될 데이터들이다.모델클래스는 이 데이터들이 잘들어온다는 가정 하에 설계되었다.
        x = tf.placeholder(tf.float32, [None, img_size_flat],name='input')
        y_true = tf.placeholder(tf.float32, [None, num_classes],name='true_label')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing

        pre_Wbin,pre_Wfluc=[],[]
        temp=[img_size_flat,*layer_sizes]
        for i in range(len(layer_sizes)):
            pre_Wbin.append(tf.placeholder(tf.float32, shape=[temp[i],temp[i+1]],name="pre_Wbin"+str(i+1)))
            pre_Wfluc.append(tf.placeholder(tf.float32, shape=[temp[i],temp[i+1]],name="pre_Wfluc"+str(i+1)))

    with tf.name_scope('Train_Structure'):
        output, updates = Model(x,y_true,keep_prob,pre_Wbin,pre_Wfluc,layer_sizes,control,'DNN').build_model(learning_rate=learning_rate, time_warp=time_warp)
        loss = get_loss(output, y_true)
        accuracy = get_accuracy(output, y_true)
        # confusion = print_confusion_matrix(output, y_true)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  ##Adam ##GradientDescentOptimizer
        gradients = binary_backprop(loss, output, updates)  # gradients=optimizer.compute_gradients(loss)
        min_op = optimizer.apply_gradients(gradients, global_step=global_step)

    # myname = str(mytry)   # mybatch,mylr,myepochs,myinit,mybin,mystochastic,mybias,mydropout,myupdate,myminus,myvar,myclip,mystd,mywrite
    myname = 'no(' + '%03d' % mycount + ')'
    myname += '_B['+ str(mybatch)
    myname += ']_LR[' + str(mylr)
    myname += ']_LR[' + str(mylr)
    myname += ']_EP[' + str(myepochs)
    myname += ']_t0[' + str(myt)
    myname += ']_DR[' + str(mydvalue)
    myname += ']_i[' + str(myinit)
    myname += ']_BIN[' + str(mybin)
    myname += ']_Sto[' + str(mystochastic)
    myname += ']_b[' + str(mybias)
    myname += ']_DO[' + str(mydropout)
    myname += ']_u[' + str(myupdate)
    myname += ']_neg[-' + str(myminus)
    myname += ']_var[' + str(myvar)
    myname += ']_c[' + str(myclip)
    myname += ']_std[' + str(mystd)
    myname += ']_Save[' + str(mywrite)+']'

    # myname += 'var_' if mode_var else 'novar_'
    # myname += 'clip_' if mode_clip else 'noclip_'
    # myname += 'probs'+str(mode_init)+'_'
    # myname += 'std'+str(mode_std)+'_'
    # myname += 'batch'+str(batch_size)+'_'
    # for _,i in enumerate(layer_sizes):
    #     if _==0:
    #         temp += '('
    #     temp +=str(i)
    #     if _==len(layer_sizes)-1:
    #         temp+=')'
    #     else:
    #         temp+=','
    # print(temp)

    with tf.name_scope('Writer'):
        # embedding_var = tf.Variable(0.)
        accuracy_train = tf.Variable(0.)
        accuracy_test = tf.Variable(0.)
        dvaluecheck = tf.get_collection('mycheck2')

        w_histogram1 = tf.summary.histogram("L1_W_hist", updates[0][2])
        w_histogram2 = tf.summary.histogram("L2_W_hist", updates[1][2])
        w_histogram3= tf.summary.histogram("L1_Wb_hist", updates[0][3])
        w_histogram4 = tf.summary.histogram("L2_Wb_hist", updates[1][3])
        w_histogram5 = tf.summary.histogram("L1_Wf_hist", updates[0][4])
        w_histogram6 = tf.summary.histogram("L2_Wf_hist", updates[1][4])
        # for W
        # w_histogram01 = tf.summary.histogram("A00_L2_#4_W", updates[1][2][:, 4])
        # w_histogram02 = tf.summary.histogram("A00_L2_#9_W", updates[1][2][:, 9])
        # w_histogram01a = tf.summary.histogram("A00_L2_#0_W", updates[1][2][:, 0])
        # w_histogram02a = tf.summary.histogram("A00_L2_#1_W", updates[1][2][:, 1])
        #
        # w_histogram03 = tf.summary.histogram("A01_P400_L1_W", updates[0][2][400, :])
        # w_histogram04 = tf.summary.histogram("A02_P401_L1_W", updates[0][2][401, :])
        # w_histogram05 = tf.summary.histogram("A03_P402_L1_W", updates[0][2][402, :])
        # w_histogram06 = tf.summary.histogram("A04_P403_L1_W", updates[0][2][403, :])
        # w_histogram07 = tf.summary.histogram("A05_P404_L1_W", updates[0][2][404, :])
        #
        # w_histogram08 = tf.summary.histogram("B01_P207_L1_W", updates[0][2][207, :])
        # w_histogram09 = tf.summary.histogram("B02_P208_L1_W", updates[0][2][208, :])
        # w_histogram10 = tf.summary.histogram("B03_P209_L1_W", updates[0][2][209, :])
        # w_histogram11 = tf.summary.histogram("B04_P210_L1_W", updates[0][2][210, :])
        # w_histogram12 = tf.summary.histogram("B05_P211_L1_W", updates[0][2][211, :])
        #
        # w_histogram13 = tf.summary.histogram("C01_P000_L1_W", updates[0][2][0, :])
        # w_histogram14 = tf.summary.histogram("C02_P001_L1_W", updates[0][2][1, :])
        # w_histogram15 = tf.summary.histogram("C03_P002_L1_W", updates[0][2][2, :])
        #
        # # for number Wb
        # w_histogram16 = tf.summary.histogram("A00_L2_#4_Wb", updates[1][3][:, 4])
        # w_histogram17 = tf.summary.histogram("A00_L2_#9_Wb", updates[1][3][:, 9])
        # w_histogram16a = tf.summary.histogram("A00_L2_#0_Wb", updates[1][3][:, 0])
        # w_histogram17a = tf.summary.histogram("A00_L2_#1_Wb", updates[1][3][:, 1])
        #
        #
        # w_histogram18 = tf.summary.histogram("A01_P400_L1_Wb", updates[0][3][400, :])
        # w_histogram19 = tf.summary.histogram("A02_P401_L1_Wb", updates[0][3][401, :])
        # w_histogram20 = tf.summary.histogram("A03_P402_L1_Wb", updates[0][3][402, :])
        # w_histogram21 = tf.summary.histogram("A04_P403_L1_Wb", updates[0][3][403, :])
        # w_histogram22 = tf.summary.histogram("A05_P404_L1_Wb", updates[0][3][404, :])
        #
        # w_histogram23 = tf.summary.histogram("B01_P207_L1_Wb", updates[0][3][207, :])
        # w_histogram24 = tf.summary.histogram("B02_P208_L1_Wb", updates[0][3][208, :])
        # w_histogram25 = tf.summary.histogram("B03_P209_L1_Wb", updates[0][3][209, :])
        # w_histogram26 = tf.summary.histogram("B04_P210_L1_Wb", updates[0][3][210, :])
        # w_histogram27 = tf.summary.histogram("B05_P211_L1_Wb", updates[0][3][211, :])
        #
        # w_histogram28 = tf.summary.histogram("C01_P000_L1_Wb", updates[0][3][0, :])
        # w_histogram29 = tf.summary.histogram("C02_P001_L1_Wb", updates[0][3][1, :])
        # w_histogram30 = tf.summary.histogram("C03_P002_L1_Wb", updates[0][3][2, :])
        #
        # # for number Wf
        # w_histogram31 = tf.summary.histogram("A00_L2_#4_Wf", updates[1][4][:, 4])
        # w_histogram32 = tf.summary.histogram("A00_L2_#9_Wf", updates[1][4][:, 9])
        # w_histogram31a = tf.summary.histogram("A00_L2_#0_Wf", updates[1][4][:, 0])
        # w_histogram32a = tf.summary.histogram("A00_L2_#1_Wf", updates[1][4][:, 1])
        #
        # w_histogram33 = tf.summary.histogram("A01_P400_L1_Wf", updates[0][4][400, :])
        # w_histogram34 = tf.summary.histogram("A02_P401_L1_Wf", updates[0][4][401, :])
        # w_histogram35 = tf.summary.histogram("A03_P402_L1_Wf", updates[0][4][402, :])
        # w_histogram36 = tf.summary.histogram("A04_P403_L1_Wf", updates[0][4][403, :])
        # w_histogram37 = tf.summary.histogram("A05_P404_L1_Wf", updates[0][4][404, :])
        #
        # w_histogram38 = tf.summary.histogram("B01_P207_L1_Wf", updates[0][4][207, :])
        # w_histogram39 = tf.summary.histogram("B02_P208_L1_Wf", updates[0][4][208, :])
        # w_histogram40 = tf.summary.histogram("B03_P209_L1_Wf", updates[0][4][209, :])
        # w_histogram41 = tf.summary.histogram("B04_P210_L1_Wf", updates[0][4][210, :])
        # w_histogram42 = tf.summary.histogram("B05_P211_L1_Wf", updates[0][4][211, :])
        #
        # w_histogram43 = tf.summary.histogram("C01_P000_L1_Wf", updates[0][4][0, :])
        # w_histogram44 = tf.summary.histogram("C02_P001_L1_Wf", updates[0][4][1, :])
        # w_histogram45 = tf.summary.histogram("C03_P002_L1_Wf", updates[0][4][2, :])

        # w_scalar0 = tf.summary.scalar("L1(210,154)W_", updates[0][2][210, 154])
        # w_scalar1 = tf.summary.scalar("L1(210,154)Wbin", updates[0][3][210, 154])
        # w_scalar2 = tf.summary.scalar("L1(210,154)Wfluc", updates[0][4][210, 154])
        # w_scalar22 = tf.summary.scalar("L1(210,154)dvalue", dvaluecheck[0][210, 154])
        # w_scalar3 = tf.summary.scalar("L1(33,5)W_", updates[0][2][33, 5])
        # w_scalar4 = tf.summary.scalar("L1(33,5)Wbin", updates[0][3][33, 5])
        # w_scalar5 = tf.summary.scalar("L1(33,5)Wfluc", updates[0][4][33, 5])
        # w_scalar222 = tf.summary.scalar("L1(33,5)dvalue", dvaluecheck[0][33,5])
        # w_scalar6 = tf.summary.scalar("L2(184,7)W_", updates[1][2][184, 7])
        # w_scalar7 = tf.summary.scalar("L2(184,7)Wbin", updates[1][3][184, 7])
        # w_scalar8 = tf.summary.scalar("L2(184,7)Wfluc", updates[1][4][184, 7])
        # w_scalar2222 = tf.summary.scalar("L1(184,7)dvalue", dvaluecheck[1][184, 7])
        # w_scalar9 = tf.summary.scalar("L2(184,4)W_", updates[1][2][184, 4])
        # w_scalar10 = tf.summary.scalar("L2(184,4)Wbin", updates[1][3][184, 4])
        # w_scalar11 = tf.summary.scalar("L2(184,4)Wfluc", updates[1][4][184, 4])
        # w_scalar11 = tf.summary.scalar("L2(184,4)dvalue", dvaluecheck[1][184, 4])
        # w_scalar9 = tf.summary.scalar("L2(184,5)W_", updates[1][2][184, 5])
        # w_scalar10 = tf.summary.scalar("L2(184,5)Wbin", updates[1][3][184, 5])
        # w_scalar11 = tf.summary.scalar("L2(184,5)Wfluc", updates[1][4][184, 5])
        # w_scalar11 = tf.summary.scalar("L2(184,5)dvalue", dvaluecheck[1][184, 5])
        # w_scalar12 = tf.summary.scalar("L2(18,5)W_", updates[1][2][18, 5])
        # w_scalar13 = tf.summary.scalar("L2(18,5)Wbin", updates[1][3][18, 5])
        # w_scalar14 = tf.summary.scalar("L2(18,5)Wfluc", updates[1][4][18, 5])
        # w_scalar22222 = tf.summary.scalar("L2(18,5)dvalue", dvaluecheck[1][18, 5])
        # w_scalar15 = tf.summary.scalar("L2(180,8)W_", updates[1][2][180, 8])
        # w_scalar16 = tf.summary.scalar("L2(180,8)Wbin", updates[1][3][180, 8])
        # w_scalar17 = tf.summary.scalar("L2(180,8)Wfluc", updates[1][4][180, 8])
        # w_scalar222222 = tf.summary.scalar("L2(180,8)dvalue", dvaluecheck[1][180, 8])
        # w_scalar15 = tf.summary.scalar("L2(180,9)W_", updates[1][2][180, 9])
        # w_scalar16 = tf.summary.scalar("L2(180,9)Wbin", updates[1][3][180, 9])
        # w_scalar17 = tf.summary.scalar("L2(180,9)Wfluc", updates[1][4][180, 9])
        # w_scalar222222 = tf.summary.scalar("L2(180,9)dvalue", dvaluecheck[1][180, 9])
        summary_op = tf.summary.merge_all() #var_batch100/test-1+/drift(0,6)1.001

        Writer = tf.summary.FileWriter(mytry+myname)
        myaccuracy_train= tf.summary.scalar("Accuracy_Train", accuracy_train)
        myaccuracy_test = tf.summary.scalar("Accuracy_Test", accuracy_test)

        # config = projector.ProjectorConfig()
        # embedding = config.embeddings.add()
        # embedding.tensor_name = embedding_var.name
        # embedding.metadata_path = os.path.join("./logs/", 'metadata.tsv')
        # projector.visualize_embeddings(Writer, config)

    with tf.Session() as sess:
        np.random.seed(3003)
 #      sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        best_accuracy = 0
        feed_dict = make_feeddict(pre_Wbin=pre_Wbin,pre_Wfluc=pre_Wfluc,mode=0)
        total_batch = int(len(trainset) / batch_size) if mode_test else int(data.train.num_examples / batch_size)
        #
        # if mode_write == 1:
        #    Writer.add_summary(sess.run(summary_op, feed_dict=feed_dict), global_step=0)
        total_cm = []

        for epoch in range(training_epochs):
            sess.run(tf.assign(learning_rate,mylr))
            sess.run(tf.assign(time_warp, 1))
            if epoch > training_epochs-10:
                sess.run(tf.assign(learning_rate, 0))
                sess.run(tf.assign(time_warp,time_warp*0.1))
            avg_cost = 0
            if mode_test==1:
                batchindex = np.arange(len(trainset))
                np.random.shuffle(batchindex)
            #######################################################################################################
            for i in range(total_batch):
                if mode_test == 0:  # General Input Data
                    batch_xs, batch_ys = data.train.next_batch(batch_size, )
                else:  # Test Input Data, e.g: 1 and 6,0 and 6
                    batch_xs = trainset[batchindex[i * batch_size:(i + 1) * batch_size]]
                    batch_ys = trainindex[batchindex[i * batch_size:(i + 1) * batch_size]]
                # We need Weight series of pre-cycle and this cycle for every Training
                feed_dict[x], feed_dict[y_true], feed_dict[keep_prob] = batch_xs, batch_ys, mode_dropout
                temp2 = sess.run(make_feeddict(updates=updates, pre_Wbin=pre_Wbin, pre_Wfluc=pre_Wfluc, mode=1),
                                 feed_dict=feed_dict)

                c, _ = sess.run([loss, min_op], feed_dict=feed_dict)
                feed_dict = make_feeddict(updates, pre_Wbin, pre_Wfluc, feed_dict=feed_dict, save=temp2, mode=2)
                avg_cost += c / total_batch
                if mode_write == 1:
                    Writer.add_summary(sess.run(summary_op, feed_dict=feed_dict),
                                       global_step=epoch * total_batch + i)  # global_step=epoch*total_batch+i+1
                    # saver = tf.train.Saver()
                    # saver.save(sess, os.path.join("./logs/", "model.ckpt"), epoch)
                """
                if i % 1000 == 0:  # Print accuracy, because accuracy changes so fast when we just use two numbers
                    print('Cycle:', '%05d' % (epoch * batch_size + i), 'Accuracy in train_set:',
                          sess.run(accuracy, feed_dict={x: trainset, y_true: trainindex, keep_prob: 1}))
                    print(' '*len("Cycle: 00000"),'cost =', '{:.9f}'.format(c))
                """
            # After finishing every epoch:
            # Stabilize Weight to evaluate:
            temp2 = sess.run(make_feeddict(updates=updates, pre_Wbin=pre_Wbin, pre_Wfluc=pre_Wfluc, mode=1),
                             feed_dict=feed_dict)
            feed_dict = make_feeddict(updates=updates, pre_Wbin=pre_Wbin, pre_Wfluc=pre_Wfluc, feed_dict=feed_dict,
                                      save=temp2, mode=2)
            # Evaluate in Train set
            feed_dict[x], feed_dict[y_true], feed_dict[keep_prob] = trainset, trainindex, 1
            temp_val = sess.run(accuracy, feed_dict=feed_dict)
            sess.run(tf.assign(accuracy_train, temp_val))
            Writer.add_summary(sess.run(myaccuracy_train), global_step=epoch)
            # Evaluate in Test set
            feed_dict[x], feed_dict[y_true], feed_dict[keep_prob] = testset, testindex, 1
            temp_val = sess.run(accuracy, feed_dict=feed_dict)
            sess.run(tf.assign(accuracy_test, temp_val))
            Writer.add_summary(sess.run(myaccuracy_test), global_step=epoch)
            # We save best accuracy:
            if accuracy_train.eval() > best_accuracy:
                best_accuracy = accuracy_train.eval()
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
            print(' ' * len('Epoch: 0000'), 'Accuracy in train_set:', accuracy_train.eval())
            print('Accuracy in test_set:', sess.run(accuracy, feed_dict=feed_dict))
            # After finishing every epoch:
            # Stabilize Weight to evaluate:
            cls_true = data.test.cls
            cls_pred = sess.run(tf.argmax(output, 1), feed_dict=feed_dict)
            # cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
            # total_cm.append(cm)
            # print(cm)

        print('Best Accuracy:',best_accuracy)
        print('Accuracy in test_set:', sess.run(accuracy, feed_dict=feed_dict))
        soft_max_out = sess.run(tf.nn.softmax(output), feed_dict=feed_dict)
        df = pd.DataFrame(soft_max_out)
        df.to_excel(myfilename, index=False, header=False)


    #########################################################################
        # Confusion matrix

        # Get the true classifications for the test-set.
        # cls_true = data.test.cls
        # cls_pred = sess.run(tf.argmax(output,1),feed_dict=feed_dict)
        #
        # # Get the predicted classifications for the test-set.
        # # cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
        #
        # # Get the confusion matrix using sklearn.
        # cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
        #
        # # Print the confusion matrix as text.
        # print(cm)

        # # Plot the confusion matrix as an image.
        # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        #
        # # Make various adjustments to the plot.
        # plt.tight_layout()
        # plt.colorbar()
        # tick_marks = np.arange(num_classes)
        # plt.xticks(tick_marks, range(num_classes))
        # plt.yticks(tick_marks, range(num_classes))
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.show()

        # Use TensorFlow to get a list of boolean values
        # whether each test-image has been correctly classified,
        # and a list for the predicted class of each image.
#####################################################################################################


######################################################################################################
        # Correctness
        cls_true = data.test.cls
        cls_pred = sess.run(tf.argmax(output, 1), feed_dict=feed_dict)
        correct_prediction = tf.equal(cls_pred, cls_true)
        correct = sess.run(correct_prediction, feed_dict=feed_dict)

        # print(correct)
        # # Negate the boolean array.
        # incorrect = (correct == False)
        #
        # # Get the images from the test-set that have been
        # # incorrectly classified.
        # images = data.test.images[incorrect]
        #
        # # Get the predicted classes for those images.
        # cls_truetrue = cls_true[incorrect]
        # logits_pred = cls_pred[incorrect]

        # Get the true classes for those images.

        # Plot the first 9 images.
        # plot_images(images=images[0:64],
        #             cls_true=cls_truetrue[0:64],
        #             cls_pred=logits_pred[0:64])
        #
        #
        # plt.show()



    sess.close()
    Writer.close()
    tf.reset_default_graph()
    # return correct, cls_pred, total_cm
    return cls_true, cls_pred



#def BinaryNet(mybatch,mystd):


# BinaryNet(mytry='004',  mybatch=1000, mylr=0.0001, myepochs=15, myinit=0, mybin=1, mystochastic=0,
#               mybias=0, mydropout=1, myupdate=0, myminus=1, myvar=1, myclip=0, mystd=1, mywrite=1)
# BinaryNet(mytry='005',  mybatch=2000, mylr=0.0001, myepochs=15, myinit=0, mybin=1, mystochastic=0,
#               mybias=0, mydropout=1, myupdate=0, myminus=1, myvar=1, myclip=0, mystd=1, mywrite=1)

# for iii in (1000,2000,3000):
#     BinaryNet(mytry='006', mybatch=iii, mylr=0.0001, myepochs=15, myinit=0, mybin=1, mystochastic=0,
#               mybias=0, mydropout=1, myupdate=0, myminus=1, myvar=1, myclip=0, mystd=1, mywrite=1)


    # BinaryNet(mytry='003',mynumbering,mybatch=1000,mylr=0.0001,myepochs=30,myinit=0,mybin=1,mystochastic=0,mybias=0,mydropout=1,myupdate=0,myminus=1,myvar=1,myclip=0,mystd=1,mywrite=1)
#
# mytry='006'
# mycount=1
# # mybatch=1000,mylr=0.0001,myepochs=60,
#
# myinit=0
# mybin=1
# mystochastic=0
# mybias=0
# mydropout=1
# myupdate=0
# myminus=1
# myvar=1
# myclip=0
# mystd=1
# mywrite=1
#
# b={}
# for mybatch in (1000,):  #100,50,25
#     for mylr in (0.0001,):   #0.0001,0.001,0.00001
#         for myepochs in (5,):  #30,60
#             for mydvalue in (0.09,0):
#                 for myt in (1000,):
#                     b['correct_' + str(mycount)] = BinaryNet('try(correct_test)_' + mytry + '_', mycount, mybatch, mylr,
#                                                              myepochs, myt, mydvalue, myinit, mybin, mystochastic,
#                                                              mybias, mydropout, myupdate, myminus, myvar, myclip, mystd,
#                                                              mywrite)
#                     b['correct_' + str(mycount)]
#
#                     # BinaryNet('004',a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,
#             #           a13,a14)
#                     mycount = mycount +1
#
#
# # dri0_morX
# # c_1 = b['correct_1']
# # c_2 = b['correct_2']
# #
# # images = data.test.images[(c_1)&(~c_2)]
# # # driX_nor0 = data.test.images[~correct_1&correct_2]
# #
# # cls_truetrue = cls_true[c_1 & (~c_2)]
# # logits_pred = cls_pred[(~c_2) & c_1]
# #
# # plot_images(images=images[0:9],
# #             cls_true=cls_truetrue[0:9],
# #             cls_pred=logits_pred[0:9])
# #
# # plt.show()


# For OPERA: tensorboard --logdir="Full path WITHOUT the last \ for indicating a director"
# For OPERA: 127.0.0.1:6006