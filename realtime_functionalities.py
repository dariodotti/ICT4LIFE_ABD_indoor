import argparse
from datetime import datetime,timedelta
from collections import Counter
import numpy as np
import time

import classifiers
import database
import conf_file
import kinect_global_trajectories

print kinect_global_trajectories




def disorientation(db,avaliable_sensor):

    ##TEMP cause i dont have realtime
    time_interval = ['2017-08-08 13:12:05', '2017-08-08 13:48:10']

    if avaliable_sensor['kinect']:

        ##get realtime data from db
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect, time_interval, multithread=0)

        if len(kinect_joints)>0:

            hours = 0
            minute = 0
            second = 2

            # extract features from kinect
            global_traj_features,patient_ID = kinect_global_trajectories.feature_extraction_video_traj(kinect_joints,[hours, minute, second], draw_joints_in_scene=1)

            ##get labels from already trained cluster
##            cluster_model = database.load_matrix_pickle('C:/Users/certhadmin/Desktop/ICT4LIFE_ABD_indoor/BOW_trained_data/BOW/cl_30_kmeans_model_2secWindow_newVersion.txt')
##            key_labels = database.load_matrix_pickle('C:/Users/certhadmin/Desktop/ICT4LIFE_ABD_indoor/BOW_trained_data/BOW/cluster_30_kmeans_word_newVersion.txt')
##            key_labels = map(lambda x: x[0],key_labels)
            cluster_model = database.load_matrix_pickle('C:/Users/certhadmin/Documents/ABD_files/test_cluster_without_outliers/cl_30_kmeans_model_2secWindow_without_outliers.txt')
            key_labels = database.load_matrix_pickle('C:/Users/certhadmin/Documents/ABD_files/test_cluster_without_outliers/cluster_3_kmeans_word__without_outliers.txt')
            key_labels = map(lambda x: x[0],key_labels)


            #####
            ##classify spatial-temporal features using bow
            bow_data = kinect_global_trajectories.bow_traj(global_traj_features[1],cluster_model,key_labels)
        
        else:
            print '--------------- no data in the time interval ---------------------'


def unix_time_ms(date):

    return (date - datetime(1970,1,1)).total_seconds() * 1000

def model_Fall_skel(nSteps, nVars, RNN, lrnRate, pDrop=0.5):

        from keras.models import Model
        from keras.layers import Dense, LSTM
        from keras.layers import Input
        from keras.optimizers import Nadam


        if RNN[1]:
            model_in = Input(batch_shape=(RNN[2], nSteps, nVars))
        else:
            model_in = Input(shape=(nSteps, nVars))

        x = LSTM(units=RNN[0],
                 implementation=0,
                 return_sequences=False,
                 stateful=RNN[1],
                 unroll=True,
                 dropout=pDrop,
                 recurrent_dropout=pDrop,
                 activation='tanh')(model_in)

        x = Dense(units=RNN[0],
                  activation='tanh')(x)

        x = Dense(units=2,
                  activation='softmax')(x)

        model_out = x



        model = Model(inputs=model_in, outputs=model_out)

        model.compile(loss='binary_crossentropy',    # binary_crossentropy, categorical
                      optimizer=Nadam(lr=lrnRate),   # Nadam, Adam, RMSprop, SGD
                      sample_weight_mode='None')

        model.summary(line_length=100)


        return model


def load_model_weights(model, path):

    model.load_weights(path)

    return model


def fall_detection(db): #, available_sensor):

    #if available_sensor['kinect']:
    colKinect = db.Kinect
    #requestDate = '2017-07-11 13:08:00'
    #requestDate = datetime.strptime(requestDate, "%Y-%m-%d %H:%M:%S")

    requestInterval = 1  # seconds
    fps = 30

    jointsOfInterest = [3, 20, 8, 9, 10, 4, 5, 6, 16, 17, 18, 12, 13, 14]

    model = model_Fall_skel(nSteps=fps, nVars=len(jointsOfInterest) * 3,
                                        RNN=[500, False, 1],
                                        lrnRate=10**-4, pDrop=0.2)

    model = load_model_weights(model, 'C:/Users/certhadmin/Desktop/ICT4LIFE_ABD_indoor/BOW_trained_data/fall_sk_0.20.hdf5')

    #requestDate = datetime.now() -timedelta(hours=2)

    while True: # enter a stopping criterion

        print "--------------------------------------\r\n"

        requestDate = datetime.now() - timedelta(hours=2) - timedelta(seconds=1)
        print requestDate

        t1 = datetime.now()
        

        timeStart = requestDate.strftime("%Y-%m-%d %H:%M:%S")

        timeEnd = (requestDate + timedelta(seconds=requestInterval)).strftime("%Y-%m-%d %H:%M:%S")
        #timeStart=timeEnd

        print timeStart
        print timeEnd


        d = database.read_kinect_data_from_db(collection=colKinect,
                                              time_interval=[timeStart, timeEnd],
                                              session='',
                                              skeletonType='raw',
                                              exclude_columns=['ColorImage', 'DepthImage',
                                                               'InfraImage', 'BodyImage'])
        print t1
        hasSkeleton = [False, False, False, False, False, False]
        idSkeleton = [0, 0, 0, 0, 0, 0]
        skeletons = []
        print len(d)

        # read data
        for i in range(len(d)):

            for j in range(6):

                m = max(idSkeleton)

                # found new skeleton
                if hasSkeleton[j] == False and d[i][j] != []:
                    idSkeleton[j] = m + 1
                    hasSkeleton[j] = True
                    skeletons.append([])

                if hasSkeleton[j] == True and d[i][j] == []:
                    hasSkeleton[j] == False

                if d[i][j] != []:
                    skeletons[idSkeleton[j] - 1].append(
                                        [unix_time_ms(d[i][j][1]), np.array(d[i][j][6])])

        # center data, select joints and delete confidence
        ts = []
        data = []
        for i in range(len(skeletons)):

            ts.append([])
            data.append([])

            for j in range(len(skeletons[i])):

                skeletons[i][j][1] -= skeletons[i][j][1][0]

                joints2delete = np.setdiff1d(range(25), jointsOfInterest)
                skeletons[i][j][1] = np.delete(skeletons[i][j][1], joints2delete, axis=0)
                skeletons[i][j][1] = np.delete(skeletons[i][j][1], [3], axis=1)

                ts[i].append( skeletons[i][j][0] )
                data[i].append( skeletons[i][j][1] )

            ts[i] = np.array(ts[i])
            data[i] = np.array(data[i])

        # if there are enough data (>14 fps)
        if len(data) > 0 and data[0].shape[0] >= requestInterval * 14:

            for k in range(len(data)):

                data_interp = np.zeros((requestInterval * 30, len(jointsOfInterest), 3))

                # interpolate data if needed
                if data[k].shape[0] != requestInterval * 30:

                    for i in range(len(jointsOfInterest)):

                        for j in range(3):

                            new_ts = np.linspace(ts[k][0], ts[k][-1], requestInterval * 30)
                            data_interp[:, i, j] = np.interp(new_ts, ts[k], data[k][:, i, j])

                else:

                    data_interp = data[k]

                data_interp = data_interp.reshape((-1, fps, len(jointsOfInterest) * 3))

                prediction = model.predict(data_interp, batch_size=64)

                t2 = datetime.now()

                print '{0} | skeleton {1} | Fall: {2:02.1f} % | Time: {3}ms'.\
                format(timeStart, k, 100 * prediction[0, 1], (t2 - t1).microseconds / 1000)

            requestDate += timedelta(seconds=requestInterval)

            print requestDate

        else:

            t2 = datetime.now()

            requestDate += timedelta(seconds=requestInterval)

            print '{0} | Not enough data ({1})'.format(timeStart, len(d))

        if (t2 - t1).seconds < requestInterval:
            time.sleep(requestInterval - (t2 - t1).seconds)
        else:
            print 'Delayed !'


def model_Fall_acc(nSteps, nVars, RNN, lrnRate, pDrop=0.5):

        from keras.models import Model
        from keras.layers import Dense, LSTM
        from keras.layers import Input
        from keras.optimizers import Nadam


        if RNN[1]:
            model_in = Input(batch_shape=(RNN[2], nSteps, nVars))
        else:
            model_in = Input(shape=(nSteps, nVars))

        x = LSTM(units=RNN[0],
                 implementation=0,
                 return_sequences=False,
                 stateful=RNN[1],
                 unroll=True,
                 dropout=pDrop,
                 recurrent_dropout=pDrop,
                 activation='tanh')(model_in)

        x = Dense(units=RNN[0],
                  activation='tanh')(x)

        x = Dense(units=2,
                  activation='softmax')(x)

        model_out = x



        model = Model(inputs=model_in, outputs=model_out)

        model.compile(loss='binary_crossentropy',    # binary_crossentropy, categorical
                      optimizer=Nadam(lr=lrnRate),   # Nadam, Adam, RMSprop, SGD
                      sample_weight_mode='None')

        model.summary(line_length=100)


        return model


def model_Festination(nSteps, nVars, RNN, lrnRate, pDrop=0.5):

        from keras.models import Model
        from keras.layers import Input, Dense, LSTM
        from keras.optimizers import Nadam, Adam, RMSprop, SGD


        if RNN[1]:
            model_in = Input(batch_shape=(RNN[2], nSteps, nVars))
        else:
            model_in = Input(shape=(nSteps, nVars))

        x = LSTM(units=RNN[0],
                 implementation=0,
                 return_sequences=True,
                 stateful=RNN[1],
                 unroll=True,
                 dropout=pDrop,
                 recurrent_dropout=pDrop,
                 activation='linear')(model_in)

        x = LSTM(units=RNN[0],
                 implementation=0,
                 return_sequences=False,
                 stateful=RNN[1],
                 unroll=True,
                 dropout=pDrop,
                 recurrent_dropout=pDrop,
                 activation='linear')(x)

        x = Dense(units=RNN[0],
                  activation='linear')(x)

        x = Dense(units=2,
                  activation='softmax')(x)

        model_out = x


        model = Model(inputs=model_in, outputs=model_out)

        model.compile(loss='binary_crossentropy',    # binary_crossentropy, categorical
                      optimizer=Nadam(lr=lrnRate),   # Nadam, Adam, RMSprop, SGD
                      sample_weight_mode='None')

        model.summary(line_length=100)


        return model


def model_LossOfBalance(nSteps, nVars, RNN, lrnRate, pDrop=0.5):

    from keras.models import Model
    from keras.layers import Input, Dense, LSTM
    from keras.optimizers import Nadam, Adam, RMSprop, SGD


    if RNN[1]:
        model_in = Input(batch_shape=(RNN[2], nSteps, nVars))
    else:
        model_in = Input(shape=(nSteps, nVars))

    x = LSTM(units=RNN[0],
             implementation=0,
             return_sequences=True,
             stateful=RNN[1],
             unroll=True,
             dropout=pDrop,
             recurrent_dropout=pDrop,
             activation='linear')(model_in)

    x = LSTM(units=RNN[0],
             implementation=0,
             return_sequences=False,
             stateful=RNN[1],
             unroll=True,
             dropout=pDrop,
             recurrent_dropout=pDrop,
             activation='linear')(x)

    x = Dense(units=RNN[0],
              activation='linear')(x)

    x = Dense(units=2,
              activation='softmax')(x)

    model_out = x


    model = Model(inputs=model_in, outputs=model_out)

    model.compile(loss='binary_crossentropy',    # binary_crossentropy, categorical
                  optimizer=Nadam(lr=lrnRate),   # Nadam, Adam, RMSprop, SGD
                  sample_weight_mode='None')

    model.summary(line_length=100)


    return model


def main_realtime_functionalities():
    ##INPUT: path of configuration file for available sensor
    parser = argparse.ArgumentParser(description='path to conf file')
    parser.add_argument('path_confFile')
    args = parser.parse_args()

    ##initialize and parse config file
    conf_file.parse_conf_file(args.path_confFile)

    avaliable_sensor = conf_file.get_available_sensors()

    # connect to the db
    db = database.connect_db('local')

    # Real time Functionalities for deliverable M12

    #consider the last 20 seconds from now
    #time_end = datetime.now()
    #time_begin= time_end-timedelta(seconds=20)

    #time_end = unicode(time_end.replace(microsecond=0))
    #time_begin = unicode(time_begin.replace(microsecond=0))

    disorientation(db,avaliable_sensor)

    ##---- CERTH --
    #fall_detection(db)
    ##------




if __name__ == '__main__':


    main_realtime_functionalities()
