import argparse
from datetime import datetime,timedelta
from collections import Counter
import numpy as np
import time
#import winsound
import os

import classifiers
import database
import conf_file
import kinect_features
import get_token



abs_folder_path = os.path.abspath(__file__)[:-27]
cluster_model = database.load_matrix_pickle(abs_folder_path + 'BOW_trained_data/BOW/cl_30_kmeans_model_2secWindow_newVersion.txt')
key_labelss = database.load_matrix_pickle(abs_folder_path + 'BOW_trained_data/BOW/cluster_30_kmeans_word_newVersion.txt')
key_labels = map(lambda x: x[0], key_labelss)
def disorientation(skeletons, timeStart,timeEnd, requestInterval, hist):

    ##get realtime data from db
    #kinect_joints = database.read_kinect_joints_from_db(, [timeStart,timeEnd], multithread=0)
    #returns a dict where key is sensorID

    if len(skeletons)>0:

        hours = 0
        minutes = 0
        seconds = requestInterval

        # extract features from kinect
        global_traj_features = kinect_features.feature_extraction_video_traj(np.array(skeletons), draw_joints_in_scene= 0,realtime=1)


        #####
        ##classify spatial-temporal features using bow
        hist, pred_result, pred_conf = kinect_features.bow_traj(global_traj_features[1], cluster_model, key_labels, hist)

    else:
        print '--------------- no data in the time interval ---------------------'
        pred_result = 500
        hist = np.zeros((1, len(key_labels)))
        pred_conf = 0

    return hist, pred_result, pred_conf



def model_Fall_skel(nSteps, nVars, RNN, lrnRate, pDrop=0.5):

        from keras.models import Model
        from keras.layers import Input, TimeDistributed, Dense, LSTM, Dropout
        from keras.optimizers import Nadam


        if RNN[1]:
            model_in = Input(batch_shape=(RNN[2], nSteps, nVars))
        else:
            model_in = Input(shape=(nSteps, nVars))

        x = LSTM(units=RNN[0],
                 implementation=0,
                 return_sequences=True,
                 stateful=RNN[1],
                 unroll=True,
                 dropout=0,#pDrop,
                 recurrent_dropout=0,#pDrop,
                 activation='tanh')(model_in)

        x = TimeDistributed(Dense(units=RNN[0],
                                  activation='tanh'))(x)

        x = TimeDistributed(Dense(units=2,
                                  activation='softmax'))(x)

        model_out = x



        model = Model(inputs=model_in, outputs=model_out)

        model.compile(loss='binary_crossentropy',    # binary_crossentropy, categorical
                      optimizer=Nadam(lr=lrnRate),   # Nadam, Adam, RMSprop, SGD
                      sample_weight_mode='None')

        model.summary(line_length=100)


        return model


def fall_down_manager(skeletons, jointsOfInterest, requestInterval, timeStart, fps, model, d, key, results,
                      d_all_MSband, idSkeleton):
    # center data, select joints and delete confidence
    ts = []
    data = []
    sk_var = []

    for i in range(len(skeletons)):

        ts.append([])
        data.append([])
        sk_var.append([])

        for j in range(len(skeletons[i])):

            # center x, z values but not y
            skeletons[i][j][1][:, 0] = skeletons[i][j][1][:, 0] - skeletons[i][j][1][0, 0]
            skeletons[i][j][1][:, 2] = skeletons[i][j][1][:, 2] - skeletons[i][j][1][0, 2]

            joints2delete = np.setdiff1d(range(25), jointsOfInterest)
            skeletons[i][j][1] = np.delete(skeletons[i][j][1], joints2delete, axis=0)
            skeletons[i][j][1] = np.delete(skeletons[i][j][1], [3], axis=1)

            # calculate skeleton variance between frames
            if j > 0:
                sk_var[i].append(np.max(skeletons[i][j][1] - skeletons[i][j - 1][1]))

            ts[i].append(skeletons[i][j][0])
            data[i].append(skeletons[i][j][1])

        ts[i] = np.array(ts[i])
        data[i] = np.array(data[i])

    # if there are enough data (>20 fps)
    if len(data) > 0 and data[0].shape[0] >= requestInterval * 13:#len(data) > 0 and data[0].shape[0] >= requestInterval * 20:

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

            prediction = model.predict(data_interp)  # , batch_size=64)


            hold_max = -1
            hold_id = ''
            for band_key in d_all_MSband.keys():
                d_band = d_all_MSband[band_key]
                d_band = np.array(d_band)
                mag = np.sqrt(np.sum(d_band[:, [2, 3, 4]] ** 2, axis=1, dtype=float))
                if np.max(mag) > hold_max:
                    hold_max = np.max(mag)
                    hold_id = band_key

            # if a band is present
            if hold_max > -1:
                if np.max(prediction[0, :, 1]) > 0.98 and \
                                np.mean(prediction[0, :, 1]) > 0.25 and \
                                hold_max > 2.3:

                    # print '{0} | skeleton {1} | Fall: {2:02.1f} % | Fall Mean: {3:02.1f} %' \
                    # ' | Acc: {4} | Time: {5}ms'.\
                    # format(timeStart, np.nonzero(idSkeleton)[0][k], 100 * np.max(prediction[0, :, 1]),
                    # 100 * np.mean(prediction[0, :, 1]), hold_max, (t2 - t1).microseconds / 1000)
                    print  'BAND fall down: ', 100 * np.max(prediction[0, :, 1]), 100 * np.mean(prediction[0, :, 1])

                    #winsound.Beep(2500, 500)
                    # results.append({'TimeStamp': timeStart,
                    #                         'Event': 'FallDown',
                    #                         'Duration': int(requestInterval),
                    #                         'Sensor': 'Band',
                    #                         'SensorID': key,
                    #                         'BodyID': int(np.nonzero(idSkeleton)[0][k])})

                    results.append({'TimeStamp': timeStart,
                                    'Event': 'FallDown',
                                    'Duration': int(requestInterval),
                                    'Sensor': 'Kinect',
                                    'SensorID': key,
                                    'BodyID': int(np.nonzero(idSkeleton)[0][k])})
                else:
                    print '---no fall down detected---'


            else:  # only kinect

                if 100 * np.max(prediction[0, :, 1]) > 98.0:
                    #winsound.Beep(4000, 500)

                    print 'KINECT fall down prediction: ', 100 * np.max(prediction[0, :, 1])

                    results.append({'TimeStamp': timeStart,
                                    'Event': 'FallDown',
                                    'Duration': int(requestInterval),
                                    'Sensor': 'Kinect',
                                    'SensorID': key,
                                    'BodyID': int(np.nonzero(idSkeleton)[0][k])})


    else:
        print '{0} | Analysing frames ({1})'.format(timeStart, len(d))
        # print 'analysing frame'

    return results


def loss_of_balance_manager(skeletons, idSkeleton, stand, requestInterval, requestDate, timeStart, results, key):

    #print '-----------loss of balance manager-------------'

    idSkeleton = np.array(idSkeleton)

    for k in range(len(skeletons)):

        skeletons[k] = np.array(skeletons[k])
        stand[k] = np.array(stand[k])

        # if there are enough data (>10 fps)
        if skeletons[k].shape[0] >= requestInterval * 10:

            # prediction = np.sqrt(np.sum(skeletons[k]**2, 1))
            prediction = np.sum(np.sqrt(np.abs(skeletons[k])), axis=1)

            #print np.all(stand[k][:, 0] > 1), prediction
            thres1 = 1
            thres2 = 0.9

            if np.all(stand[k][:, 0] > thres1) and np.any(prediction > thres2):
                print '{0} | skeleton {1} | Loss of Balance: {2:02.1f} % | Time: {3}ms'. \
                    format(timeStart, np.nonzero(idSkeleton)[0][k],
                           100 * np.average(np.bitwise_and(stand[k][:, 0] > 1, prediction > thres2)),
                           1000)

                if (100 * np.average(np.bitwise_and(stand[k][:, 0] > thres1, prediction > thres2))) > 80 :
                    print 'sending notification ..'
                    #winsound.Beep(2000, 500)


                    results.append({'TimeStamp': timeStart,
                                    'Event': 'LossOfBalance',
                                    'Duration': int(requestInterval),
                                    'Sensor': 'Kinect',
                                    'SensorID': key,
                                    'BodyID': int(np.nonzero(idSkeleton)[0][k])})

    return results


def heart_rate_manager(d_all_MSband, timeStart, requestInterval, thrLower, thrUpper, thrLength, results):

    for key in d_all_MSband.keys():

        d = d_all_MSband[key]

        hr = []
        for i in range(len(d)):
            if d[i][9] > 0: # hr confidence
                hr.append(d[i][8])

        hr = np.array(hr)

        indLower = hr < thrLower
        indUpper = hr > thrUpper

        if np.sum(indLower) > thrLength:
            results.append({'TimeStamp': timeStart,
                            'Event': 'HeartRateLow',
                            'Duration': int(requestInterval),
                            'Sensor': 'MSBand',
                            'SensorID': key,
                            'BodyID': -1})

        if np.sum(indUpper) > thrLength:
            results.append({'TimeStamp': timeStart,
                            'Event': 'HeartRateHigh',
                            'Duration': int(requestInterval),
                            'Sensor': 'MSBand',
                            'SensorID': key,
                            'BodyID': -1})

    return results


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
                 dropout=0,#pDrop,
                 recurrent_dropout=0,#pDrop,
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
                 dropout=0,#pDrop,
                 recurrent_dropout=0,#pDrop,
                 activation='linear')(model_in)

        x = LSTM(units=RNN[0],
                 implementation=0,
                 return_sequences=False,
                 stateful=RNN[1],
                 unroll=True,
                 dropout=0,#pDrop,
                 recurrent_dropout=0,#pDrop,
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
             dropout=0,#pDrop,
             recurrent_dropout=0,#pDrop,
             activation='linear')(model_in)

    x = LSTM(units=RNN[0],
             implementation=0,
             return_sequences=False,
             stateful=RNN[1],
             unroll=True,
             dropout=0,#pDrop,
             recurrent_dropout=0,#pDrop,
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


def load_models(jointsOfInterest, fps):
    model_fall_down = model_Fall_skel(nSteps=fps, nVars=len(jointsOfInterest) * 3,
                                                  RNN=[500, False, 1],
                                                  lrnRate=10 ** -4, pDrop=0.2)

    # model_fall_down = load_model_weights(model_fall_down, 'C:/Users/Dell/Desktop/ICT4LIFE_ABD_indoor/BOW_trained_data/fall_sk_0.20.hdf5')

    model_fall_down = kinect_features.load_model_weights(model_fall_down,
                                                     abs_folder_path+'BOW_trained_data/fall_sk_10.50.hdf5')

    file_calibaration = abs_folder_path+'BOW_trained_data/APM_KINECT_Calibration_12_9_17.txt'

    RT = np.genfromtxt(file_calibaration, delimiter=',')
    R = RT[0:3, :]
    T = RT[-1, :]
    T = T.reshape(1, -1)

    #R = np.array([[-0.8849, -0.2369, -0.4010], [-0.4654, 0.4194, 0.7794], [-0.0164, 0.8763, -0.4814]])  # Entrance
    #T = np.array([244.972, -52.2714, 189.5946]).reshape(1, -1)


    return model_fall_down,R,T


def standing(Skeleton, confidence, threshold):
    # calculate bone distances
    if Skeleton[20, 2] < 120:
        return -2, -1, Skeleton[20, 0], Skeleton[20, 1], Skeleton[20, 2]

    b13_14, b13_14c = joint_distance(Skeleton, confidence, 13, 14)
    b17_18, b17_18c = joint_distance(Skeleton, confidence, 17, 18)

    b13_12, b13_12c = joint_distance(Skeleton, confidence, 13, 12)
    b17_16, b17_16c = joint_distance(Skeleton, confidence, 17, 16)

    b0_1, b0_1c = joint_distance(Skeleton, confidence, 0, 1)
    b1_20, b1_20c = joint_distance(Skeleton, confidence, 1, 20)

    if (b13_14c | b17_18c) & (b13_12c | b17_16c) & b0_1c & b1_20c:
        # sum of bones
        d1 = (b13_14 * b13_14c + b17_18 * b17_18c) / (int(b13_14c) + int(b17_18c))
        d2 = (b13_12 * b13_12c + b17_16 * b17_16c) / (int(b13_12c) + int(b17_16c))
        d = d1 + d2 + b0_1 + b1_20

        return Skeleton[20, 2] / d, d, Skeleton[20, 0], Skeleton[20, 1], Skeleton[20, 2]

    else:

        return -1, -1, Skeleton[20, 0], Skeleton[20, 1], Skeleton[20, 2]


def joint_distance(Skeleton, confidence, j1, j2):
    from scipy.spatial import distance
    # return distance.euclidean(joint_coords(Skeleton,j1),joint_coords(Skeleton,j2))
    d = distance.euclidean(Skeleton[j1, :], Skeleton[j2, :])
    conf = (confidence[j1] + confidence[j2]) == 4
    return d, conf


def main_realtime_functionalities():

    ##INPUT: path of configuration file for available sensor
    parser = argparse.ArgumentParser(description='path to conf file')
    parser.add_argument('path_confFile')
    args = parser.parse_args()

    ##initialize and parse config file
    conf_file.parse_conf_file(args.path_confFile)

    avaliable_sensor = conf_file.get_available_sensors()

    if avaliable_sensor['kinect']:

        # connect to the db
        db = database.connect_db('local')

        # Real time Functionalities tested during mid-term review

        ## calling fall detection and loss of balance one after the other every second
        colKinect = db.Kinect
        requestInterval = 1  # seconds
        fps = 30

        jointsOfInterest = [3, 20, 8, 9, 10, 4, 5, 6, 16, 17, 18, 12, 13, 14]

        model_fall_down, R_lb, T_lb = load_models(jointsOfInterest, fps)

        ## initialize bow hist for disorientation
        hist = np.zeros((1, len(key_labels)))
        bow_data = database.load_matrix_pickle(abs_folder_path+'BOW_trained_data/BOW/BOW_30_kmeans_16subject_2sec.txt')
        labels_bow_data = database.load_matrix_pickle(abs_folder_path+'BOW_trained_data/BOW/BOW_30_kmeans_labels_16subject_2sec.txt')
        ##labels meaning: 0 - normal activity , 1- normal-activity, 2-confusion, 3-repetitive, 4-questionnaire at table, 5- making tea
        classifiers.logistic_regression_train(bow_data, np.ravel(labels_bow_data), save_model=0)

        ##To test it nonrealtime##
        #date_end = datetime.strptime('2018-01-09 08:44:13', '%Y-%m-%d %H:%M:%S')

        ## start loop for realtime
        while True:

            results = []

            requestDate = datetime.utcnow() - timedelta(seconds=requestInterval)
            ##To test it nonrealtime##
            #requestDate = date_end - timedelta(seconds=requestInterval)

            print "--------------------------------------\r\n"
            print requestDate

            t1 = datetime.now()

            timeStart = requestDate.strftime("%Y-%m-%d %H:%M:%S")

            timeEnd = (requestDate + timedelta(seconds=requestInterval)).strftime("%Y-%m-%d %H:%M:%S")

            d_all = database.read_kinect_data_from_db(collection=colKinect,
                                                      time_interval=[timeStart, timeEnd],
                                                      session='',
                                                      skeletonType=['filtered','rawGray'],
                                                      exclude_columns=['ColorImage', 'DepthImage',
                                                                       'InfraImage', 'BodyImage'])

            d_all_MSband = database.read_MSBand_from_db(collection=db.MSBand,
                                                        time_interval=[timeStart, timeEnd],
                                                        session='')

            if len(d_all_MSband) == 0: print 'no data from the band'
            if len(d_all) == 0: print 'no data from kinect'

            for key in d_all.keys():

                d = d_all[key]

                hasSkeleton = [False, False, False, False, False, False]
                idSkeleton = [0, 0, 0, 0, 0, 0]
                skeletons_4_falldown = []
                skeletons_4_lb = []
                skeletons_4_disorientation = []

                coords = []
                confidence = []
                skeletoncoords = []
                stand = []

                # read data
                for i in range(len(d)):

                    for j in range(6):

                        m = max(idSkeleton)

                        # found new skeleton
                        if hasSkeleton[j] == False and d[i][j] != []:
                            idSkeleton[j] = m + 1
                            hasSkeleton[j] = True
                            skeletons_4_falldown.append([])
                            skeletons_4_lb.append([])
                            stand.append([])

                        if hasSkeleton[j] == True and d[i][j] == []:
                            hasSkeleton[j] == False

                        if d[i][j] != []:
							skeletons_4_falldown[idSkeleton[j] - 1].append([kinect_features.unix_time_ms(d[i][j][1]), np.array(d[i][j][7])])
							coords.append(100 * np.array(d[i][j][7])[:, :-1])
							confidence.append(np.array(d[i][j][7])[:, -1])
							skeletons_4_lb[idSkeleton[j] - 1].append([d[i][j][3], d[i][j][4]])
							skeletoncoords.append((np.matmul(R_lb.T, coords[-1].T) + T_lb.T).T)
							skeletons_4_disorientation.append(np.array(d[i][j][8]))


							stand[idSkeleton[j] - 1].append(standing(np.array(skeletoncoords[-1]),
																				 np.array(confidence[-1]), 0))

                results = fall_down_manager(skeletons_4_falldown, jointsOfInterest, requestInterval, timeStart, fps,
                                            model_fall_down, d, key, results, d_all_MSband, idSkeleton)

                results = loss_of_balance_manager(skeletons_4_lb, idSkeleton, stand, requestInterval, requestDate,
                                                  timeStart, results, key)

                #hist, behavior_label, behavior_conf = disorientation(skeletons_4_disorientation, timeStart, timeEnd, requestInterval, hist)

                ##TODO: show behavior label only if it is abnormal with high confidence
                # conf_thresh = 3
                # if (behavior_label == 'confusion' or behavior_label == 'repetitive') and behavior_conf > conf_thresh:
                #     ##create dictionay same structure of other Abnorma events
                #     event= {'TimeStamp': timeStart,
                #                          'Event': behavior_label,
                #                          'Duration': int(requestInterval),
                #                          'Sensor': 'Kinect',
                #                          'SensorID': key,
                #                          'BodyID': int(np.nonzero(idSkeleton)[0][k])}

                #     database.write_event(db, event)


                ## writing the results into the db
                if len(results) > 0:
                    for event in results:
						print event
                        ## writing the results into the db
						database.write_event(db, event)
                        ## send live notification
                        #get_token.real_report('MSFT Band UPM f6:65', event['Event'])

            results = []
            results = heart_rate_manager(d_all_MSband, timeStart, requestInterval, 50, 120, 10, results)

            # write results to db
            for event in results:
                # writing the results into the db
                database.write_event(db, event)
                # send live notification
                #get_token.real_report('MSFT Band UPM f6:65', event['Event'])

            requestDate = requestDate + timedelta(seconds=requestInterval)
            ##To test it non real time
            #date_end = requestDate

            t2 = datetime.now()
            if (t2 - t1).seconds < requestInterval:
                time.sleep(requestInterval - (t2 - t1).seconds)
            else:
                print 'Delayed !'






if __name__ == '__main__':


    main_realtime_functionalities()
