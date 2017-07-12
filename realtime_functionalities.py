import argparse
from datetime import datetime,timedelta
from collections import Counter

import numpy as np
import time

import classifiers
import database
import conf_file
import kinect_global_trajectories


def disorientation(db,avaliable_sensor,time_interval):

    ##TEMP cause i dont have realtime
    time_interval = ['2016-12-07 13:08:00', '2016-12-07 13:10:40']

    if avaliable_sensor['kinect']:

        ##get realtime data from db
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect, time_interval, multithread=0)

        hours = 0
        minute = 0
        second = 3

        # extract features from kinect
        global_traj_features,patient_ID = kinect_global_trajectories.feature_extraction_video_traj(kinect_joints,[hours, minute, second], draw_joints_in_scene=1)

        ##get labels from already trained cluster
        cluster_model = database.load_matrix_pickle('C:/Users/ICT4life/Documents/python_scripts/BOW_trained_data/cl_model_2secWindow_band03.txt')
        labels_cluster = database.load_matrix_pickle('C:/Users/ICT4life/Documents/python_scripts/BOW_trained_data/cluster_labels.txt')
        labels_counter_counter = Counter(labels_cluster).most_common(40)


        #####
        ##classify spatial-temporal features using bow
        bow_data = kinect_global_trajectories.bow_traj(global_traj_features[1],cluster_model,labels_counter_counter)

def unix_time_ms(date):

    return (date - datetime(1970,1,1)).total_seconds() * 1000


def fall_detection(db): #, available_sensor):

    #if available_sensor['kinect']:
    colKinect = db.Kinect
    requestDate = datetime.now()

    requestInterval = 1  # seconds
    fps = 30

    jointsOfInterest = [3, 20, 8, 9, 10, 4, 5, 6, 16, 17, 18, 12, 13, 14]

    model = classifiers.model_Fall_skel(nSteps=fps, nVars=len(jointsOfInterest) * 3,
                                        RNN=[500, False, 1],
                                        lrnRate=10**-4, pDrop=0.2)

    #model = classifiers.load_model_weights(model, 'path to model weights')

    while True: # enter a stopping criterion

        t1 = datetime.now()

        timeStart = requestDate.strftime("%Y-%m-%d %H:%M:%S")
        timeEnd = (requestDate + timedelta(seconds=requestInterval)).strftime("%Y-%m-%d %H:%M:%S")

        d = database.read_kinect_data_from_db(collection=colKinect,
                                              time_interval=[timeStart, timeEnd],
                                              session='',
                                              skeletonType='raw',
                                              exclude_columns=['ColorImage', 'DepthImage',
                                                               'InfraImage', 'BodyImage'])

        hasSkeleton = [False, False, False, False, False, False]
        idSkeleton = [0, 0, 0, 0, 0, 0]
        skeletons = []

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

        else:

            t2 = datetime.now()

            requestDate += timedelta(seconds=requestInterval)

            print '{0} | Not enough data ({1})'.format(timeStart, len(d))

        if (t2 - t1).seconds < requestInterval:
                time.sleep(requestInterval - (t2 - t1).seconds)




def main_realtime_functionalities():
    ##INPUT: path of configuration file for available sensor
    parser = argparse.ArgumentParser(description='path to conf file')
    parser.add_argument('path_confFile')
    args = parser.parse_args()

    ##initialize and parse config file
    conf_file.parse_conf_file(args.path_confFile)

    avaliable_sensor = conf_file.get_available_sensors()

    # connect to the db
    db = database.connect_db('my_first_db')

    # Real time Functionalities for deliverable M12

    #consider the last 20 seconds from now
    time_end = datetime.now()
    time_begin= time_end-timedelta(seconds=20)

    time_end = unicode(time_end.replace(microsecond=0))
    time_begin = unicode(time_begin.replace(microsecond=0))

    disorientation(db,avaliable_sensor,[time_begin,time_end])

    ##---- CERTH --
    #fall_detection(db)
    ##------




if __name__ == '__main__':
    main_realtime_functionalities()