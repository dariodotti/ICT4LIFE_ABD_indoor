import argparse
from datetime import datetime,timedelta
from collections import Counter


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




if __name__ == '__main__':
    main_realtime_functionalities()