import argparse
from datetime import datetime,timedelta

import database
import conf_file
import kinect_global_trajectories


def disorientation(db,avaliable_sensor,time_interval):

    ##TEMP cause i dont have realtime
    time_interval = ['2016-12-07 13:08:00', '2016-12-07 13:08:40']

    if avaliable_sensor['kinect']:

        ##get realtime data from db
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect, time_interval, multithread=0)

        # extract features from kinect
        global_traj_features = kinect_global_trajectories.feature_extraction_video_traj(kinect_joints)

        ##get labels from cluster
        #####
        ##create bag of words with the traj
        #bow_data = kinect_global_trajectories.bow_traj(global_traj_features,cluster_model,labels_training)

        ##classify the new entry

        #classifier = database.get_classifier_model('')
        #prediction = classifier.predict(bow_data)






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