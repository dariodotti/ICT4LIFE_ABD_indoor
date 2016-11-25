import numpy as np
import argparse

import database
import kinect_global_trajectories
import ambient_sensor
import visualization
import classifiers
import conf_file




def daily_motion_training(db,avaliable_sensor):

    time_interval = ['12:12:12','13:13:13']

    if avaliable_sensor['kinect']:

        ##get data from database
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect,time_interval)

        #extract features from kinect
        global_traj_features = kinect_global_trajectories.feature_extraction_video_traj(kinect_joints)

        #Get labels from clustering
        labels_cluster = classifiers.cluster_kmeans(global_traj_features[1],k=3)

        #Train in supervised way a classifier with extracted features and labels
        model = classifiers.logistic_regression_train(global_traj_features[1],labels_cluster)

        #Save model in database
        database.save_classifier_model(model,filename='startPeriod_endPeriod')


        ##visulaization daily motion
        visualization.bar_plot_occupancy_selectedAreas_over_time(global_traj_features[0])
        visualization.bar_plot_motion_over_time(global_traj_features[1])
        visualization.pie_plot_motion_day(global_traj_features[1])

    if avaliable_sensor['zenith']:
        print 'add img processing zenith camera'
        zenith_data = database.read_zenith_from_db(db.Zenith,time_interval)

    if avaliable_sensor['UPMBand']:
        print 'add upm band processing'
        upmBand_data = database.read_UPMBand_from_db(db.UPMBand,time_interval)


def daily_motion_test(db,avaliable_sensor):

    time_interval = ['12:12:12','13:13:13']
    if avaliable_sensor['kinect']:

        ##get data from database
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect,time_interval)

        #extract features from kinect
        global_traj_features = kinect_global_trajectories.feature_extraction_video_traj(kinect_joints)

        #read model from database
        model= database.get_classifier_model(filename='startPeriod_endPeriod')

        #test classification
        classifiers.logistic_regression_predict(global_traj_features[1])


        ##visulaization daily motion
        visualization.bar_plot_occupancy_selectedAreas_over_time(global_traj_features[0])
        visualization.bar_plot_motion_over_time(global_traj_features[1])
        visualization.pie_plot_motion_day(global_traj_features[1])

    if avaliable_sensor['zenith']:
        print 'add img processing zenith camera'
        zenith_data = database.read_zenith_from_db(db.Zenith,time_interval)

    if avaliable_sensor['UPMBand']:
        print 'add upm band processing'
        upmBand_data = database.read_UPMBand_from_db(db.UPMBand,time_interval)


def night_motion(db,avaliable_sensor):

    ##get data in the selected time interval from database
    time_interval = ['12:12:12','13:13:13']

    if avaliable_sensor['ambientSensor']:
        ambient_sensor_data = database.read_ambient_sensor_from_db(db.Binary,time_interval)

        ambient_sensor.night_motion(ambient_sensor_data)

    if avaliable_sensor['UPMBand']:
        upmBand_data = database.read_UPMBand_from_db(db.UPMBand,time_interval)


def disorientation(db,avaliable_sensor):

    ##get data in the selected time interval from database
    time_interval = ['12:12:12','13:13:13']

    if avaliable_sensor['kinect']:
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect,time_interval)

    if avaliable_sensor['zenith']:
        zenith_data = database.read_zenith_from_db(db.Zenith,time_interval)

    if avaliable_sensor['UPMBand']:
        upmBand_data = database.read_UPMBand_from_db(db.UPMBand,time_interval)


def apathy():


    f_matrix = database.get_feature_matrix(filename='start-period_end-period')

    visualization.bar_plot_motion_in_region_over_long_time(f_matrix)


def visit_bathroom(db,avaliable_sensor):

    if avaliable_sensor['ambientSensor']:
        ambient_sensor_data = database.read_ambient_sensor_from_db(db.Binary)
        ambient_sensor.nr_visit_bathroom(ambient_sensor_data)


def main_nonrealtime_functionalities():

    ##INPUT: path of configuration file for available sensor
    parser = argparse.ArgumentParser(description='path to conf file')
    parser.add_argument('path_confFile')
    args=parser.parse_args()

    ##initialize and parse config file
    conf_file.parse_conf_file(args.path_confFile)

    avaliable_sensor = conf_file.get_available_sensors()


    #connect to the db
    db = database.connect_db('my_first_db')
    
    #Functionalities for deliverable M12
    daily_motion_training(db,avaliable_sensor)

    night_motion(db,avaliable_sensor)

    disorientation(db,avaliable_sensor)

    #apathy()

    visit_bathroom(db,avaliable_sensor)








if __name__ == '__main__':
    main_nonrealtime_functionalities()