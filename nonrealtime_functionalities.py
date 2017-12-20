import numpy as np
import argparse
from datetime import datetime

import database
import kinect_global_trajectories
import ambient_sensor
import visualization
import classifiers
import conf_file

import requests
import socket



def daily_motion_training(db,avaliable_sensor):

    time_interval = ['2017-07-12 08:03:00', '2017-07-12 08:05:00']

    if avaliable_sensor['kinect']:
        print 'kinect for daily motion'

        ##get data from database
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect,time_interval,multithread=0)

        #extract features from kinect
        hours = 0
        minutes = 0
        seconds = 4
        date = '2017-07-11'
        time_slice_size = [hours, minutes, seconds]
        global_traj_features,patient_ID = kinect_global_trajectories.feature_extraction_video_traj(kinect_joints,time_slice_size, draw_joints_in_scene=1)

        #Get labels from clustering
        #labels_cluster = classifiers.cluster_kmeans(global_traj_features[1],k=3)

        #Train in supervised way a classifier with extracted features and labels
        #model = classifiers.logistic_regression_train(global_traj_features[1],labels_cluster)

        #Save model in database
        #database.save_classifier_model(model,filename='startPeriod_endPeriod')


        ##visulaization daily motion
        #visualization.bar_plot_occupancy_selectedAreas_over_time(global_traj_features[0])
        #visualization.bar_plot_motion_over_time(global_traj_features[1])
        kinect_motion_amount = visualization.pie_plot_motion_day(global_traj_features[1])

        ##make it dictionary
        kinect_motion_amount = {'stationary': kinect_motion_amount[0][0],'slow_mov': kinect_motion_amount[0][1],'fast_mov': kinect_motion_amount[0][2]}



    if avaliable_sensor['zenith']:
        print 'add img processing zenith camera'
        zenith_data = database.read_zenith_from_db(db.Zenith,time_interval)

    if avaliable_sensor['UPMBand']:
        print 'add upm band processing'
        upmBand_data = database.read_UPMBand_from_db(db.UPMBand,time_interval)

    return kinect_motion_amount,patient_ID


def daily_motion_test(db,avaliable_sensor):

    time_interval = ['2016-12-07 13:00:00', '2016-12-07 14:00:00']

    if avaliable_sensor['kinect']:

        ##get data from database
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect,time_interval)

        #extract features from kinect
        hours = 0
        minutes = 0
        seconds = 25
        time_slice_size = [hours, minutes, seconds]
        global_traj_features = kinect_global_trajectories.feature_extraction_video_traj(kinect_joints,time_slice_size)

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


def as_day_motion(db, avaliable_sensor):

    ##get data in the selected time interval from database
    time_interval = ['2017-07-27 10:18:00', '2017-07-27 10:18:30']


    if avaliable_sensor['ambientSensor']:
        ambient_sensor_data = database.read_ambient_sensor_from_db(db.Binary, time_interval)

        ambient_sensor_activation = ambient_sensor.as_motion(ambient_sensor_data)

        print ambient_sensor_activation

        ##save the results according to table

    if avaliable_sensor['UPMBand']:
        upmBand_data = database.read_UPMBand_from_db(db.UPMBand, time_interval)

    return ambient_sensor_activation


def as_night_motion(db, avaliable_sensor):

    ##get data in the selected time interval from database
    time_interval = ['2017-07-11 13:18:00', '2017-07-11 13:18:30']


    if avaliable_sensor['ambientSensor']:
        ambient_sensor_data = database.read_ambient_sensor_from_db(db.Binary,time_interval)

        ambient_sensor_activation = ambient_sensor.as_motion(ambient_sensor_data)

        ##save the results according to table

    if avaliable_sensor['UPMBand']:
        upmBand_data = database.read_UPMBand_from_db(db.UPMBand,time_interval)

    return ambient_sensor_activation


def abnormal_behavior_classification_training(db, avaliable_sensor):

    ##get data in the selected time interval from database
    time_interval = ['2016-12-07 13:08:00', '2016-12-07 13:15:00']

    if avaliable_sensor['kinect']:
        print 'kinect disorientation tr'
        ##get data from database
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect, time_interval, multithread=0)

        # extract features from kinect
        hours = 0
        minutes = 0
        seconds = 25
        time_slice_size = [hours,minutes,seconds]
        global_traj_features = kinect_global_trajectories.feature_extraction_video_traj(kinect_joints,time_slice_size)

        ## perform clustering
        #classifiers.cluster_meanShift(global_traj_features[1],save_model=1)

        ##retrieve model and get labels
        #cluster_model = database.get_classifier_model('')

        ##get labels from clustering
        #labels_cluster = cluster_model.predict(global_traj_features[1])


        ##Create bag of words with trajectories
        #bow_vocabulary = kinect_global_trajectories.bow_traj(global_traj_features[1],cluster_model,labels_cluster)

        #classifiers.logistic_regression_train(bow_vocabulary,labels_cluster,save_model=1)




    if avaliable_sensor['zenith']:
        zenith_data = database.read_zenith_from_db(db.Zenith,time_interval)

    if avaliable_sensor['UPMBand']:
        upmBand_data = database.read_UPMBand_from_db(db.UPMBand,time_interval)


def apathy():


    f_matrix = database.get_feature_matrix(filename='start-period_end-period')

    visualization.bar_plot_motion_in_region_over_long_time(f_matrix)


def visit_bathroom(db,avaliable_sensor):

    ##get data in the selected time interval from database
    time_interval = ['2016-12-07 13:08:00', '2016-12-07 13:15:00']

    if avaliable_sensor['ambientSensor']:
        ambient_sensor_data = database.read_ambient_sensor_from_db(db.Binary,time_interval)
        toilet_visit = ambient_sensor.nr_visit_bathroom(ambient_sensor_data)

    ##make it a dictionary


    ##TODO: return also confidence value
    return toilet_visit


def get_freezing_loss_of_balance_detection(db):

    time_interval = ['2017-07-12 12:24:00', '2017-07-12 12:27:50']

    #kinect_global_trajectories.freezing_detection(db, time_interval)

    kinect_global_trajectories.loss_of_balance_detection(db, time_interval)



def main_nonrealtime_functionalities():

    # Perform reidentification
    json_data = {"init_hour": "11:00:00", "init_date": "23-11-2017", "fin_hour": "12:00:00", "fin_date": "23-11-2017"}
    r = requests.post("http://"+str(socket.gethostbyname(socket.gethostname()))+":8000/get_all_data/", json=json_data)

    ##INPUT: path of configuration file for available sensor
    parser = argparse.ArgumentParser(description='path to conf file')
    parser.add_argument('path_confFile')
    args = parser.parse_args()

    ##initialize and parse config file
    conf_file.parse_conf_file(args.path_confFile)

    avaliable_sensor = conf_file.get_available_sensors()


    #connect to the db
    db = database.connect_db('local')
    
    #Functionalities for deliverable M12

    #kinect_motion_amount,patient_ID = daily_motion_training(db,avaliable_sensor)

    day_motion_as_activation = as_day_motion(db, avaliable_sensor)

    #night_motion_as_activation = as_night_motion(db, avaliable_sensor)

    #abnormal_behavior_classification_training(db, avaliable_sensor)

    #apathy()

    ## ---------CERTH
    #get_freezing_loss_of_balance_detection(db)

    # summarize HBR, GSR
    #database.summary_MSBand(db,[2017, 7, 6])

    #database.summarize_events_certh('freezing', path)
    ##-------


    ##if we want the total number of visit we take it from the day and night motion
    nr_visit={'number': int,'beginning':[],'duration':[]}
    for k in day_motion_as_activation.keys():
        nr_visit['beginning'].append(day_motion_as_activation[k][0] + night_motion_as_activation[k][0])
        nr_visit['duration'].append(day_motion_as_activation[k][1] + night_motion_as_activation[k][1])
    nr_visit['number'] = len(nr_visit['beginning'][0])


    ## at the end of the day summarize and write all the results in a file that will be uploaded into amazon web services

    ##TODO: complete the method that now are added artificially (i.g. confusion , leave the house)
    #data = ['daily_motion: ',kinect_motion_amount,'as_day_motion: ', day_motion_as_activation, 'as_night_motion: ', night_motion_as_activation,
    #        'visit_bathroom: ',nr_visit, 'confusion_behavior_detection: ',{'number':2,'beginning':[],'duration':[]},'leave the house: ' , '1','leave_house_confused: ','0']


    # Matching person band with uuid and send summarization for one patient
    path_to_lambda = ""
    client = MongoClient('localhost', 27017)
    dbIDs = client['local']['BandPersonIDs']
    uuids = dbIDs.find()
    for uuid in uuids:
        if uuid["SensorID"] == personMSBand:
            uuid_person = uuid["PersonID"]
            final_sumarization = {"date": time.strftime("%Y-%m-%d"), "daily_motion": kinect_motion_amount, "as_day_motion": day_motion_as_activation, "as_night_motion": night_motion_as_activation, "freezing": freezing_analysis, "festination": festination_analysis, "loss_of_balance": loss_of_balance_analisys, "fall_down": fall_down_analysis, "visit_bathroom": nr_visit, "confusion_behaviour_detection": confusion_analysis, "leave_the_house": lh_number, "leave_house_confused": lhc_number}
            with open(path_to_lambda+uuid_person+'.json', 'w') as outfile:
                json.dump(final_sumarization, outfile)



    #database.write_output_file(patient_ID,data,'C:/')







if __name__ == '__main__':
    main_nonrealtime_functionalities()
