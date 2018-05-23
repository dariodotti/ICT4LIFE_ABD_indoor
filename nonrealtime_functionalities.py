import numpy as np
import argparse
import time
from datetime import datetime,timedelta
from pymongo import MongoClient as Connection

import database
import kinect_features
import ambient_sensor
import visualization
import classifiers
import conf_file

import requests
import os

import copy


def daily_motion_training(db,avaliable_sensor, time_interval):

    if avaliable_sensor['kinect']:
        print 'kinect for daily motion'

        ##get data from database
        kinect_joints = database.read_kinect_joints_from_db(db.Kinect,time_interval)
        #kinect_joints = kinect_joints[::-1]
        
        if len(kinect_joints) <1:
            print '----- no data in time interval -----'
            return 0

        #extract features from kinect from each user
        bands_ids = database.get_bands_ID(db)
        global_traj_features = kinect_features.feature_extraction_video_traj(kinect_joints, bands_ids, draw_joints_in_scene=1, realtime=0)

        kinect_motion_amount_band = {b[0]: None for b in bands_ids}
        
        ## compute analysis per band
        for i_b,b in enumerate(bands_ids):
            if len(global_traj_features[1][i_b])>0:
                #Get labels from clustering
                #labels_cluster = classifiers.cluster_kmeans(global_traj_features[1],k=3)

                #Train in supervised way a classifier with extracted features and labels
                #model = classifiers.logistic_regression_train(global_traj_features[1],labels_cluster)

                #Save model in database
                #database.save_classifier_model(model,filename='startPeriod_endPeriod')


                ##visulaization daily motion
                #visualization.bar_plot_occupancy_selectedAreas_over_time(global_traj_features[0])
                #visualization.bar_plot_motion_over_time(global_traj_features[1])
                kinect_motion_amount = visualization.pie_plot_motion_day(global_traj_features[1][i_b],plot=0)

                # normalize motion
                total_motion = np.sum(kinect_motion_amount[0])
                for n_k, k_data in enumerate(kinect_motion_amount[0]): kinect_motion_amount[0][n_k] = k_data / total_motion

                ##make it dictionary
                kinect_motion_amount_band[b[0]]= {'stationary': kinect_motion_amount[0][0],'slow_mov': kinect_motion_amount[0][1],'fast_mov': kinect_motion_amount[0][2]}
                print b[0], {'stationary': kinect_motion_amount[0][0],'slow_mov': kinect_motion_amount[0][1],'fast_mov': kinect_motion_amount[0][2]}
            else:
                print 'no data from band: ', b[0]
                kinect_motion_amount_band[b[0]] = {'stationary': -1,'slow_mov': -1,'fast_mov': -1}
    else:
        print '--------------- no data in the time interval ---------------------'
        kinect_motion_amount = 0
        #patient_ID = 0


    if avaliable_sensor['zenith']:
        print 'add img processing zenith camera'
        zenith_data = database.read_zenith_from_db(db.Zenith,time_interval)

    if avaliable_sensor['UPMBand']:
        print 'add upm band processing'
        upmBand_data = database.read_UPMBand_from_db(db.UPMBand,time_interval)

    return kinect_motion_amount_band

##TODO UPDate this old method
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
        global_traj_features = kinect_features.feature_extraction_video_traj(kinect_joints, time_slice_size)

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


def as_day_motion(db, avaliable_sensor,time_interval):

    ##get data in the selected time interval from database
    #time_interval = ['2017-07-27 10:18:00', '2017-07-27 10:18:30']


    if avaliable_sensor['ambientSensor']:
        ambient_sensor_data = database.read_ambient_sensor_from_db(db.Binary, time_interval)

        ambient_sensor_activation = ambient_sensor.as_motion(ambient_sensor_data)

        #print ambient_sensor_activation

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
        global_traj_features = kinect_features.feature_extraction_video_traj(kinect_joints, time_slice_size)

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


def get_freezing_festination(db):

    time_interval = ['2018-01-09 08:43:13','2018-01-09 08:44:13']

    fr_events = kinect_features.freezing_detection(db, time_interval)

    fe_events = kinect_features.festination(db, time_interval)

    return fr_events, fe_events



def main_nonrealtime_functionalities():
    ##INPUT: path of configuration file for available sensor
    parser = argparse.ArgumentParser(description='path to conf file')
    parser.add_argument('arguments',nargs=2)
    args = parser.parse_args()

    ##initialize and parse config file
    conf_file.parse_conf_file(args.arguments[0])
    day_to_analyze = args.arguments[1]

    avaliable_sensor = conf_file.get_available_sensors()
    end_period = datetime.strptime(day_to_analyze, '%Y-%m-%d %H:%M:%S')
    begin_period = end_period - timedelta(days=1)
    
    print 'requested period: ', begin_period,end_period
    
    # Perform reidentification
    #date1 = begin_period.strftime('%Y-%m-%d %H:%M:%S').split(" ")
    #f1 = date1[0].split("-")
    #date2 = end_period.strftime('%Y-%m-%d %H:%M:%S').split(" ")
    #f2 = date2[0].split("-")
    
    ##json_data = {"init_hour": "10:38:00", "init_date": "1-03-2018", "fin_hour": "10:46:00", "fin_date": "1-03-2018"}
    #json_data = {"init_hour": date1[1], "init_date": f1[2]+"-"+f1[1]+"-"+f1[0], "fin_hour": date2[1], "fin_date": f2[2]+"-"+f2[1]+"-"+f2[0]}
    #r = requests.post("http://"+os.environ['IP_ICT4LIFE']+":8000/get_all_data/", json=json_data)

    #connect to the db
    db = database.connect_db('local')

    ## time interval to analyze
    time_interval = [begin_period.strftime('%Y-%m-%d %H:%M:%S'),end_period.strftime('%Y-%m-%d %H:%M:%S')]

    kinect_motion_amount = daily_motion_training(db,avaliable_sensor,time_interval)

    day_motion_as_activation = as_day_motion(db, avaliable_sensor,time_interval)

    #night_motion_as_activation = as_night_motion(db, avaliable_sensor)

    #abnormal_behavior_classification_training(db, avaliable_sensor)

    #apathy()

    freezing_analysis, festination_analysis = get_freezing_festination(db)

    ##TODO: open the summarization file from the realtime activities and read the values for :
    ##loss_of_balance_analisys, fall_down_analysis, confusion_analysis

    # real-time event summarization
    rt_events_summary = database.summarize_events(db)

    # @todo: choose when to delete events from db
    #deleted = database.delete_events()

    client = Connection('localhost', 27017)
    dbIDs = client['local']['BandPersonIDs']
    uuids = dbIDs.find()
    jdata = {}
    for uuid_person in uuids:
    	jdata[uuid_person["SensorID"]] = 0

    jdata_lob = copy.deepcopy(jdata)
    jdata_fall = copy.deepcopy(jdata)
    jdata_hrL = copy.deepcopy(jdata)
    jdata_hrH = copy.deepcopy(jdata)
    for rID in rt_events_summary:
        if rt_events_summary[rID].has_key("LossOfBalance"):
            jdata_lob[rID] = rt_events_summary[rID]["LossOfBalance"]

        if rt_events_summary[rID].has_key("FallDown"):
            jdata_fall[rID] = rt_events_summary[rID]["FallDown"]

        if rt_events_summary[rID].has_key("HeartRateLow"):
            jdata_hrL[rID] = rt_events_summary[rID]["HeartRateLow"]

        if rt_events_summary[rID].has_key("HeartRateHigh"):
            jdata_hrH[rID] = rt_events_summary[rID]["HeartRateHigh"]

    freezing_analysis = jdata
    festination_analysis = jdata
    loss_of_balance_analysis = jdata_lob
    fall_down_analysis = jdata_fall
    heart_rate_low = jdata_hrL
    heart_rate_high = jdata_hrH
    confusion_analysis = jdata
    lh_number = jdata
    lhc_number = jdata
    nr_visit = jdata
    day_motion_as_activation = jdata



    # @todo: summarize HBR, GSR in case of confusion
    hr, gsr = database.summary_MSBand(db, [begin_period.strftime('%Y-%m-%d %H:%M:%S'), day_to_analyze])

    ##if we want the total number of visit we take it from the day and night motion
    # as_motion_for_json = {key: [] for key in day_motion_as_activation.keys()}
    # for k in day_motion_as_activation.keys():
    #     for aa in day_motion_as_activation[k]:
    #         print aa
    #     as_motion_for_json[k].append({'beginning': day_motion_as_activation[k][0] + night_motion_as_activation[k][0],\
    #      'duration':day_motion_as_activation[k][1] + night_motion_as_activation[k][1] })
    #     print as_motion_for_json[k]

    ###### summarization ####
    ##at the end of the day summarize and write all the results in a file that will be uploaded into amazon web services
    # Matching person band with uuid and send summarization for one patient
    database.write_summarization_nonrealtime_f_json(
            kinect_motion_amount,
            day_motion_as_activation,
            day_motion_as_activation,
            freezing_analysis,
            festination_analysis,
            loss_of_balance_analysis,
            fall_down_analysis,
            nr_visit,
            confusion_analysis,
            lh_number,
            lhc_number,
            heart_rate_low,
            heart_rate_high,
            gsr,
            hr)


if __name__ == '__main__':
    main_nonrealtime_functionalities()
