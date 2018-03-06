from pymongo import MongoClient as Connection
import pymongo
import numpy as np
from datetime import datetime
import cPickle
import multiprocessing
import time
import json
import pandas as pd
from lib_amazon_web_server import S3FileManager
#from multiprocessing.dummy import Pool as threadPool


import pandas as pd
from datetime import datetime, timedelta


begin_period = ''
end_period = ''


def connect_db(name_db, ip='localhost'):

    con = Connection(ip, 27017)
    db = con[name_db]
    print db

    return db


def read_events(db):

    # The following assumptions are made for the captured events
    #   - each event is associated with only one sensor (e.g., LossOfBalance <=> Kinect)
    #   - each event is registered only once (2 Kinects cannot report the same LossOfBalance event)
    #   - the events are deleted at the end of each day

    r = db.rtevents.find({}).sort('TimeStamp', 1) # 1 = ASCENDING

    events = dict()
    no_reid_counter = 0
    for i, e in enumerate(r):

        # search for re-id field in Kinect table:
        if e['Sensor'] == 'Kinect':
            timeStart = datetime.strptime(e['TimeStamp'], "%Y-%m-%d %H:%M:%S")
            timeEnd = timeStart + timedelta(seconds=1)
            timeStart = timeStart.strftime("%Y-%m-%d %H:%M:%S")
            timeEnd = timeEnd.strftime("%Y-%m-%d %H:%M:%S")
            d_all = read_kinect_data_from_db(collection=db.Kinect,
                                             time_interval=[timeStart, timeEnd],
                                             session='',
                                             skeletonType=['filtered','rawGray'],
                                             exclude_columns=['ColorImage', 'DepthImage',
                                                              'InfraImage', 'BodyImage'])
            reid = ''
            if e['SensorID'] in d_all.keys():
                d = d_all[e['SensorID']]
                for j in range(len(d)):
                    if d[j][e['BodyID']][2] != '':
                        reid = d[j][e['BodyID']][2]
                        break

            e['reid'] = reid

        # set SensorID as the reid field in MSBand
        if e['Sensor'] == 'MSBand':
            e['reid'] = e['SensorID']

        # split events based on event type
        if events.has_key(e['Event']) == False:
            events[e['Event']] = dict()

        # further split events by re-id (events with no re-id field are discarded)
        if events[e['Event']].has_key(e['reid']) == False and e['reid'] != '':
            events[e['Event']][e['reid']] = []

        # format events as timestamp, duration, sensor
        if e['reid'] != '':
            dt = datetime.strptime(e['TimeStamp'], "%Y-%m-%d %H:%M:%S")
            events[e['Event']][e['reid']].append([dt, e['Duration'], e['Sensor']])
        else:
            no_reid_counter += 1

    print 'Found {0} events that cannot be matched to a person (no re-id)'.format(no_reid_counter)


    return events


def write_event(db, event):

    c = db.rtevents
    c.insert_one(event)


def write_exercise_evaluation(db, event):

    c = db.exeval
    c.insert_one(event)


def summarize_events(db):

    # The following assumptions are made for the captured events
    #   - if timestamp + duration of event(t) = timestamp of event(t+1),
    #     the 2 events are combined into 1

    events = read_events(db)

    # join subsequent events together
    for e_type in events.keys(): # for each event type
        for rID in events[e_type].keys(): # for each re-id value
            for i in range(len(events[e_type][rID]) - 1): # for each event
                e_t = events[e_type][rID][i]
                e_tp1 = events[e_type][rID][i + 1]
                if e_t[0] + timedelta(seconds=e_t[1]) == e_tp1[0]:
                    # the 2 events can be joined into 1
                    events[e_type][rID][i + 1][0] = events[e_type][rID][i][0]
                    events[e_type][rID][i + 1][1] += events[e_type][rID][i][1]
                    events[e_type][rID][i] = None

    # save the summarized events
    event_summary = dict()
    unique_rIDs = []
    for e_type in events.keys(): # for each event type
        event_summary[e_type] = dict()
        for rID in events[e_type].keys(): # for each re-id value
            event_summary[e_type][rID] = []
            for i in range(len(events[e_type][rID])): # for each event
                if events[e_type][rID][i] is not None:
                    event_summary[e_type][rID].append(events[e_type][rID][i])
                if rID not in unique_rIDs:
                    unique_rIDs.append(rID)

    # format events into the desired output (per re-id value)
    output = {rID: {} for rID in unique_rIDs}
    for rID in unique_rIDs: # for each person
        for e_type in event_summary.keys(): # for each event type
            if rID in event_summary[e_type].keys():
                n_events = len(event_summary[e_type][rID])
                e = []
                for i in range(n_events):
                    ts = event_summary[e_type][rID][i][0].strftime("%H:%M:%S")
                    dur = event_summary[e_type][rID][i][1]
                    e.append({"beginning": ts, "duration": str(dur)})
                if n_events > 0:
                    output[rID][e_type] = dict([("number", n_events), ("event", e)])


    return output


def delete_events(db):

    col = db.rtevents
    deleted = col.delete_many({})


    return deleted.deleted_count


def delete_sensor_documents(db, time_interval):

    begin_period, end_period = check_arguments(time_interval, '')

    collections = db.collection_names(include_system_collections=False)
    my_collections = ['Kinect', 'Zenith', 'MSBand', 'HexiwearBand', 'UPMBand', 'Binary', 'WSN']

    result = dict()
    for c in my_collections:
        if c in collections:
            col = db.get_collection(c)
            deleted = col.delete_many({'_id': {'$gt': begin_period, '$lt': end_period}})

            result[c] = deleted.deleted_count


    return result


def read_kinect_data_from_db(collection, time_interval, session, skeletonType, exclude_columns=None):

    """

    Read kinect data (skeleton joints and lean values) from db

    ------------------------------------------------------------------------------------------
    Parameters:

    collection:
        The mongo db collection instance

    time_interval: List of string
        The begin and end time stamps for the query (e.g., '2017-10-01 23:59:59.999')

    session: string
        The session name for the query

    skeletonType: string
        The skeleton type to return. Must be one of raw, rawColor, rawGray,
        filtered, filteredColor, filteredGray

    exclude_columns: list
        The columns to exclude from the query (excluding large fields greatly reduces query times)

    ------------------------------------------------------------------------------------------
    Tested: -
    ------------------------------------------------------------------------------------------

    """

    if skeletonType[0] not in ['raw', 'rawColor', 'rawGray', 'filtered', \
                            'filteredColor', 'filteredGray']:
        raise RuntimeError('skeletonType must be one of raw, rawColor, rawGray, ' +
                            'filtered, filteredColor, filteredGray')

    frames = read_data_from_db_tomas(collection, time_interval, session, exclude_columns)

    # Parse data
    #
    # TODO: parse only frames with skeleton.
    #
    all_data = dict()
    for n_frame, f in enumerate(frames):

        if all_data.has_key(f['SensorID']) == False:
            all_data[f['SensorID']] = []

        all_joints = [[], [], [], [], [], []]
        for n_id, body_frame in enumerate(f['BodyFrame']):

            if body_frame['isTracked']:

                frame_body_joints = []

                frame_body_joints.append( n_frame )
                frame_body_joints.append( f['_id'] )
                try:
                    frame_body_joints.append( body_frame['re_id'] )
                except:
                    #print '------ re_id not available ----'
                    frame_body_joints.append( '' )

                frame_body_joints.append( body_frame['leanFB'] )
                frame_body_joints.append( body_frame['leanLR'] )
                frame_body_joints.append( body_frame['leanConfidence'] )
                frame_body_joints.append( body_frame['firstTrack'] )

                # joints = []
                # counter = 0
                # for j in body_frame['skeleton'][skeletonType]:
                #     joints.append([])

                #     joints[counter].append( j['x'] )
                #     joints[counter].append( j['y'] )
                #     joints[counter].append( j['z'] )
                #     joints[counter].append( j['confidence'] )

                #     counter += 1

                # frame_body_joints.append(joints)



                for skeletonTypes in skeletonType:
                    joints = []
                    counter = 0
                    #joints.append([skeletonTypes])
                    for j in xrange(len(body_frame['skeleton'][skeletonTypes])):
                        joints.append([])

                        joints[counter].append( body_frame['skeleton'][skeletonTypes][j]['x'] )
                        joints[counter].append( body_frame['skeleton'][skeletonTypes][j]['y'] )

                        if skeletonTypes == 'rawGray':
                            ## raw gray has wrong z coordinates so I take the raw one
                            joints[counter].append( body_frame['skeleton']['raw'][j]['z'] )
                        else:
                            joints[counter].append( body_frame['skeleton'][skeletonTypes][j]['z'] )
                        joints[counter].append( body_frame['skeleton'][skeletonTypes][j]['confidence'] )

                        counter += 1

                    frame_body_joints.append(joints)

                all_joints[n_id] = frame_body_joints

        all_data[f['SensorID']].append(all_joints)


    return all_data


def read_kinect_joints_from_db(kinect_collection,time_interval):

    global begin_period,end_period
    begin_period = datetime.strptime(time_interval[0], '%Y-%m-%d %H:%M:%S')
    end_period = datetime.strptime(time_interval[1], '%Y-%m-%d %H:%M:%S')

    ##find all the data in the database
    #frame_with_joints = kinect_collection.find({})

    ## start to find from the most recent ##
    frame_with_joints =kinect_collection.find().sort([('_id',pymongo.DESCENDING)])#pymongo.DESCENDING


    joint_points = []

    for n_frame,f in enumerate(frame_with_joints):

        # takes only data within the selected time interval
        if begin_period <= f['_id'] <= end_period:

            for n_id,body_frame in enumerate(f['BodyFrame']):

                if body_frame['isTracked']:

                    frame_body_joints = np.zeros((len(body_frame['skeleton']['rawGray'])+1,3),dtype='S30')

                    #frameID
                    frame_body_joints[0,0] = n_frame
                    #time
                    frame_body_joints[0,1] = f['_id']
                    #trackID is read from re-id algorithm if it had been applied, otherwise we read kinect id
                    try:
                        frame_body_joints[0,2] = body_frame['re_id']
                    except Exception as e:
                        frame_body_joints[0,2] = n_id

                    #joints
                    i = 1
                    for j in body_frame['skeleton']['rawGray']:
                        frame_body_joints[i,0] = j['x']
                        frame_body_joints[i,1] = j['y']
                        #frame_body_joints[i,2] = j['raw']['z']
                        i+=1
                    i = 1
                    for j in body_frame['skeleton']['raw']:
                        frame_body_joints[i,2] = j['z']
                        i+=1

                    joint_points.append(frame_body_joints)

        #elif f['_id'] > end_period:
            #break
        elif f['_id'] < begin_period:
            break


    joint_points = joint_points[::-1]
    print 'retrieved trajectory matrix size: ', np.array(joint_points).shape
    return joint_points


def read_ambient_sensor_from_db(binary_collection,time_interval):

    begin_period = datetime.strptime(time_interval[0], '%Y-%m-%d %H:%M:%S')
    end_period = datetime.strptime(time_interval[1], '%Y-%m-%d %H:%M:%S')

    binary_data = []
    for data in binary_collection.find({}):

        if begin_period <= data['_id'] <= end_period:
            binary_data.append(data['Value'])

    return binary_data


def read_zenith_from_db(zenith_collection,time_interval):


    for data in zenith_collection.find({}):

        data['ColorData']
        data['Timestamp']


def read_UPMBand_from_db(band_collection,time_interval):

    upm_band_data = []
    for data in band_collection.find({}):

        upm_band_data.append(data['Value'])

    return upm_band_data


def check_arguments(time_interval, session):

    if len(time_interval) == 0:
        begin_period = ''
        end_period = ''
        if session == '':
            raise RuntimeError('Both time_interval and session cannot be empty.')

    elif len(time_interval) == 1:
        begin_period = datetime.strptime(time_interval[0], '%Y-%m-%d %H:%M:%S')
        end_period = ''

    elif len(time_interval) == 2:
        begin_period = datetime.strptime(time_interval[0], '%Y-%m-%d %H:%M:%S')
        end_period = datetime.strptime(time_interval[1], '%Y-%m-%d %H:%M:%S')

    else:
        raise RuntimeError('time_interval has more than 2 elements.')


    return begin_period, end_period


def read_data_from_db_tomas(collection, time_interval, session, exclude_columns=None):

    begin_period, end_period = check_arguments(time_interval, session)

    proj = dict()
    if exclude_columns is not None:
        for c in exclude_columns:
            proj[c] = 0

    if session == '' and len(time_interval) == 1:
        if exclude_columns is None:
            result = \
                collection.find({'_id': {'$gt': begin_period}})
        else:
            result = \
                collection.find({'_id': {'$gt': begin_period}}, projection=proj)

    elif session == '' and len(time_interval) == 2:
        if exclude_columns is None:
            result = \
                collection.find({'_id': {'$gt': begin_period, '$lt': end_period}})
        else:
            result = \
                collection.find({'_id': {'$gt': begin_period, '$lt': end_period}}, projection=proj)

    elif len(time_interval) == 0:
        if exclude_columns is None:
            result = \
                collection.find({'Session': session})
        else:
            result = \
                collection.find({'Session': session}, projection=proj)

    elif len(time_interval) == 1:
        if exclude_columns is None:
            result = \
                collection.find({'_id': {'$gt': begin_period}, 'Session': session})
        else:
            result = \
                collection.find({'_id': {'$gt': begin_period}, 'Session': session}, projection=proj)

    else:  # len(time_interval) == 2:
        if exclude_columns is None:
            result = \
                collection.find({'_id': {'$gt': begin_period, '$lt': end_period},
                                 'Session': session})
        else:
            result = \
                collection.find({'_id': {'$gt': begin_period, '$lt': end_period},
                                 'Session': session}, projection=proj)


    return result


def read_MSBand_from_db(collection, time_interval, session):

    """

    Read MSBand data from db

    ------------------------------------------------------------------------------------------
    Parameters:

    collection:
        The mongo db collection instance

    time_interval: List of string
        The begin and end time stamps for the query (e.g., '2017-10-01 23:59:59.999')

    session: string
        The session name for the query

    ------------------------------------------------------------------------------------------
    Tested: -
    ------------------------------------------------------------------------------------------

    """

    frames = read_data_from_db_tomas(collection, time_interval, session)

    all_data = dict()
    for n_frame, f in enumerate(frames):

        if all_data.has_key(f['SensorID']) == False:
            all_data[f['SensorID']] = []

        meas = []

        meas.append( n_frame )
        meas.append( f['_id'] )

        meas.append( f['Acceleration']['X'] )
        meas.append( f['Acceleration']['Y'] )
        meas.append( f['Acceleration']['Z'] )

        meas.append( f['Gyroscope']['VelX'] )
        meas.append( f['Gyroscope']['VelY'] )
        meas.append( f['Gyroscope']['VelZ'] )

        meas.append( f['HeartRate']['Value'] )
        meas.append( f['HeartRate']['Confidence'] )
        meas.append( f['HeartRate']['Time'] )

        meas.append( f['GSR']['Value'] )
        meas.append( f['GSR']['Time'] )

        meas.append( f['SensorID'] )

        all_data[f['SensorID']].append(meas)


    return all_data


def read_MSBand_from_db_asDict(collection, time_interval, session):

    """

    Read MSBand data from db

    ------------------------------------------------------------------------------------------
    Parameters:

    collection:
        The mongo db collection instance

    time_interval: List of string
        The begin and end time stamps for the query (e.g., '2017-10-01 23:59:59.999')

    session: string
        The session name for the query

    ------------------------------------------------------------------------------------------
    Tested: -
    ------------------------------------------------------------------------------------------

    """

    frames = read_data_from_db_tomas(collection, time_interval, session)

    all_data = dict()
    for n_frame, f in enumerate(frames):

        if all_data.has_key(f['SensorID']) == False:
            all_data[f['SensorID']] = []

        meas = dict()

        meas['n'] = n_frame
        meas['TS'] = f['_id']
        meas['id'] = f['SensorID']

        meas['accX'] =  f['Acceleration']['X']
        meas['accY'] =  f['Acceleration']['Y']
        meas['accZ'] =  f['Acceleration']['Z']
        meas['gyrX'] =  f['Gyroscope']['VelX']
        meas['gyrY'] =  f['Gyroscope']['VelY']
        meas['gyrZ'] =  f['Gyroscope']['VelZ']
        meas['accTS'] = f['Acceleration']['Time']

        meas['calToday'] = f['Calories']['Today']
        meas['calTotal'] = f['Calories']['Total']
        meas['calTS'] =    f['Calories']['Time']

        meas['distPace'] =  f['Distance']['Pace']
        meas['distSpeed'] = f['Distance']['Speed']
        meas['dist'] =      f['Distance']['Today']
        meas['distTS'] =    f['Distance']['Time']

        meas['hr'] =           f['HeartRate']['Value']
        meas['hrConfidence'] = f['HeartRate']['Confidence']
        meas['hrTS'] =         f['HeartRate']['Time']

        meas['pedToday'] = f['Pedometer']['Today']
        meas['pedTotal'] = f['Pedometer']['Total']
        meas['pedTS'] =    f['Pedometer']['Time']

        meas['temp'] =   f['SkinTemp']['Value']
        meas['tempTS'] = f['SkinTemp']['Time']

        meas['gsr'] =   f['GSR']['Value']
        meas['gsrTS'] = f['GSR']['Time']

        all_data[f['SensorID']].append(meas)


    return all_data


def save_classifier_model(model,filename):
    print 'save model in db'


def get_classifier_model(filename):
    print 'get classifier model from db'


def save_feature_matrix(f_matrix,filename):
    print 'save feature matrix in db'


def get_feature_matrix(filename):
    print 'get feature matrix from db'


def save_img(filename):
    print 'save images on db'


def get_img(filename):
    print 'get img from db'


def write_output_file(participant_ID, content, path):

    day ='19-4-2017'
    file_name = path +'participantID' + str(participant_ID) + '_' + day + '.txt'

    file = open(file_name, 'w')
    ##participant name

    ##functionalities for every patient ID
    for c in range(1, len(content),2):
        file.write(content[c-1])
        file.write('\t')
        file.write(str(content[c]))
        file.write('\n')


def load_matrix_pickle(path):
    with open(path, 'rb') as handle:
        file = cPickle.load(handle)
    return file

def save_matrix_pickle(file, path):
    with open(path, 'wb') as handle:
        cPickle.dump(file,handle, protocol=2)
    return file


def summary_MSBand(db, time_interval):

    """

    Summarize biological measurements for a specific time period


    """

    colMSBand = db.MSBand

    timeStart = time_interval[0]
    timeEnd = time_interval[1]

    d_all = read_MSBand_from_db_asDict(collection=colMSBand,
                                       time_interval=[timeStart, timeEnd],
                                       session='')

    out_hbr = dict()
    out_gsr = dict()
    for key in d_all.keys():

        d = d_all[key]

        if key == '':
            print 'Discarding {0} measurements with invalid key \'{1}\''.format(len(d), key)
            continue

        hr = []
        gsr = []
        for i in range(len(d)):

            if d[i]['hr'] > 0 and d[i]['hrConfidence'] > 0:
                hr.append(d[i]['hr'])

            if d[i]['gsr'] > 0 and d[i]['gsr'] < 200000:
                gsr.append(d[i]['gsr'])

        hr = np.array(hr)
        gsr = np.array(gsr)

        result_hr = {'start': timeStart,
                     'end': timeEnd,
                     'n': 0,
                     'min': 0,
                     '25': 0,
                     '50': 0,
                     '75': 0,
                     'max': 0}

        result_gsr = {'start': timeStart,
                     'end': timeEnd,
                     'n': 0,
                     'min': 0,
                     '25': 0,
                     '50': 0,
                     '75': 0,
                     'max': 0}

        if hr.shape[0] > 0:
            result_hr['n'] = int(hr.shape[0])
            result_hr['min'] = np.min(hr)
            result_hr['25'] = np.percentile(hr, 25)
            result_hr['50'] = np.percentile(hr, 50)
            result_hr['75'] = np.percentile(hr, 75)
            result_hr['max'] = np.max(hr)

        if gsr.shape[0] > 0:
            result_gsr['n'] = int(gsr.shape[0])
            result_gsr['min'] = np.min(gsr)
            result_gsr['25'] = np.percentile(gsr, 25)
            result_gsr['50'] = np.percentile(gsr, 50)
            result_gsr['75'] = np.percentile(gsr, 75)
            result_gsr['max'] = np.max(gsr)

        out_hbr[key] = result_hr
        out_gsr[key] = result_gsr


    return out_hbr, out_gsr


#personMSBand = "MSFT Band UPM f6:65"
def write_summarization_nonrealtime_f_json(kinect_motion_amount, day_motion_as_activation, night_motion_as_activation,freezing_analysis,festination_analysis,\
 loss_of_balance_analisys, fall_down_analysis, nr_visit, confusion_analysis, lh_number, lhc_number, heart_rate_low, heart_rate_high):

    path_to_lambda = "C:\\libs3\\"
    client = Connection('localhost', 27017)
    dbIDs = client['local']['BandPersonIDs']
    uuids = dbIDs.find()
    for uuid_person in uuids:
        final_sumarization = {'patientID': uuid_person["PersonID"],"date": time.strftime("%Y-%m-%d"),"daily_motion": kinect_motion_amount[uuid_person["SensorID"]], "as_day_motion": day_motion_as_activation[uuid_person["SensorID"]], "as_night_motion": night_motion_as_activation[uuid_person["SensorID"]], "freezing": freezing_analysis[uuid_person["SensorID"]], "festination": festination_analysis[uuid_person["SensorID"]], "loss_of_balance": loss_of_balance_analisys[uuid_person["SensorID"]], "fall_down": fall_down_analysis[uuid_person["SensorID"]], "visit_bathroom": nr_visit[uuid_person["SensorID"]], "confusion_behaviour_detection": confusion_analysis[uuid_person["SensorID"]], "leave_the_house": lh_number[uuid_person["SensorID"]], "leave_house_confused": lhc_number[uuid_person["SensorID"]], "heart_rate_low": heart_rate_low[uuid_person["SensorID"]], "heart_rate_high": heart_rate_high[uuid_person["SensorID"]]}

        date_in_title = time.strftime("%Y-%m-%d").split('-')
        filename_json = path_to_lambda + uuid_person["PersonID"] + '_' + date_in_title[0]+date_in_title[1]+date_in_title[2]+ '.json'
        with open(filename_json, 'w') as outfile:
            json.dump(final_sumarization, outfile)

        fileManager = S3FileManager.S3FileManger('hetra/out', '')
        fileManager.upload_file(filename_json, uuid_person["PersonID"] + '_' + date_in_title[0]+date_in_title[1]+date_in_title[2]+ '.json')
        print 'json summarization uploaded to the amazon web server!'


def get_bands_ID(db):
    bands_collection = db.BandPersonIDs.find()

    band_ids= []
    for b in bands_collection:
        print 'band in use: ', b['SensorID'],' uuid: ', b['PersonID']
        band_ids.append([b['SensorID'],b['PersonID']])

    return band_ids



