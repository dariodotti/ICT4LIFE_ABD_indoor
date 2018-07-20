from pymongo import MongoClient as Connection
import pymongo
import numpy as np
from datetime import datetime
import cPickle
import multiprocessing
import time
import json
import pandas as pd
from scipy import stats
from lib_amazon_web_server import S3FileManager
#from multiprocessing.dummy import Pool as threadPool
import re
from time import mktime as mktime

import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt


begin_period = ''
end_period = ''


def connect_db(name_db, ip='localhost'):

    con = Connection(ip, 27017)
    db = con[name_db]
    print db

    return db


def read_events(db, time_interval):

    # The following assumptions are made for the captured events
    #   - each event is associated with only one sensor (e.g., LossOfBalance <=> Kinect)
    #   - each event is registered only once (2 Kinects cannot report the same LossOfBalance event)
    #   - the events are deleted at the end of each day

    if time_interval == None:
        r = db.rtevents.find({}).sort('TimeStamp', 1) # 1 = ASCENDING
    else:
        assert len(time_interval) == 2, 'Two dates are required'
        d1 = datetime.strptime(time_interval[0], '%Y-%m-%d')
        d2 = datetime.strptime(time_interval[1], '%Y-%m-%d')
        assert d1 <= d2, 'The starting date cannot be greater than the ending date'

        dates = [d1]
        while d1 < d2:
            d1 = d1 + timedelta(days=1)
            dates.append(d1)

        dates = [datetime.strftime(d, '%Y-%m-%d') for d in dates]

        s = ''
        for d in dates:
            s = s + '|^' + d

        s = s[1:]
        regex = re.compile(s)

        r = db.rtevents.find({'TimeStamp': regex}).sort('TimeStamp', 1) # 1 = ASCENDING

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
                                             exclude_columns=['ColorImage','DepthImage',
                                                              'InfraImage','BodyImage'])
            reid = ''
            if e['SensorID'] in d_all.keys():
                d = d_all[e['SensorID']]
                for j in range(len(d)):
                    if d[j][e['BodyID']] != [] and d[j][e['BodyID']][2] != '':
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

            ## Adding additional value for heart rate measurements
            if e['Event'] == 'HeartRateHigh' or e['Event'] == 'HeartRateLow':

                try: # since the additional value has been added late, most of the DB istances do not have it
                    events[e['Event']][e['reid']].append([dt, e['Duration'],e['Sensor'],e['Value']])
                except:
                    events[e['Event']][e['reid']].append([dt, e['Duration'],e['Sensor']])

            else:
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


def summarize_events(db, time_interval=None):

    # The following assumptions are made for the captured events
    #   - if timestamp + duration of event(t) = timestamp of event(t+1),
    #     the 2 events are combined into 1

    events = read_events(db, time_interval)

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

                    if e_type == 'HeartRateHigh' or e_type == 'HeartRateLow':

                        try:# since the additional value has been added late, most of the DB istances do not have it
                            v = event_summary[e_type][rID][i][3]
                            e.append({"beginning": ts, "duration": str(dur),"value": str(v)})
                        except:
                            e.append({"beginning": ts, "duration": str(dur)})

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

        if 'BodyFrame' not in f.keys():
            continue

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
						#joints[counter].append( body_frame['skeleton'][skeletonTypes][j]['z'] )
						#joints[counter].append( body_frame['skeleton'][skeletonTypes][j]['confidence'] )

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
    #frame_with_joints =kinect_collection.find().sort([('_id',pymongo.DESCENDING)])#pymongo.ASCENDING
    frame_with_joints = kinect_collection.find({'_id':{'$lt': end_period, '$gte': begin_period}})
    
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

#        elif f['_id'] > end_period:
#            break
        #elif f['_id'] < begin_period:
            #joint_points = joint_points[::-1]
    
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


def convert_timestamp(ts):
    tsD = ts.split(" ")
    tsH = tsD[1].split(".")
    tsHs = tsH[0].split(":")
    tsY = tsD[0].split("-")
    if len(tsH) == 2:
        milis = float("0."+tsH[1])
    else:
        milis = 0.0
    fechaIn = datetime(int(tsY[0]), int(tsY[1]), int(tsY[2]), 0, 0)
    fechaIn = mktime(datetime.timetuple(fechaIn))
    dateFin = fechaIn+int(tsHs[0])*60*60+int(tsHs[1])*60+int(tsHs[2])+milis
    return dateFin

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y

def extract_one_sensor(db, time_interval):
    imuData = read_data_from_db_tomas(collection=db.MSBand, time_interval=[time_interval[0], time_interval[1]], session='')
    oneImuData = []
    for data in imuData:
        if data['Contact']['Value'] == False:
            continue
        oneImuData.append([[data["Acceleration"]["X"],data["Acceleration"]["Y"],data["Acceleration"]["Z"]],convert_timestamp(str(data["_id"]))])
    return oneImuData

def learning_count_jerks_pace(imuData):
    # Filter requirements.
    order = 5
    fs = 90.0       # sample rate, Hz
    cutoff = 3 # desired cutoff frequency of the filter, Hz

    grav = 9.80665

    accelData = []
    timestamps = []
    accelDataNoFilter = []
    for i in range(0,len(imuData)):
        accelData.append([np.sqrt(imuData[i][0][0]**2+imuData[i][0][1]**2+imuData[i][0][2]**2),imuData[i][1]])
        accelDataNoFilter.append(np.sqrt(imuData[i][0][0]**2+imuData[i][0][1]**2+imuData[i][0][2]**2))
        timestamps.append(imuData[i][1])

    accelDataFilter = butter_lowpass_filter(accelDataNoFilter, cutoff, fs, order)

    for i in range(0,len(accelDataFilter)):
        accelData[i][0] = accelDataFilter[i]
    #show_graphs_filter(accelDataNoFilter, timestamps, cutoff, fs, order)

    PACE = 0.5
    GRAVITY = 0.98
    last_state = None
    current_state = None
    last_peak = None
    last_trough = None
    peaks = []
    troughs = []
    middles = []
    zero = GRAVITY
    jerk_mean = 0.3
    pace_mean = PACE
    alpha = 0.125
    jerk_dev = 0.125
    pace_dev = 0.1 #pace_mean/0.4
    beta = 0.125
    meta = []

    for i in range(0,len(accelData)):
        if accelData[i][0] < zero and last_state is not None:
            current_state = 'trough'
        elif accelData[i][0] > zero:
            current_state = 'peak'
        if current_state is not last_state:
            if last_state is 'trough':
                if last_peak and last_trough:
                    jerk =  last_peak['val'] - last_trough['val']
                    if jerk > jerk_mean - 4*jerk_dev:
                        pace = pace_mean
                        if len(peaks) > 1 :
                            pace = peaks[-1]['ts']-peaks[-2]['ts']
                        if pace > pace_mean - 3 * pace_dev and  pace < pace_mean + 5 * pace_dev:
                            troughs.append(last_trough)
                            jerk_dev = abs(jerk_mean - jerk) * beta + jerk_dev * (1 - beta)
                            jerk_mean = jerk * alpha + jerk_mean * (1- alpha)
                            pace_dev = abs(pace_mean - pace) * beta + pace_dev * (1- beta)
                            pace_mean = pace * alpha + pace_mean * (1- alpha)
                            meta.append([accelData[i][1], jerk_mean, jerk_dev, pace_mean, pace_dev])
                            first_event = min(last_peak['index'], last_trough['index'])
                            second_event = max(last_peak['index'], last_trough['index'])
                            last_index = int((second_event - first_event)/2) + first_event
                        #else:
                            #print "PACE FAIL", pace_mean, pace
                    #else:
                        #print "STEP FAIL"
                last_trough = None
                last_peak = None
            elif last_state is 'peak':
                peaks.append(last_peak)

        if current_state is 'trough' and (last_trough is None or accelData[i][0] < last_trough["val"]):
            last_trough = {"ts": accelData[i][1], "val": accelData[i][0], "index": i, "min_max":"min"}
        elif current_state is 'peak' and (last_peak is None or accelData[i][0] > last_peak["val"]):
            last_peak = {"ts": accelData[i][1], "val": accelData[i][0], "index": i, "min_max":"max"}

        last_state = current_state

    return np.array(peaks), np.array(troughs), np.array(accelDataFilter), np.array(timestamps)

def vel_estimator(peaks,troughs,r,timestamps):
    peak_ts =  [peak['ts'] for peak in peaks]
    peak_val =  [peak['val'] for peak in peaks]
    peak_ind =  [peak['index'] for peak in peaks]
    trough_ts =  [trough['ts'] for trough in troughs]
    trough_val =  [trough['val'] for trough in troughs]
    trough_ind =  [trough['index'] for trough in troughs]
    r = r.tolist()
    timestamps = timestamps.tolist()

    k = 0.38
    indP = 0
    indT = 0
    pico = [0,0]
    subpic = [0,0]
    picDet = False
    subir = False
    velEst = []
    velDef = 80
    n_steps = 0

    for i in range(0,len(r)):
        addVel = False
        try:
            if i == peak_ind[indP] and picDet == False:
                pico = [peak_val[indP],peak_ts[indP]]
                picDet = True
                indP += 1
            elif i == trough_ind[indT] and picDet == True:
                subpic = [trough_val[indT],trough_ts[indT]]
                picDet = False
                indT += 1

            if pico[0] != 0.0 and subpic[0] != 0.0:
                step = k*(pico[0]-subpic[0])**0.25
                if subir == False:
                    vel = step/(subpic[1]-pico[1])*100
                    velEst.append([vel,timestamps[i]])
                    n_steps += 1
                    addVel = True
                else:
                    vel = step/(pico[1]-subpic[1])*100
                    velEst.append([vel,timestamps[i]])
                    n_steps += 1
                    addVel = True
                if subir == False:
                    pico = [0,0]
                    subir = True
                else:
                    subpic = [0,0]
                    subir = False
            else:
                addVel = True
                if len(velEst) == 0:
                    velEst.append([velDef,timestamps[i]])
                else:
                    velEst.append([velEst[-1][0],timestamps[i]])

            if i>peak_ind[indP] or i>trough_ind[indT]:
                indP += 1
                indT += 1
                picDet = False
                pico = [0,0]
                subPic = [0,0]
                subir = False

        except:
            if addVel == False:
                if len(velEst) == 0:
                    velEst.append([velDef,timestamps[i]])
                else:
                    velEst.append([velEst[-1][0],timestamps[i]])
            continue

    alpha = 0.9
    for n in range(len(velEst)):
        if n != 0:
            velEst[n][0] = (1-alpha)*velEst[n][0]+alpha*velEst[n-1][0]  

    return velEst, n_steps


def steps_accelerometer_method(db, time_interval):
    print("Steps alternative method")

    imuDataAcc = extract_one_sensor(db, time_interval)
    peaks, troughs, accelDataFilter, timestamps = learning_count_jerks_pace(imuDataAcc)
    vel, n_steps = vel_estimator(peaks,troughs,accelDataFilter,timestamps)
    print(n_steps)

    return n_steps


def summary_steps(db, time_interval, step_interval_mins):

    """

    Summarize the steps taken during a specific time period
    every 'step_interval_mins' minutes


    """

    colMSBand = db.MSBand

    timeStart = time_interval[0]
    timeEnd = time_interval[1]

    d_all = read_MSBand_from_db_asDict(collection=colMSBand,
                                       time_interval=[timeStart, timeEnd],
                                       session='')

    out_steps = dict()

    dbIDs = db.BandPersonIDs
    uuids = dbIDs.find()
    for key in uuids:
        key = key["SensorID"]
        try:
            d = d_all[key]

            if key == '':
                print 'Discarding {0} measurements with invalid key \'{1}\''.format(len(d), key)
                continue
        except:
            d = []

        steps = []
        c = 0
        timeStart = ''
        timeEnd   = ''
        stepsStart = 0
        stepsEnd   = 0
        while c < len(d):

            if d[c]['pedTotal'] > 0:

                if timeStart == '':
                    timeStart  = d[c]['pedTS']
                    stepsStart = d[c]['pedTotal']
                else:
                    timeEnd  = d[c]['pedTS']
                    stepsEnd = d[c]['pedTotal']
                    elapsed_mins = (timeEnd - timeStart).total_seconds() / 60.
                    if elapsed_mins >= step_interval_mins:
                        steps.append([int(elapsed_mins), int(stepsEnd) - int(stepsStart)])
                        timeStart = ''
                        timeEnd   = ''
                        stepsStart = 0
                        stepsEnd   = 0

            c += 1

        if len(steps) == 0:
            steps = steps_accelerometer_method(db, time_interval)
        else:
            steps = steps[0][1]

        out_steps[key] = steps


    return out_steps


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
    
    client = Connection('localhost', 27017)
    dbIDs = client['local']['BandPersonIDs']
    uuids = dbIDs.find()
    
    for key in uuids:
        key = key["SensorID"]
        
        try:
            d = d_all[key]

            if key == '':
                print 'Discarding {0} measurements with invalid key \'{1}\''.format(len(d), key)
                continue
    
            hr = []
            gsr = []
            for i in range(len(d)):
    
                if d[i]['hr'] > 0 and d[i]['hrConfidence'] > 0: ## confidence, if it is smaller than 0 the band is guessing
                    hr.append(d[i]['hr'])
    
                if d[i]['gsr'] > 0 and d[i]['gsr'] < 1000000: ##  increase the threshold if too low
                    gsr.append(d[i]['gsr'])
        except:
            hr = []
            gsr = []
        
        
        hr = np.array(hr)
        gsr = np.array(gsr)

        result_hr = {'start': timeStart,
                     'end': timeEnd,
                     'n': 0,
                     'mean':0,
                     'mode':0,
                     'median':0,
                     'skew':0,
                     'kurtosis':0,
                     'min': 0,
                     '25': 0,
                     '50': 0,
                     '75': 0,
                     'max': 0}

        result_gsr = {'start': timeStart,
                     'end': timeEnd,
                     'n': 0,
                     'mean':0,
                     'mode':0,
                     'median':0,
                     'skew':0,
                     'kurtosis':0,
                     'min': 0,
                     '25': 0,
                     '50': 0,
                     '75': 0,
                     'max': 0}

        if hr.shape[0] > 0:
            
            result_hr['n'] = int(hr.shape[0])
            result_hr['mean'] = float(np.mean(hr))
            result_hr['mode'] = float(stats.mode(hr)[0][0])
            result_hr['median'] = float(np.median(hr))
            result_hr['skew'] = float(stats.skew(hr))
            result_hr['kurtosis'] = float(stats.kurtosis(hr))
            result_hr['min'] = float(np.min(hr))
            result_hr['25'] = float(np.percentile(hr, 25))
            result_hr['50'] = float(np.percentile(hr, 50))
            result_hr['75'] = float(np.percentile(hr, 75))
            result_hr['max'] = float(np.max(hr))

        if gsr.shape[0] > 0:
            result_gsr['n'] = int(gsr.shape[0])
            result_gsr['mean'] = float(np.mean(gsr))
            result_gsr['mode'] = float(stats.mode(gsr)[0][0])
            result_gsr['median'] = float(np.median(gsr))
            result_gsr['skew'] =  float(stats.skew(gsr))
            result_gsr['kurtosis'] = float(stats.kurtosis(gsr))
            result_gsr['min'] = float(np.min(gsr))
            result_gsr['25'] = float(np.percentile(gsr, 25))
            result_gsr['50'] = float(np.percentile(gsr, 50))
            result_gsr['75'] = float(np.percentile(gsr, 75))
            result_gsr['max'] = float(np.max(gsr))

        out_hbr[key] = result_hr
        out_gsr[key] = result_gsr


    return out_hbr, out_gsr


#personMSBand = "MSFT Band UPM f6:65"
def write_summarization_nonrealtime_f_json(kinect_motion_amount, day_motion_as_activation, night_motion_as_activation,freezing_analysis,festination_analysis,\
 loss_of_balance_analisys, fall_down_analysis, nr_visit, confusion_analysis, lh_number, lhc_number, heart_rate_low, heart_rate_high, gsr, hr, steps):

    path_to_lambda = "C:\\libs3\\"
    client = Connection('localhost', 27017)
    dbIDs = client['local']['BandPersonIDs']
    uuids = dbIDs.find()

    for uuid_person in uuids:

        # Check 0 values to put the correct format
        try:
            val_kma = kinect_motion_amount[uuid_person["SensorID"]]
        except:
            val_kma = {"slow_mov": -1, "stationary": -1, "fast_mov": -1}

        if day_motion_as_activation[uuid_person["SensorID"]] == 0:
            day_motion_as_activation[uuid_person["SensorID"]] = {"toilet": [], "entrance": [],"bedroom": []}

        if night_motion_as_activation[uuid_person["SensorID"]] == 0:
            night_motion_as_activation[uuid_person["SensorID"]] = {"toilet": [], "entrance": [],"bedroom": []}

        if heart_rate_low[uuid_person["SensorID"]] == 0:
            heart_rate_low[uuid_person["SensorID"]] = []

        if heart_rate_high[uuid_person["SensorID"]] == 0:
            heart_rate_high[uuid_person["SensorID"]] = []

        if festination_analysis[uuid_person["SensorID"]] == 0:
            festination_analysis[uuid_person["SensorID"]] = []

        if freezing_analysis[uuid_person["SensorID"]] == 0:
            freezing_analysis[uuid_person["SensorID"]] = []

        if fall_down_analysis[uuid_person["SensorID"]] == 0:
            fall_down_analysis[uuid_person["SensorID"]] = []

        if confusion_analysis[uuid_person["SensorID"]] == 0:
            confusion_analysis[uuid_person["SensorID"]] = []

        if nr_visit[uuid_person["SensorID"]] == 0:
            nr_visit[uuid_person["SensorID"]] = []

        if loss_of_balance_analisys[uuid_person["SensorID"]] == 0:
            loss_of_balance_analisys[uuid_person["SensorID"]] = []

        final_sumarization = {'patientID': uuid_person["PersonID"],"date": time.strftime("%Y-%m-%d"),"daily_motion": val_kma, \
        "as_day_motion": day_motion_as_activation[uuid_person["SensorID"]], "as_night_motion": night_motion_as_activation[uuid_person["SensorID"]], \
        "freezing": freezing_analysis[uuid_person["SensorID"]], "festination": festination_analysis[uuid_person["SensorID"]], \
        "loss_of_balance": loss_of_balance_analisys[uuid_person["SensorID"]], "fall_down": fall_down_analysis[uuid_person["SensorID"]], \
        "visit_bathroom": nr_visit[uuid_person["SensorID"]], "confusion_behavior_detection": confusion_analysis[uuid_person["SensorID"]], \
        "leave_the_house": lh_number[uuid_person["SensorID"]], "leave_house_confused": lhc_number[uuid_person["SensorID"]], \
        "heart_rate_low": heart_rate_low[uuid_person["SensorID"]], "heart_rate_high": heart_rate_high[uuid_person["SensorID"]],\
        "gsr": gsr[uuid_person["SensorID"]], "hr": hr[uuid_person["SensorID"]], "steps": steps[uuid_person["SensorID"]]}

        date_in_title = time.strftime("%Y-%m-%d").split('-')
        filename_json = path_to_lambda + uuid_person["PersonID"] + '_' + date_in_title[0]+date_in_title[1]+date_in_title[2]+ '.json'
        #filename_json = path_to_lambda + uuid_person["PersonID"] + '_20180601.json'
        with open(filename_json, 'w') as outfile:
            json.dump(final_sumarization, outfile)

        fileManager = S3FileManager.S3FileManger('hetra/out', '')
        fileManager.upload_file(filename_json, uuid_person["PersonID"] + '_' + date_in_title[0]+date_in_title[1]+date_in_title[2]+ '.json')
        print uuid_person["SensorID"], ' json summarization uploaded to the amazon web server!'


def get_bands_ID(db):
    bands_collection = db.BandPersonIDs.find()

    band_ids= []
    for b in bands_collection:
        #print 'band in use: ', b['SensorID'],' uuid: ', b['PersonID']
        band_ids.append([b['SensorID'],b['PersonID']])

    return band_ids



