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
    for i, e in enumerate(r):

        # split events based on event type
        if events.has_key(e['Event']) == False:
            events[e['Event']] = dict()

        # further split events by sensorID
        if events[e['Event']].has_key(e['SensorID']) == False:
            events[e['Event']][e['SensorID']] = dict()

        # further split events by BodyID
        if events[e['Event']][e['SensorID']].has_key(e['BodyID']) == False:
            events[e['Event']][e['SensorID']][e['BodyID']] = []

        # format events as timestamp, duration, sensor
        dt = datetime.strptime(e['TimeStamp'], "%Y-%m-%d %H:%M:%S")
        events[e['Event']][e['SensorID']][e['BodyID']].append(
            [dt, e['Duration'], e['Sensor']])


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
    #   - the bodyID of a person stays the same (to be solved by re-id)

    events = read_events(db)

    # join subsequent events together
    for e_type in events.keys(): # for each event type
        for sID in events[e_type].keys(): # for each sensorID
            for bID in events[e_type][sID].keys(): # for each bodyID
                for i in range(len(events[e_type][sID][bID]) - 1): # for each event
                    e_t = events[e_type][sID][bID][i]
                    e_tp1 = events[e_type][sID][bID][i + 1]
                    if e_t[0] + timedelta(seconds=e_t[1]) == e_tp1[0]:
                        # the 2 events can be joined into 1
                        events[e_type][sID][bID][i + 1][0] = events[e_type][sID][bID][i][0]
                        events[e_type][sID][bID][i + 1][1] += events[e_type][sID][bID][i][1]
                        events[e_type][sID][bID][i] = None

    # save the summarized events
    event_summary = []
    for e_type in events.keys(): # for each event type
        for sID in events[e_type].keys(): # for each sensorID
            for bID in events[e_type][sID].keys(): # for each bodyID
                for i in range(len(events[e_type][sID][bID])): # for each event
                    if events[e_type][sID][bID][i] is not None:
                        events[e_type][sID][bID][i].append(e_type)
                        events[e_type][sID][bID][i].append(bID)
                        event_summary.append(events[e_type][sID][bID][i])

    # format events into the desired output
    event_summary = np.array(event_summary)
    unique_bIDs = np.unique(event_summary[:, 4]).astype(int)
    output = {bID: {} for bID in unique_bIDs}
    for bID in unique_bIDs: # for each person
        e_per_bID = event_summary[event_summary[:, 4] == bID]
        for e_type in np.unique(e_per_bID[:, 3]).astype(str): # for each event type
            n_events = np.sum(e_per_bID[:, 3] == e_type)
            e = []
            for i in range(n_events):
                e_per_bID_per_e_type = e_per_bID[e_per_bID[:, 3] == e_type]
                ts = e_per_bID_per_e_type[i][0].strftime("%H:%M:%S")
                dur = e_per_bID_per_e_type[i][1]
                e.append({"beginning": ts, "duration": str(dur)})
            if n_events > 0:
                output[bID][e_type] = dict([("number", n_events), ("event", e)])


    return output


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


# def multithread_joints_from_db(n_frame,f):
#
#     # while in_queue.get()[1] != 'STOP':
#     #     print in_queue.get()[1]
#     #
#     #     f = in_queue.get()[0]
#     #     n_frame = in_queue.get()[1]
#     print n_frame
#     shared_list = []
#     # takes only data within the selected time interval
#     for n_id, body_frame in enumerate(f['BodyFrame']):
#
#         if body_frame['isTracked']:
#
#             frame_body_joints = np.zeros((len(body_frame['skeleton']['rawGray']) + 1, 3), dtype='S30')
#
#             # frameID
#             frame_body_joints[0, 0] = n_frame
#             # time
#             frame_body_joints[0, 1] = f['_id']
#             # trackID
#             frame_body_joints[0, 2] = n_id
#
#             # joints
#             i = 1
#             for j in body_frame['skeleton']['rawGray']:
#                 frame_body_joints[i, 0] = j['x']
#                 frame_body_joints[i, 1] = j['y']
#                 frame_body_joints[i, 2] = j['z']
#                 i += 1
#             shared_list.append(frame_body_joints)
#
#     return shared_list


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
                try:
                    frame_body_joints.append( body_frame['re_id'] )
                except:
                    #print '------ re_id not available ----'
                    frame_body_joints.append( f['_id'] )
                
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


def read_kinect_joints_from_db(kinect_collection,time_interval,multithread):

    global begin_period,end_period
    begin_period = datetime.strptime(time_interval[0], '%Y-%m-%d %H:%M:%S')
    end_period = datetime.strptime(time_interval[1], '%Y-%m-%d %H:%M:%S')

    ##find all the data in the database
    #frame_with_joints = kinect_collection.find({})

    ## start to find from the most recent ##
    frame_with_joints =kinect_collection.find().sort([('_id',pymongo.DESCENDING)])#pymongo.DESCENDING


    joint_points = []

    if multithread:

        ##TODO create multi threading process
        print 'still no multithreading'

        #manager = multiprocessing.Manager()
        ##list shared by the workers to save te data
        # worker1 = manager.Queue()
        #
        # joint_points = manager.list()
        #
        # pool = multiprocessing.Pool(processes=4)
        #
        # pool.apply_async(multithread_joints_from_db,args=(worker1,joint_points))
        #
        #
        # for n_frame,f in enumerate(frame_with_joints):
        #
        #     if f['_id'] > end_period:
        #         worker1.put(['STOP','STOP'])
        #         break
        #
        #     elif begin_period <= f['_id'] <= end_period:
        #         worker1.put([f,n_frame])
        #
        # print 'pool close'
        # worker1.put(['STOP', 'STOP'])
        # pool.close()
        # pool.join()

    else:
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
                        #trackID
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


def summary_MSBand(db, inpdate=None):

    """

    Read MSBand data from db

    ------------------------------------------------------------------------------------------
    Parameters:

    db:
        The mongo db database

    date: empty (-> today)  or vector
        requested date


    ------------------------------------------------------------------------------------------
    Example: summary_MSBand('localhost', date=None):
    ------------------------------------------------------------------------------------------

    """

    colMSBand = db.MSBand

    if inpdate==None:
        date = datetime.now()
    else:
        date = datetime(inpdate[0], inpdate[1], inpdate[2], 0, 0, 0)

    timeStart = date.strftime("%Y-%m-%d 00:00:00")
    timeEnd = date.strftime("%Y-%m-%d 23:59:59")

    d_all = read_MSBand_from_db_asDict(collection=colMSBand,
												time_interval=[timeStart, timeEnd],
												session='')

    for key in d_all.keys():

        d = d_all[key]

        if key == '':
            print 'Discarding {0} measurements with invalid key \'{1}\''.format(len(d), key)
            continue

        # Remove duplicates and detect when minute/hour is changing
        AllHBR_S = []
        AllGSR_S = []
        lastHBRTimestamp = []
        lastGSRTimestamp = []

        qq=timedelta(seconds=-1)

        SensorIDs = dict()

        SensorIDs[key] = len(SensorIDs)
        AllHBR_S.append(pd.DataFrame())
        AllGSR_S.append(pd.DataFrame())
        lastHBRTimestamp.append(d[0]['hrTS'] + qq)
        lastGSRTimestamp.append(d[0]['gsrTS'] + qq)
        numberOfBands = len(SensorIDs)

        start_time = time.time()
        print "measuring"
        for i in range(len(d)):

            HBR_NMI = False
            HBR_NHI = False
            GSR_NMI = False
            GSR_NHI = False

            Band = SensorIDs[key]
            if d[i]['hrTS']!=lastHBRTimestamp[Band]:
                if d[i]['hrTS'].minute!=lastHBRTimestamp[Band].minute:
                    HBR_NMI=True
                if d[i]['hrTS'].hour!=lastHBRTimestamp[Band].hour:
                    HBR_NHI=True

                if d[i]['hr'] > 0 and d[i]['hrConfidence'] > 0:
                    AllHBR_S[Band] = AllHBR_S[Band].append({'HBR Time': d[i]['hrTS'],'New Minute Index':HBR_NMI,'New Hour Index':HBR_NHI, 'HBR value': d[i]['hr']}, ignore_index=True)
                    lastHBRTimestamp[Band] = d[i]['hrTS']

            if d[i]['gsrTS']!=lastGSRTimestamp:
                if d[i]['gsrTS'].minute!=lastGSRTimestamp[Band].minute:
                    GSR_NMI=True
                if d[i]['gsrTS'].hour!=lastGSRTimestamp[Band].hour:
                    GSR_NHI=True

                if d[i]['gsr'] > 0 and d[i]['gsr'] < 200000:
                    AllGSR_S[Band] = AllGSR_S[Band].append({'GSR Time': d[i]['gsrTS'],'New Minute Index':GSR_NMI,'New Hour Index':GSR_NHI, 'GSR value': d[i]['gsr']}, ignore_index=True)
                    lastGSRTimestamp[Band] = d[i]['gsrTS']

        elapsed_time = time.time() - start_time
        print elapsed_time

        t=list(SensorIDs.keys())
        BandNames = list(SensorIDs.keys())
        for i in range(len(t)):
            BandNames[SensorIDs[t[i]]] = t[i]

        import unicodedata

        for i in range(len(BandNames)):
            BandNames[i]= unicodedata.normalize('NFKD', BandNames[i]).encode('ascii','ignore')
            BandNames[i] = BandNames[i].replace(':','_')

        for band in range(numberOfBands):

            AllHBR = AllHBR_S[band]
            AllGSR = AllGSR_S[band]

            # Collect minute and hour data
            # HBR
            ww=AllHBR.loc[AllHBR['New Minute Index']==1.0]
            HDR_min_idx = np.array(ww.index.tolist())

            ww=AllHBR.loc[AllHBR['New Hour Index']==1.0]
            HDR_hour_idx=np.zeros([1,1])
            if ww.empty==0:
                HDR_hour_idx = np.append(HDR_hour_idx,ww.index)
            HDR_hour_idx = np.append(HDR_hour_idx,AllHBR.shape[0])

            # GSR
            ww=AllGSR.loc[AllGSR['New Minute Index']==1.0]
            GSR_min_idx = np.array(ww.index.tolist())

            ww=AllGSR.loc[AllGSR['New Hour Index']==1.0]
            GSR_hour_idx=np.zeros([1,1])
            if ww.empty==0:
                GSR_hour_idx = np.append(GSR_hour_idx,ww.index)
            GSR_hour_idx = np.append(GSR_hour_idx,AllGSR.shape[0])

            HBR_PerMin =[]
            m=AllHBR.loc[HDR_min_idx]['HBR value']
            HBR_PerMin.append(m.tolist())
            m=AllHBR.loc[HDR_min_idx]['HBR Time'].dt.hour
            HBR_PerMin.append(m.tolist())
            m=AllHBR.loc[HDR_min_idx]['HBR Time'].dt.minute
            HBR_PerMin.append(m.tolist())

            GSR_PerMin =[]
            m=AllGSR.loc[GSR_min_idx]['GSR value']
            GSR_PerMin.append(m.tolist())
            m=AllGSR.loc[GSR_min_idx]['GSR Time'].dt.hour
            GSR_PerMin.append(m.tolist())
            m=AllGSR.loc[GSR_min_idx]['GSR Time'].dt.minute
            GSR_PerMin.append(m.tolist())

            # Statistics per hour
            # HBR
            timestamps_HBR=[]
            stats_HBR =[]
            for i in range(len(HDR_hour_idx)-1):
                hour_HBR_data = AllHBR.loc[HDR_hour_idx[i]:HDR_hour_idx[i+1]]['HBR value']
                s=[]
                t = hour_HBR_data.describe()
                s.append([t['50%'], t['25%'],t['75%'],2.5*t['25%']-1.5*t['75%'],2.5*t['75%']-1.5*t['25%']])
                outlier_low  = hour_HBR_data.loc[hour_HBR_data<s[0][3]].tolist()
                outlier_high = hour_HBR_data.loc[hour_HBR_data>s[0][4]].tolist()
                s.append(outlier_low)
                s.append(outlier_high)
                stats_HBR.append(s)
                timestamps_HBR.append(AllHBR.loc[HDR_hour_idx[i]]['HBR Time'])

            # GSR
            timestamps_GSR=[]
            stats_GSR =[]
            for i in range(len(GSR_hour_idx)-1):
                hour_GSR_data = AllGSR.loc[GSR_hour_idx[i]:GSR_hour_idx[i+1]]['GSR value']
                s=[]
                t = hour_GSR_data.describe()
                s.append([t['50%'], t['25%'],t['75%'],2.5*t['25%']-1.5*t['75%'],2.5*t['75%']-1.5*t['25%']])
                outlier_low  = hour_GSR_data.loc[hour_GSR_data<s[0][3]].tolist()
                outlier_high = hour_GSR_data.loc[hour_GSR_data>s[0][4]].tolist()
                s.append(outlier_low)
                s.append(outlier_high)
                stats_GSR.append(s)
                timestamps_GSR.append(AllGSR.loc[GSR_hour_idx[i]]['GSR Time'])

            # Save to file
            # HBR (one per minute)
            filename = 'HBR_per_minute_'+str(inpdate[0])+'-'+str(inpdate[1])+'-'+str(inpdate[2])+ '_' + BandNames[band] +'.txt'
            print filename
            file = open(filename, 'w')

            ss="{'number':"+str(len(HBR_PerMin[0]))+", 'time':["
            for i in range(len(HBR_PerMin[0])):
                ss=ss+("['"+'{0:02d}:{1:02d}'.format(HBR_PerMin[1][i], HBR_PerMin[2][i])+"']")

                if i<len(HBR_PerMin[0])-1:
                    ss=ss+","
            ss=ss+"], 'HBR':"+str((HBR_PerMin[0]))+'}'

            file.write(ss)
            file.close()

            # GSR (one per minute)
            filename = 'GSR_per_minute_'+str(inpdate[0])+'-'+str(inpdate[1])+'-'+str(inpdate[2])+ '_' + BandNames[band] +'.txt'
            print filename
            file = open(filename, 'w')

            ss="{'number':"+str(len(GSR_PerMin[0]))+", 'time':["
            for i in range(len(GSR_PerMin[0])):
                ss=ss+("['"+'{0:02d}:{1:02d}'.format(GSR_PerMin[1][i], GSR_PerMin[2][i])+"']")

                if i<len(GSR_PerMin[0])-1:
                    ss=ss+","
            ss=ss+"], 'GSR':"+str((GSR_PerMin[0]))+'}'

            file.write(ss)
            file.close()

            # HBR (boxplot stats)
            filename = 'HBR_per_hour_stats_'+str(inpdate[0])+'-'+str(inpdate[1])+'-'+str(inpdate[2])+ '_' + BandNames[band] +'.txt'
            file = open(filename, 'w')

            ss="{'number':"+str(len(stats_HBR))+", 'time':["
            for i in range(len(stats_HBR)-1):
                ss=ss+ "['"+ '{0:02d}:{1:02d}'.format(timestamps_HBR[i].hour, timestamps_HBR[i].minute) + "]"
            ss=ss+ "['"+ '{0:02d}:{1:02d}'.format(timestamps_HBR[len(stats_HBR)-1].hour, timestamps_HBR[len(stats_HBR)-1].minute) + "]"

            for i in range(len(stats_HBR)):
                ss=ss+", ['50%':"+str(stats_HBR[i][0][0])
                ss=ss+", '25%':"+str(stats_HBR[i][0][1])
                ss=ss+", '75%':"+str(stats_HBR[i][0][2])
                ss=ss+", '+1.5IQR':"+str(stats_HBR[i][0][3])
                ss=ss+", '-1.5IQR':"+str(stats_HBR[i][0][4])
                ss=ss+", 'outliers_below':"+str(stats_HBR[i][1])
                ss=ss+", 'outliers_above':"+str(stats_HBR[i][2])
                ss=ss +']'

            ss=ss +'}'

            file.write(ss)
            file.close()

            # GSR (boxplot stats)
            filename = 'GSR_per_hour_stats_'+str(inpdate[0])+'-'+str(inpdate[1])+'-'+str(inpdate[2])+ '_' + BandNames[band] +'.txt'
            file = open(filename, 'w')

            ss="{'number':"+str(len(stats_GSR))+", 'time':["
            for i in range(len(stats_GSR)-1):
                ss=ss+ "['"+ '{0:02d}:{1:02d}'.format(timestamps_GSR[i].hour, timestamps_GSR[i].minute) + "]"
            ss=ss+ "['"+ '{0:02d}:{1:02d}'.format(timestamps_GSR[len(stats_GSR)-1].hour, timestamps_GSR[len(stats_GSR)-1].minute) + "]"

            for i in range(len(stats_GSR)):
                ss=ss+", ['50%':"+str(stats_GSR[i][0][0])
                ss=ss+", '25%':"+str(stats_GSR[i][0][1])
                ss=ss+", '75%':"+str(stats_GSR[i][0][2])
                ss=ss+", '+1.5IQR':"+str(stats_GSR[i][0][3])
                ss=ss+", '-1.5IQR':"+str(stats_GSR[i][0][4])
                ss=ss+", 'outliers_below':"+str(stats_GSR[i][1])
                ss=ss+", 'outliers_above':"+str(stats_GSR[i][2])
                ss=ss +']'

            ss=ss +'}'

            file.write(ss)
            file.close()


def summarize_events_certh(event_name, path):

    import numpy as np


    data = np.genfromtxt(path, delimiter=',', dtype='str')

    dataTS = data[:, 0]
    dataV = data[:, 1].astype(float)

    datetimeFormat = '%Y-%m-%d %H:%M:%S'

    events = []
    candidates = np.nonzero(dataV)[0]
    for i in range(candidates.shape[0]):
        if i == 0:
            events.append([])
            events[-1].append(candidates[i])
        else:
            if candidates[i] - events[-1][-1] == 1:
                events[-1].append(candidates[i])
            else:
                events.append([])
                events[-1].append(candidates[i])

    times = []
    durations = []
    for i in range(len(events)):
        times.append( dataTS[events[i][0]] )
        t1 = datetime.strptime(dataTS[events[i][0]], datetimeFormat)
        t2 = datetime.strptime(dataTS[events[i][-1]], datetimeFormat)
        tsec = 1 + (t2-t1).seconds
        h = tsec / 3600
        m = (tsec - h * 3600) / 60
        s = tsec - h * 3600 - m * 60
        durations.append('{0:02d}:{1:02d}:{2:02d}'.format(h, m, s))


    return '{0}: {{\'number\': {1}, \'beginning\': {2}, \'duration\': {3}}}'.format(\
                                                        event_name, len(events), times, durations)

personMSBand = "MSFT Band UPM f6:65"
def write_summarization_nonrealtime_f_json(kinect_motion_amount, day_motion_as_activation, night_motion_as_activation,freezing_analysis,festination_analysis,\
 loss_of_balance_analisys, fall_down_analysis, nr_visit, confusion_analysis,lh_number,lhc_number  ):

    path_to_lambda = "C:/libs3/"
    client = Connection('localhost', 27017)
    dbIDs = client['local']['BandPersonIDs']
    uuids = dbIDs.find()
    uuids_list=[]
    for uuid_person in ['d20d7fc0-c0eb-4d49-8551-745bc149594e',"c2c3ed00-c5fd-433f-9db7-4bf6dce11488"]:#uuids:
        if 1:#uuid["SensorID"] == personMSBand:
            #uuid_person = uuid["PersonID"]
            final_sumarization = {'patientID':uuid_person,"date": time.strftime("%Y-%m-%d"),"daily_motion": kinect_motion_amount, "as_day_motion": day_motion_as_activation,\
                "as_night_motion": night_motion_as_activation, "freezing": freezing_analysis, "festination": festination_analysis,\
                "loss_of_balance": loss_of_balance_analisys, "fall_down": fall_down_analysis, "visit_bathroom": nr_visit, \
                "confusion_behaviour_detection": confusion_analysis, "leave_the_house": lh_number, "leave_house_confused": lhc_number }
            #uuids_list.append(final_sumarization)
    
            date_in_title = time.strftime("%Y-%m-%d").split('-')
            filename_json = path_to_lambda+uuid_person + '_' + date_in_title[0]+date_in_title[1]+date_in_title[2]+ '.json'
            with open(filename_json, 'w') as outfile:
                json.dump(final_sumarization, outfile)


            fileManager = S3FileManager.S3FileManger('hetra/out', '')
            fileManager.upload_file(filename_json, \
                                    uuid_person + '_' + date_in_title[0]+date_in_title[1]+date_in_title[2]+ '.json')
            print 'json summarization uploaded to the amazon web server!'
    

    
