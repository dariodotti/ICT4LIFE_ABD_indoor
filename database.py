from pymongo import MongoClient as Connection
import numpy as np
from datetime import datetime
import cPickle
import multiprocessing
#from multiprocessing.dummy import Pool as threadPool

begin_period = ''
end_period = ''

def connect_db(name_db):

    con = Connection('localhost', 27017)
    db = con[name_db]

    return db


def multithread_joints_from_db(n_frame,f):

    # while in_queue.get()[1] != 'STOP':
    #     print in_queue.get()[1]
    #
    #     f = in_queue.get()[0]
    #     n_frame = in_queue.get()[1]
    print n_frame
    shared_list = []
    # takes only data within the selected time interval
    for n_id, body_frame in enumerate(f['BodyFrame']):

        if body_frame['isTracked']:

            frame_body_joints = np.zeros((len(body_frame['skeleton']['rawGray']) + 1, 3), dtype='S30')

            # frameID
            frame_body_joints[0, 0] = n_frame
            # time
            frame_body_joints[0, 1] = f['_id']
            # trackID
            frame_body_joints[0, 2] = n_id

            # joints
            i = 1
            for j in body_frame['skeleton']['rawGray']:
                frame_body_joints[i, 0] = j['x']
                frame_body_joints[i, 1] = j['y']
                frame_body_joints[i, 2] = j['z']
                i += 1
            shared_list.append(frame_body_joints)

    return shared_list


def read_kinect_joints_from_db(kinect_collection,time_interval,multithread):

    global begin_period,end_period
    begin_period = datetime.strptime(time_interval[0], '%Y-%m-%d %H:%M:%S')
    end_period = datetime.strptime(time_interval[1], '%Y-%m-%d %H:%M:%S')


    #frame_with_joints = kinect.find({"BodyFrame.isTracked": True})
    frame_with_joints = kinect_collection.find({})


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
                            frame_body_joints[i,2] = j['z']
                            i+=1
                        joint_points.append(frame_body_joints)

            elif f['_id'] > end_period:
                break

    #print joint_points[0]
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