from pymongo import MongoClient as Connection
import numpy as np



def connect_db(name_db):

    con = Connection('localhost', 27017)
    db = con[name_db]

    return db


def read_kinect_joints_from_db(kinect_collection,time_interval):

    #TODO: retrieve the data only in the selected time interval
    #frame_with_joints = kinect.find({"BodyFrame.isTracked": True})
    frame_with_joints = kinect_collection.find({})


    joint_points = []

    for n_frame,f in enumerate(frame_with_joints):

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

    #print joint_points[0]
    return joint_points


def read_ambient_sensor_from_db(binary_collection,time_interval):

    binary_data = []
    for data in binary_collection.find({}):

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