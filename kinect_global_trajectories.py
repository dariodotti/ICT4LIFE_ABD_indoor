import cv2
import numpy as np
from sklearn.preprocessing import normalize
from datetime import datetime, timedelta
from scipy.ndimage.filters import gaussian_filter
from lxml import etree
from collections import Counter
import time

import img_processing_kinect as my_img_proc
import database
import visualization as vis
import classifiers

kinect_max_distance=0
kinect_min_distance=0


def org_data_in_timeIntervals(skeleton_data, timeInterval_slice):
    #get all time data from the list dropping the decimal
    content_time = map(lambda line: line[0,1].split(' ')[1].split('.')[0],skeleton_data)

    #date time library

    init_t = datetime.strptime(content_time[0],'%H:%M:%S') #+ ' ' + timeInterval_slice[3]
    end_t = datetime.strptime(content_time[len(content_time)-1],'%H:%M:%S')
    x = datetime.strptime('0:0:0','%H:%M:%S')
    tot_duration = (end_t-init_t)

    #decide the size of time slices
    # size_slice= tot_duration/12
    # hours, remainder = divmod(size_slice.seconds, 3600)
    # minutes, seconds = divmod(remainder, 60)
    hours = timeInterval_slice[0]
    minutes = timeInterval_slice[1]
    seconds = timeInterval_slice[2]

    my_time_slice = timedelta(hours=hours,minutes=minutes,seconds=seconds)

    print 'time slice selected: ' + str(my_time_slice)

    #initialize list
    time_slices = []
    time_slices_append = time_slices.append

    c = (end_t-my_time_slice)
    #get data in every timeslices
    while init_t < (end_t-my_time_slice):
        list_time_interval = []
        list_time_interval_append = list_time_interval.append

        for t in xrange(len(content_time)):

            if datetime.strptime(content_time[t],'%H:%M:%S')>= init_t and datetime.strptime(content_time[t],'%H:%M:%S') < init_t + my_time_slice:
                list_time_interval_append(skeleton_data[t])

            if datetime.strptime(content_time[t],'%H:%M:%S') > init_t + my_time_slice:
                break
        #print len(list_time_interval)

        ##save time interval without distinction of part of the day
        time_slices_append(list_time_interval)

        init_t= init_t+my_time_slice


    return time_slices


def get_coordinate_points(time_slice, joint_id):

    #get all the coordinate points of head joint
    # list_points = []
    # list_points_append = list_points.append
    #
    # #get x,y,z,id
    # map(lambda line: list_points_append([line[1][0],line[1][1]]),time_slice)
    # zs = map(lambda line: float(line[1][2]),time_slice)
    # ids =map(lambda line: np.int64(line[0][2]),time_slice)

    #apply filter to cancel noise
    #x_f,y_f =my_img_proc.median_filter(list_points)

    xs = map(lambda line: float(line[joint_id][0]) ,time_slice)
    ys = map(lambda line: float(line[joint_id][1]), time_slice)
    zs = map(lambda line: float(line[joint_id][2]), time_slice)
    ids = map(lambda line: np.int64(line[0][2]), time_slice)

    x_f = gaussian_filter(xs,2)
    y_f = gaussian_filter(ys,2)


    global kinect_max_distance
    global kinect_min_distance
    kinect_max_distance = max(zs)
    kinect_min_distance = min(zs)





    return x_f,y_f,zs,ids


def occupancy_histograms_in_time_interval(my_room, list_poly, time_slices):
    # #get number of patches
    slice_col = my_img_proc.get_slice_cols()
    slice_row = my_img_proc.get_slice_rows()
    slice_depth = my_img_proc.get_slice_depth()

    my_data_temp = []
    my_data_temp_append = my_data_temp.append

    for i in xrange(0,len(time_slices)):

        ## Checking the start time of every time slice
        if len(time_slices[i])>1:
            print 'start time: %s' %time_slices[i][0][0][1].split(' ')[1].split('.')[0]
        else:
            print 'no data in this time slice'


        ## counter for every id should be empty
        track_points_counter = np.zeros((slice_col*slice_row*slice_depth))

        ##get x,y,z of every traj point after smoothing process
        x_filtered,y_filtered,zs,ids = get_coordinate_points(time_slices[i], joint_id=3)

        ## display traj on img
        #temp_img = copy.copy(my_room)
        #my_img_proc.display_trajectories(temp_img, list_poly, x_filtered, y_filtered)

        ## count the occurances of filtered point x,y in every patches
        for p in xrange(0,len(list_poly)):

            for ci in xrange(0,len(x_filtered)):
                ## 2d polygon
                if list_poly[p].contains_point((int(x_filtered[ci]),int(y_filtered[ci]))):
                    ## 3d cube close to the camera
                    if zs[ci] < (kinect_max_distance/2):

                        track_points_counter[p*2] = track_points_counter[p*2] + 1
                        continue
                    else: ## 3d cube far from the camera

                        track_points_counter[(p*2)+1] = track_points_counter[(p*2)+1] + 1
                        continue


        ## save the data of every group in the final matrix
        my_data_temp_append(track_points_counter)

    ## normalize the final matrix
    normalized_finalMatrix = np.array(normalize(np.array(my_data_temp),norm='l2'))
    print 'OC final matrix size:' , normalized_finalMatrix.shape

    return normalized_finalMatrix


def histograms_of_oriented_trajectories(list_poly,time_slices):

    cube_size = int((kinect_max_distance - kinect_min_distance)/3)


    hot_all_data_matrix = []
    hot_all_data_matrix_append = hot_all_data_matrix.append

    for i in xrange(0,len(time_slices)):
        ##Checking the start time of every time slice
        # if(len(time_slices[i])>1):
        #     print 'start time: %s' %str(time_slices[i][0].split(' ')[8])
        # else:
        #     print 'no data in this time slice'

        #get x,y,z of every traj point after smoothing process
        x_filtered,y_filtered,zs,ids = get_coordinate_points(time_slices[i], joint_id= 3)

        #initialize histogram of oriented tracklets
        hot_matrix = []

        for p in xrange(0,len(list_poly)):
            tracklet_in_cube_f = []
            tracklet_in_cube_c = []
            tracklet_in_cube_middle = []
            tracklet_in_cube_append_f = tracklet_in_cube_f.append
            tracklet_in_cube_append_c = tracklet_in_cube_c.append
            tracklet_in_cube_append_middle = tracklet_in_cube_middle.append

            for ci in xrange(0,len(x_filtered)):
                #2d polygon
                if list_poly[p].contains_point((int(x_filtered[ci]),int(y_filtered[ci]))):
                    ## 3d cube close to the camera
                    if zs[ci] <= (kinect_min_distance+cube_size):

                        tracklet_in_cube_append_c([x_filtered[ci],y_filtered[ci],ids[ci]])


                    elif zs[ci] > (kinect_min_distance+cube_size) and zs[ci] < (kinect_min_distance+(cube_size*2)):
                        tracklet_in_cube_append_middle([x_filtered[ci], y_filtered[ci], ids[ci]])

                    elif zs[ci]>= kinect_min_distance + (cube_size*2): ##3d cube far from the camera
                        tracklet_in_cube_append_f([x_filtered[ci],y_filtered[ci],ids[ci]])


            for three_d_poly in [tracklet_in_cube_c, tracklet_in_cube_middle, tracklet_in_cube_f]:
                if len(three_d_poly)>0:

                    ## for tracklet in cuboids compute HOT following paper
                    hot_single_poly = my_img_proc.histogram_oriented_tracklets(three_d_poly)

                    ## compute hot+curvature
                    #hot_single_poly = my_img_proc.histogram_oriented_tracklets_plus_curvature(three_d_poly)

                else:
                    hot_single_poly = np.zeros((24))


                ##add to general matrix
                if len(hot_matrix)>0:
                    hot_matrix = np.hstack((hot_matrix,hot_single_poly))
                else:
                    hot_matrix = hot_single_poly


        hot_all_data_matrix_append(hot_matrix)


    ## normalize the final matrix
    normalized_finalMatrix = np.array(normalize(np.array(hot_all_data_matrix),norm='l2'))

    ##add extra bin containing time


    ##return patinet id
    patient_id = ids[0]

    print 'HOT final matrix size: ', normalized_finalMatrix.shape

    return normalized_finalMatrix, patient_id


def measure_joints_accuracy(skeleton_data):

    frame_step = 1

    mean_displcement_list = np.zeros((len(skeleton_data[0])-1,1))

    joint_distances = []
    joint_distances_append = joint_distances.append

    # for joint_id in xrange(1,len(skeleton_data[0])):
    #
    #     #euclidean distance between joint time[0] and joint time[framestep]
    #     eu_difference = map(lambda i: np.sqrt((int(float(skeleton_data[i+frame_step][joint_id,0]))- int(float(skeleton_data[i][joint_id,0])))**2 + \
    #         (int(float(skeleton_data[i+frame_step][joint_id,1])) - int(float(skeleton_data[i][joint_id,1])))**2) \
    #         if skeleton_data[i][joint_id,0] != 0. or skeleton_data[i+1][joint_id,0] != 0. else 0 \
    #            ,xrange(0,len(skeleton_data)-frame_step))
    #
    #     mean_displcement_list[joint_id-1] = np.sum(eu_difference)/len(eu_difference)
    #
    #     joint_distances_append(eu_difference)

    #print mean_displcement_list
##############
    #subject7_exit_entrance = [19676 ,16250,  1943]
    subject4_exit_entrance = [3867,6053,9053,11898,17584,25777]

    ##not optimized code but more understadable
    for joint_id in xrange(1,len(skeleton_data[0])):
        eu_difference = np.zeros((len(skeleton_data),1))
        for i in xrange(0,len(skeleton_data)-1):
            if skeleton_data[i][joint_id,0] == 0. or skeleton_data[i+1][joint_id,0] == 0.:
                continue
            if i in subject4_exit_entrance:
                continue
            #euclidean distance between joint time[0] and joint time[framestep]
            eu_difference[i] = np.sqrt((int(float(skeleton_data[i+1][joint_id,0]))- int(float(skeleton_data[i][joint_id,0])))**2 + \
            (int(float(skeleton_data[i+1][joint_id,1])) - int(float(skeleton_data[i][joint_id,1])))**2)
        joint_distances_append(eu_difference)
        mean_displcement_list[joint_id-1] = np.sum(eu_difference)/len(eu_difference)


    joint_distances_filtered = []
    joint_distances_filtered_append = joint_distances_filtered.append
    for joint_id in xrange(1,len(skeleton_data[0])):

        ##store x,y for 1 joint each time over all frames
        list_points = []
        list_points_append = list_points.append
        for i in xrange(0,len(skeleton_data)):
            #if skeleton_data[i][joint_id,0] == 0.:
                #continue
            #if i in subject7_exit_entrance:
                #continue
            list_points_append((int(float(skeleton_data[i][joint_id,0])),int(float(skeleton_data[i][joint_id,1]))))

        ##apply filter
        x_f,y_f =my_img_proc.median_filter(list_points)

        eu_difference_filtered = np.zeros((len(skeleton_data),1))
        for i in xrange(0,len(x_f)-1):

            #if x_f[i+1] == 0. or x_f[i] == 0.:
                #continue
            if i in subject4_exit_entrance:
                continue

            eu_difference_filtered[i] = np.sqrt((x_f[i+1]-x_f[i])**2 + (y_f[i+1]-y_f[i])**2)
        joint_distances_filtered_append(eu_difference_filtered)
    # print mean_displcement_list

###############

    ##get only the desired joint
    ##TODO: joints id is different with hetra
    my_joint_raw = map(lambda x: x,joint_distances[:1][0])
    my_joint_filtered=map(lambda x: x,joint_distances_filtered[:1][0])

    #difference between raw and filtered features
    diff = map(lambda pair: pair[0]-pair[1] , zip(my_joint_raw,my_joint_filtered))

    #get frames where joint displacement over threshold
    threshold = 15

    frames_where_joint_displacement_over_threshold = []
    map(lambda (i,d): frames_where_joint_displacement_over_threshold.append(i) if d>threshold else False , enumerate(diff))
    print len(frames_where_joint_displacement_over_threshold)


    ##display mean distance of every joints between frames
    #vis.plot_mean_joints_displacement(mean_displcement_list)

    ##display error each frame from selected joint
    vis.plot_single_joint_displacement_vs_filtered_points(my_joint_raw,my_joint_filtered)

    return frames_where_joint_displacement_over_threshold


def feature_extraction_video_traj(skeleton_data, timeInterval_slice, draw_joints_in_scene):

    ##divide image into patches(polygons) and get the positions of each one
    my_room = np.zeros((424,512,3),dtype=np.uint8)
    my_room += 255
    list_poly = my_img_proc.divide_image(my_room)


    ##--------------Pre-Processing----------------##

    ##reliability method
    #measure_joints_accuracy(skeleton_data)
    #print skeleton_data[0]

    ## In case there are more than 1 ID in the scene I analyze the 1 trajectory per time

    ids = map(lambda line: line[0][2], skeleton_data)
    print 'skeleton id: ', Counter(ids).most_common()

    main_id = Counter(ids).most_common()[1][0]

    new_joints_points = []
    for i_point, points in enumerate(skeleton_data):
        if points[0][2] == main_id:
            new_joints_points.append(points)

    skeleton_data = new_joints_points

    if draw_joints_in_scene:
        ##Draw the skeleton and patches
        vis.draw_joints_and_tracks(skeleton_data, list_poly, my_room)

    ##Organize skeleton data in time interval segment
    skeleton_data_in_time_slices = org_data_in_timeIntervals(skeleton_data, timeInterval_slice)


    ##--------------Feature Extraction-------------##
    print 'feature extraction'

    ## count traj points in each region and create hist
    #occupancy_histograms = occupancy_histograms_in_time_interval(my_room, list_poly, skeleton_data_in_time_slices)
    occupancy_histograms = 0

    ## create Histograms of Oriented Tracks
    HOT_data,patient_ID = histograms_of_oriented_trajectories(list_poly,skeleton_data_in_time_slices)


    return [occupancy_histograms,HOT_data],patient_ID

    #cluster_prediction = my_exp.main_experiments(HOT_data)


def bow_traj(data_matrix,cluster_model,labels_training):

    ##get bow already computed
    bow_data = database.load_matrix_pickle('C:/Users/ICT4life/Documents/python_scripts/BOW_trained_data/BOW_16subject_2sec.txt')
    labels_bow_data = database.load_matrix_pickle('C:/Users/ICT4life/Documents/python_scripts/BOW_trained_data/BOW_labels_16subject_2sec.txt')

    ##labels meaning: 0 - normal activity , 1- normal-activity, 2-confusion, 3-repetitive, 4-questionnaire at table, 5- making tea


    classifiers.logistic_regression_train(bow_data, np.ravel(labels_bow_data), save_model=0)

    key_labels = map(lambda x: x[0],labels_training)

    hist = np.zeros((1, len(key_labels)))

    # vocabulary=[]
    for row in data_matrix:
        #prediction from cluster to see which word is most similar to
        similar_word = cluster_model.predict(np.array(row).reshape((1, -1)))
        index = np.where(similar_word == key_labels)[0][0]

        hist[0][index] +=1
        print classifiers.logistic_regression_predict(hist)


    #hist = normalize(np.array(hist),norm='l1')
    # if len(vocabulary)>0:
    #     vocabulary = np.vstack((vocabulary,hist))
    # else:
    #     vocabulary = hist

    return hist

def unix_time_ms(date):

    from datetime import datetime

    return (date - datetime(1970,1,1)).total_seconds() * 1000

def freezing_detection(db, time_interval):


    colMSBand = db.MSBand
    requestDate = datetime.strptime(time_interval[0], "%Y-%m-%d %H:%M:%S")

    requestInterval = 4  # seconds
    fps = 30

    model = classifiers.model_Freezing(nSteps=requestInterval*fps, nVars=4,
                                       RNN=[200, False, 1],
                                       lrnRate=10**-4, pDrop=0.2)

    #model = classifiers.load_model_weights(model, 'path to model weights')

    while True:

        t1 = datetime.now()

        timeStart = requestDate.strftime("%Y-%m-%d %H:%M:%S")
        timeEnd = (requestDate + timedelta(seconds=requestInterval)).strftime("%Y-%m-%d %H:%M:%S")

        d = database.read_MSBand_from_db(collection=colMSBand,
                                         time_interval=[timeStart, timeEnd],
                                         session='')

        ts = []
        data = []
        for i in range(len(d)):
            ts.append( unix_time_ms(d[i][1]) )
            data.append( d[i][2:5] )

        ts = np.array(ts)
        data = np.array(data)

        # if there are enough available data (>14 fps)
        if data.shape[0] >= requestInterval * 14:

            # add magnitude
            data = np.hstack((data, np.sqrt(np.sum(data**2, axis=1)).reshape(-1, 1)))

            data_interp = np.zeros((requestInterval * 30 + 1, 4))

            # interpolate data if needed
            if data.shape[0] != requestInterval * 30 + 1:

                for j in range(4):

                    new_ts = np.linspace(ts[0], ts[-1], requestInterval * 30 + 1)
                    data_interp[:, j] = np.interp(new_ts, ts, data[:, j])

            else:

                data_interp = data

            # normalize data
            data_interp = np.diff(data_interp, axis=0)

            data_interp = data_interp.reshape((1, requestInterval * fps, 4))

            prediction = model.predict(data_interp, batch_size=64)

            t2 = datetime.now()

            print '{0} | Freeze: {1:02.1f} % | Time: {2}ms'.\
            format(timeStart, 100 * prediction[0, 1], (t2 - t1).microseconds / 1000)

            requestDate += timedelta(seconds=requestInterval)

        else:

            print '{0} - {1}: Not enough data ({2})'.format(timeStart, timeEnd, len(d))

            t2 = datetime.now()

            requestDate += timedelta(seconds=requestInterval)

        if (t2 - t1).seconds < requestInterval:
                time.sleep(requestInterval - (t2 - t1).seconds)

        if datetime.strptime(time_interval[1], "%Y-%m-%d %H:%M:%S") < \
           datetime.strptime(timeEnd, "%Y-%m-%d %H:%M:%S"):
               break



def loss_of_balance_detection(db, time_interval):


    colKinect = db.Kinect
    requestDate = datetime.strptime(time_interval[0], "%Y-%m-%d %H:%M:%S")

    requestInterval = 1  # seconds

    while True:

        t1 = datetime.now()

        timeStart = requestDate.strftime("%Y-%m-%d %H:%M:%S")
        timeEnd = (requestDate + timedelta(seconds=requestInterval)).strftime("%Y-%m-%d %H:%M:%S")

        d = database.read_kinect_data_from_db(collection=colKinect,
                                              time_interval=[timeStart, timeEnd],
                                              session='',
                                              skeletonType='raw',
                                              exclude_columns=['ColorImage', 'DepthImage',
                                                               'InfraImage', 'BodyImage'])

        hasSkeleton = [False, False, False, False, False, False]
        idSkeleton = [0, 0, 0, 0, 0, 0]
        skeletons = []

        # read data
        for i in range(len(d)):

            for j in range(6):

                m = max(idSkeleton)

                # found new skeleton
                if hasSkeleton[j] == False and d[i][j] != []:
                    idSkeleton[j] = m + 1
                    hasSkeleton[j] = True
                    skeletons.append([])

                if hasSkeleton[j] == True and d[i][j] == []:
                    hasSkeleton[j] == False

                if d[i][j] != []:
                    skeletons[idSkeleton[j] - 1].append([d[i][j][2], d[i][j][3]])

        for k in range(len(skeletons)):

            skeletons[k] = np.array(skeletons[k])
            skeletons[k] = np.abs(skeletons[k])

            if np.sum(skeletons[k] > 0.9) > 0:
                prediction = 1
            else:
                prediction = 0

            t2 = datetime.now()

            print '{0} | skeleton {1} | Loss of balance: {1:02.1f} % | Time: {2}ms'.\
            format(timeStart, k, 100 * prediction, (t2 - t1).microseconds / 1000)

        t2 = datetime.now()

        requestDate += timedelta(seconds=requestInterval)

        if (t2 - t1).seconds < requestInterval:
                time.sleep(requestInterval - (t2 - t1).seconds)

        if datetime.strptime(time_interval[1], "%Y-%m-%d %H:%M:%S") < \
           datetime.strptime(timeEnd, "%Y-%m-%d %H:%M:%S"):
               break