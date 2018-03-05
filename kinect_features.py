import cv2
import numpy as np
from sklearn.preprocessing import normalize
from datetime import datetime, timedelta
from scipy.ndimage.filters import gaussian_filter
from lxml import etree
from collections import Counter
import time
import peakutils

import img_processing_kinect as my_img_proc
import database
import visualization as vis
import classifiers
import get_token


kinect_max_distance= 6.5
kinect_min_distance=0.5
cube_size = (kinect_max_distance-kinect_min_distance)/3


def org_data_in_timeIntervals(skeleton_data, timeInterval_slice):
    #get all time data from the list dropping the decimal
    content_time = map(lambda line: line[0,1].split('.')[0],skeleton_data) #
    

    #date time library

    init_t = datetime.strptime( content_time[0],'%Y-%m-%d %H:%M:%S') #+ ' ' + timeInterval_slice[3]
    end_t = datetime.strptime(content_time[len(content_time)-1],'%Y-%m-%d %H:%M:%S')
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

    #print 'time slice selected: ' + str(my_time_slice)

    #initialize list
    time_slices = []
    time_slices_append = time_slices.append

    c = (end_t-my_time_slice)
    #print init_t,end_t
    #get data in every timeslices
    while init_t < (end_t-my_time_slice):
        list_time_interval = []
        list_time_interval_append = list_time_interval.append

        for t in xrange(len(content_time)):

            if datetime.strptime(content_time[t],'%Y-%m-%d %H:%M:%S')>= init_t and datetime.strptime(content_time[t], '%Y-%m-%d %H:%M:%S') < init_t + my_time_slice:
                list_time_interval_append(skeleton_data[t])

            if datetime.strptime(content_time[t],'%Y-%m-%d %H:%M:%S') > init_t + my_time_slice:
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
    #ids = map(lambda line: np.int64(line[0][2]), time_slice)

    x_f = gaussian_filter(xs,2)
    y_f = gaussian_filter(ys,2)

    
    #global kinect_max_distance
    #global kinect_min_distance
    #kinect_max_distance = max(zs)
    #kinect_min_distance = min(zs)
    #kinect_min_distance = 0.5
    

    return x_f,y_f,zs


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
            continue


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
    #print 'OC final matrix size:' , normalized_finalMatrix.shape

    return normalized_finalMatrix


def histograms_of_oriented_trajectories_realtime(list_poly,time_slice):

    hot_all_data_matrix = []
    hot_all_data_matrix_append = hot_all_data_matrix.append

    if(len(time_slice)<1):
            print 'no data in this time slice'
            return
    else:
        #get x,y,z of every traj point after smoothing process
        x_filtered,y_filtered,zs = get_coordinate_points(time_slice, joint_id= 3)

        
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
                if np.isinf(x_filtered[ci]) or np.isinf(y_filtered[ci]): continue


                #2d polygon
                if list_poly[p].contains_point((int(x_filtered[ci]),int(y_filtered[ci]))):
                    ## 3d cube close to the camera
                    if zs[ci] <= (kinect_min_distance+cube_size):

                        tracklet_in_cube_append_c([x_filtered[ci],y_filtered[ci]])


                    elif zs[ci] > (kinect_min_distance+cube_size) and zs[ci] < (kinect_min_distance+(cube_size*2)): #
                        tracklet_in_cube_append_middle([x_filtered[ci], y_filtered[ci]])

                    elif zs[ci]>= kinect_min_distance + (cube_size*2): ##3d cube far from the camera
                        tracklet_in_cube_append_f([x_filtered[ci],y_filtered[ci]])


            #print 'size 3d patches in the scene ',len(tracklet_in_cube_c),len(tracklet_in_cube_middle),len(tracklet_in_cube_f)
            
            for three_d_poly in [tracklet_in_cube_c, tracklet_in_cube_middle, tracklet_in_cube_f]:
                if len(three_d_poly)>0:

                    ## for tracklet in cuboids compute HOT following paper
                    hot_single_poly = my_img_proc.histogram_oriented_tracklets(three_d_poly)

                    ## compute hot+curvature
                    #hot_single_poly = my_img_proc.histogram_oriented_tracklets_plus_curvature(three_d_poly)

                else:
                    hot_single_poly = np.zeros((24))


                ##add to general matrix
                if len(hot_matrix)>0: hot_matrix = np.hstack((hot_matrix,hot_single_poly))
                else: hot_matrix = hot_single_poly

        if np.sum(hot_matrix) > 0.0:
            #print np.sum(hot_matrix)
            ## normalize the final matrix
            normalized_finalMatrix = np.array(normalize(np.array(hot_matrix),norm='l2'))
        else:
            print 'empty feature variable'
            normalized_finalMatrix = np.zeros((hot_matrix.shape))

        #print 'HOT final matrix size: ', normalized_finalMatrix.shape
        

    return normalized_finalMatrix



def histograms_of_oriented_trajectories(list_poly,time_slices):

    #print kinect_max_distance, kinect_min_distance
    
    hot_all_data_matrix = []
    hot_all_data_matrix_append = hot_all_data_matrix.append


    for i in xrange(0,len(time_slices)):

        ##Checking the start time of every time slice
        if(len(time_slices[i])<1):
            #print time_slices[i]
            #print 'no data in this time slice'
            continue


        #get x,y,z of every traj point after smoothing process
        x_filtered,y_filtered,zs = get_coordinate_points(time_slices[i], joint_id= 3)
        
        
        #initialize histogram of oriented tracklets
        hot_matrix = []

        #print len(list_poly)
        for p in xrange(0,len(list_poly)):
            tracklet_in_cube_f = []
            tracklet_in_cube_c = []
            tracklet_in_cube_middle = []
            tracklet_in_cube_append_f = tracklet_in_cube_f.append
            tracklet_in_cube_append_c = tracklet_in_cube_c.append
            tracklet_in_cube_append_middle = tracklet_in_cube_middle.append

            for ci in xrange(0,len(x_filtered)):
                if np.isinf(x_filtered[ci]) or np.isinf(y_filtered[ci]): continue


                #2d polygon
                if list_poly[p].contains_point((int(x_filtered[ci]),int(y_filtered[ci]))):
                    ## 3d cube close to the camera
                    if zs[ci] <= (kinect_min_distance+cube_size):

                        tracklet_in_cube_append_c([x_filtered[ci],y_filtered[ci]])


                    elif zs[ci] > (kinect_min_distance+cube_size) and zs[ci] < (kinect_min_distance+(cube_size*2)): #
                        tracklet_in_cube_append_middle([x_filtered[ci], y_filtered[ci]])

                    elif zs[ci]>= kinect_min_distance + (cube_size*2): ##3d cube far from the camera
                        tracklet_in_cube_append_f([x_filtered[ci],y_filtered[ci]])


            #print len(tracklet_in_cube_c),len(tracklet_in_cube_middle),len(tracklet_in_cube_f)

            
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

    if np.sum(hot_all_data_matrix) > 0.0:
        #print np.sum(hot_all_data_matrix)
        ## normalize the final matrix
        normalized_finalMatrix = np.array(normalize(np.array(hot_all_data_matrix), norm='l2'))
    else:
        #print 'empty feature variable'
        normalized_finalMatrix = np.zeros((np.array(hot_all_data_matrix).shape))

    ##add extra bin containing time


    #print 'HOT final matrix size: ', normalized_finalMatrix.shape

    return normalized_finalMatrix


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
    #print len(frames_where_joint_displacement_over_threshold)


    ##display mean distance of every joints between frames
    #vis.plot_mean_joints_displacement(mean_displcement_list)

    ##display error each frame from selected joint
    vis.plot_single_joint_displacement_vs_filtered_points(my_joint_raw,my_joint_filtered)

    return frames_where_joint_displacement_over_threshold


def feature_extraction_video_traj(skeleton_data, bands_ids, draw_joints_in_scene, realtime):


    ##divide image into patches(polygons) and get the positions of each one
    my_room = np.zeros((424,512,3),dtype=np.uint8)
    #my_room = cv2.imread('C:/Users/certhadmin/Documents/ABD_files/pecs_room.jpg')
    my_room += 255
    list_poly = my_img_proc.divide_image(my_room)


    ##--------------Pre-Processing----------------##

    ##reliability method
    #measure_joints_accuracy(skeleton_data)
    #print skeleton_data[0]

    if draw_joints_in_scene: vis.draw_joints_and_tracks(skeleton_data, list_poly, my_room)


    ##--------------Feature Extraction-------------##
    #print 'feature extraction'

    if realtime:
        ## create Histograms of Oriented Tracks directly on skeleton data because only 1 sec is considered
        HOT_data = histograms_of_oriented_trajectories_realtime(list_poly, skeleton_data)
    else:
        ## NON real time  ##
        ## divide the skeleton according to re-id
        ids_data =  np.unique(np.array(skeleton_data)[:,0,2])

        HOT_data = []
        for b in bands_ids:
            band_id = b[0]
            uuid = b[1]

            idx_current_sk = np.where(np.array(skeleton_data)[:,0,2] == band_id)[0]
            skeleton_data_current_sk = np.array(skeleton_data)[idx_current_sk]

            if skeleton_data_current_sk.shape[0]>0:
                ## split considered period is small time intervals
                hours = 0
                minutes = 0
                seconds = 2

                skeleton_data_in_time_slices = org_data_in_timeIntervals(skeleton_data_current_sk, [hours,minutes,seconds])
                #print len(skeleton_data),len(list_poly)
                HOT_data.append(histograms_of_oriented_trajectories(list_poly, skeleton_data_in_time_slices))
            else: 
                HOT_data.append([])

    
    ## count traj points in each region and create hist
    #occupancy_histograms = occupancy_histograms_in_time_interval(my_room, list_poly, skeleton_data_in_time_slices)
    occupancy_histograms = 0

    return [occupancy_histograms,HOT_data]



def bow_traj(data_matrix,cluster_model,key_labels,hist):

    ##get bow already computed
##    bow_data = database.load_matrix_pickle('C:/Users/certhadmin/Desktop/ICT4LIFE_ABD_indoor/BOW_trained_data/BOW/BOW_30_kmeans_16subject_2sec.txt')
##    labels_bow_data = database.load_matrix_pickle('C:/Users/certhadmin/Desktop/ICT4LIFE_ABD_indoor/BOW_trained_data/BOW/BOW_30_kmeans_labels_16subject_2sec.txt')


    
    ##labels meaning: 0 - normal activity , 1- normal-activity, 2-confusion, 3-repetitive, 4-questionnaire at table, 5- making tea
    #classifiers.logistic_regression_train(bow_data, np.ravel(labels_bow_data), save_model=0)

    

    #hist = np.zeros((1, len(key_labels)))

    #database.save_matrix_pickle(data_matrix,'C:/Users/certhadmin/Desktop/behaviors_2.txt')

    list_similar_words =[]
    
    # vocabulary=[]
    for row in data_matrix:
        #hist = np.zeros((1, len(key_labels)))
        #prediction from cluster to see which word is most similar to
        similar_word = cluster_model.predict(np.array(row).reshape((1, -1)))
        #print 's_w', similar_word
        list_similar_words.append(similar_word[0])
        index = np.where(similar_word == key_labels)[0][0]
        hist[0][index]+=1

        

        #print classifiers.logistic_regression_predict(normalize(hist,norm='l1'))
        pred_result, pred_conf = classifiers.logistic_regression_predict(hist.reshape(1,-1))

    
    #vis.plot_word_distribution(key_labels,hist)
    #hist = normalize(np.array(hist),norm='l1')
    # if len(vocabulary)>0:
    #     vocabulary = np.vstack((vocabulary,hist))
    # else:
    #     vocabulary = hist

    return hist, pred_result, pred_conf


def load_model_weights(model, path):

    model.load_weights(path)

    return model


def model_Freeze(nSteps, nVars, RNN, lrnRate, pDrop=0.5):

    from keras.models import Model
    from keras.layers import Dense, LSTM, Input
    from keras.optimizers import Nadam


    if RNN[1]:
        model_in = Input(batch_shape=(RNN[2], nSteps, nVars))
    else:
        model_in = Input(shape=(nSteps, nVars))

    x = LSTM(units=RNN[0],
             implementation=0,
             return_sequences=False,
             stateful=RNN[1],
             unroll=True,
             dropout=pDrop,
             recurrent_dropout=pDrop,
             activation='tanh')(model_in)

    x = Dense(units=RNN[0],
              activation='tanh')(x)

    x = Dense(units=2,
              activation='softmax')(x)

    model_out = x



    model = Model(inputs=model_in, outputs=model_out)

    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=lrnRate),
                  sample_weight_mode='None')

    model.summary(line_length=100)


    return model


def computePSD(data, sr, window):

        fc = np.fft.fft(data * window)
        n = (fc.size / 2) + 1

        fc_real = fc[0:n]

        psd = (1.0 / (sr * np.sum(window ** 2))) * np.abs(fc_real) ** 2

        angle = np.unwrap(np.angle(fc_real)) * (np.pi / 180.0)
        freq = np.linspace(0, sr / 2.0, psd.size)


        return fc, fc_real, psd, angle, freq


def filter_predictions_sum(predictedY, filter_len, filter_val, zero_out=False):

    """

    Filter predictions to reduce false positives

    ------------------------------------------------------------------------------------------
    Parameters:

    predictedY: Matrix
        The predicted labels

    filter_len: int
        The filter length

    filter_val: int
        The filter threshold

    ------------------------------------------------------------------------------------------

    """

    import copy


    predY = copy.deepcopy(predictedY)
    s = np.zeros_like(predY)

    n = len(predY)
    m = np.floor(1.0 * filter_len / 2).astype(int)

    for i in range(m, n - m + 1):
        s[i - m : i + m + 1] = np.average( predictedY[i - m : i + m + 1] )
        if np.sum( predictedY[i - m : i + m + 1] ) >= filter_val:
            predY[i] = 1
        else:
            if zero_out:
                predY[i] = 0


    return predY, s


def freezing_detection(db, time_interval):

    requestInterval = 3  # seconds
    fps = 32

    msSeqThreshold = 360

    model = model_Freeze(nSteps=9, nVars=34 * 4,
                         RNN=[1000, False, 1],
                         lrnRate=10**-4, pDrop=0.5)

    #model = load_model_weights(model, '\\freeze_1.16.hdf5')

    #requestDate = '2017-09-08 10:56:34'
    #requestDate = datetime.strptime(requestDate, "%Y-%m-%d %H:%M:%S")

    # =======================================================================

    t1 = datetime.now()

    # get data for the last day
    #timeStart = (requestDate - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    #timeEnd = requestDate.strftime("%Y-%m-%d %H:%M:%S")

    # TEMP: get period with data #
    timeStart = time_interval[0]
    timeEnd = time_interval[1]


    d_all = database.read_MSBand_from_db(collection=db.MSBand,
                                         time_interval=[timeStart, timeEnd],
                                         session='')

    if len(d_all) == 0: print 'no data from the band'
    # =======================================================================

    results = dict()
    for key in d_all.keys():

        d = d_all[key]
        d = np.array(d)

        ts = np.array(d[:, 1], dtype=datetime)
        acc = np.array(d[:, [2, 3, 4]], dtype=float)
        acc = np.hstack((acc, np.sqrt(np.sum(acc**2, axis=1)).reshape(-1, 1)))
        features = []

        # loop through data
        time1 = ts[0]
        time2 = time1 + timedelta(seconds=3)

        while time2 < datetime.strptime(timeEnd, "%Y-%m-%d %H:%M:%S"):

            ind = np.bitwise_and(ts >= time1, ts < time2)

            ts_window = ts[ind]
            acc_window = acc[ind]

            for i in range(ts_window.shape[0]):
                ts_window[i] = (ts_window[i] - datetime(1970,1,1)).total_seconds() * 1000

            ts_window = np.array(ts_window, dtype=float)

            feat_window = []

            # if there are enough data (>15 fps)
            if acc_window.shape[0] >= requestInterval * 15:

                data_interp = np.zeros((requestInterval * fps, 4))

                # interpolate data if needed
                if acc_window.shape[0] != requestInterval * fps:

                    for j in range(4):
                        new_ts = np.linspace(ts_window[0], ts_window[-1], requestInterval * fps)
                        data_interp[:, j] = np.interp(new_ts, ts_window, acc_window[:, j])

                else:

                    data_interp = acc_window

                # extract features
                for j in range(4):

                    d_norm = data_interp[:, j]
                    d_norm = d_norm - np.mean(d_norm)

                    _, _, psd, _, freq = computePSD(d_norm, fps, np.ones(data_interp.shape[0]))
                    te = np.sum(psd)

                    feat = psd / te

                    ind1 = np.nonzero(freq >= 0.5)[0][0]
                    ind2 = np.nonzero(freq <= 12)[0][-1]

                    feat = feat[ind1 : ind2]

                    feat_window.append(feat)

                features.append([ts[ind][0], ts[ind][-1], ts_window[0], ts_window[-1],
                                 ts_window.shape[0], feat_window])

            time1 = time1 + timedelta(seconds=1/3.0)
            time2 = time1 + timedelta(seconds=3)

        t2 = datetime.now()

        del ind, ts_window, acc_window, i, feat_window, data_interp, new_ts, d_norm
        del feat, psd, freq, te, j, ind1, ind2

        print 'Feature extraction {0}ms'.format((t2 - t1).microseconds / 1000)

        # create sequences
        seq = []
        k = -1
        for i in range(1, len(features)):
            if features[i][2] - features[i - 1][2] > msSeqThreshold:
                seq.append([])
                k = k + 1

            seq[k].append(i)

        seq_filtered = []
        for i in range(len(seq)):
            if len(seq[i]) >= 9:
                r = range(seq[i][0], seq[i][-1] - 9 + 2)
                seq_filtered.append(r)

        del i, k, r

        # predict and filter results
        data_filtered = []
        predictions = []
        filtered_predictions = []
        for i in range(len(seq_filtered)):

            predictions.append([])
            for j in seq_filtered[i]:
                data = []
                for k in range(9):
                    data.append( np.hstack(features[j + k][5]) )

                data = np.array(data)
                data = np.reshape(data, (1, 9, 34 * 4))

                data_filtered.append([j, features[j][0], data])

                prediction = model.predict(data)
                predictions[i].append( prediction[0, 1] )

            predictions[i] = np.array(predictions[i])
            predictions[i] = filter_predictions_sum(predictions[i], 4, 2, True)[0]

            predictions[i][predictions[i] < 0.9] = 0
            filtered_predictions.append( predictions[i] )

        # summarize events
        events = []
        pos = 0;
        for i in range(len(filtered_predictions)):
            events.append([])
            start = -1
            end = -1
            for j in range(filtered_predictions[i].shape[0]):

                if start == -1 and filtered_predictions[i][j] == 1:
                    start = pos

                if start > -1:
                    if filtered_predictions[i][j] == 0:
                        end = pos - 1
                    if j == filtered_predictions[i].shape[0] - 1:
                        end = pos

                if start > -1 and end > -1:
                    events[i].append([start, end])
                    start = -1
                    end = -1

                pos = pos + 1

        events_sum = []
        for i in range(len(events)):

            events_sum.append([])
            for j in range(len(events[i])):

                start = events[i][j][0]
                end = events[i][j][1]

                td = data_filtered[end][1] - data_filtered[start][1]

                events_sum[i].append([data_filtered[start][1], td.total_seconds()])

            try:
                events_sum.remove([])
            except:
                "No events"

        results[key] = events_sum


    return results


def unix_time_ms(date):

    return (date - datetime(1970,1,1)).total_seconds() * 1000


def festination(db, time_interval):

    fps = 30

    #requestDate = '2017-12-07 09:27:00'
    #requestDate = datetime.strptime(requestDate, "%Y-%m-%d %H:%M:%S")

    # =======================================================================

    # get data for the last day
    timeStart = time_interval[0]#.strptime("%Y-%m-%d %H:%M:%S")#(requestDate - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    timeEnd = time_interval[1]#.strptime("%Y-%m-%d %H:%M:%S")#requestDate.strftime("%Y-%m-%d %H:%M:%S")



    d_all = database.read_kinect_data_from_db(collection=db.Kinect,
                                              time_interval=[timeStart, timeEnd],
                                              session='',
                                              skeletonType=['filtered'],
                                              exclude_columns=['ColorImage', 'DepthImage',
                                                               'InfraImage', 'BodyImage'])

    d_all_MSband = database.read_MSBand_from_db_asDict(collection=db.MSBand,
                                                       time_interval=[timeStart, timeEnd],
                                                       session='')

    results = dict()
    for key in d_all.keys():

        results[key] = dict()

        d = d_all[key]

        hasSkeleton = [False, False, False, False, False, False]
        idSkeleton = [0, 0, 0, 0, 0, 0]
        skeletons = []
        ts_skeletons = []

        # read data
        for i in range(len(d)):

            for j in range(6):

                m = max(idSkeleton)

                # found new skeleton
                if hasSkeleton[j] == False and d[i][j] != []:
                    idSkeleton[j] = m + 1
                    hasSkeleton[j] = True
                    ts_skeletons.append([])
                    skeletons.append([])

                if hasSkeleton[j] == True and d[i][j] == []:
                    hasSkeleton[j] == False

                if d[i][j] != []:
                    ts_skeletons[idSkeleton[j] - 1].append( unix_time_ms(d[i][j][1]) )
                    skeletons[idSkeleton[j] - 1].append( [d[i][j][1], np.array(d[i][j][6])] )

        for j in range(len(skeletons)):
            ts_skeletons[j] = np.array(ts_skeletons[j])

        idSkeleton = np.array(idSkeleton)


        # calculate magnitude and find peaks
        acc = []
        ts_acc = []
        for band_key in d_all_MSband.keys():

            d_band = d_all_MSband[band_key]

            for band_i in range(len(d_band)):
                ts_acc.append( unix_time_ms(d_band[band_i]['accTS']) )
                acc.append([
                    d_band[band_i]['accTS'],
                    np.sqrt(d_band[band_i]['accX']**2 + \
                            d_band[band_i]['accY']**2 + \
                            d_band[band_i]['accZ']**2)])

            mag = np.array(np.array(acc)[:, 1], dtype=float)

            # detect peaks
            peaks = peakutils.indexes(mag, thres=0, min_dist=fps / 6)
            ind = mag[peaks] <= 1.2
            peaks = np.delete(peaks, np.nonzero(ind)[0])
			
			# count steps per second
            cad = []
            w_len = fps * 2
            w_step = 1
            for i in range(0, mag.shape[0] - w_len, w_step):
                ind = np.arange(i, i + w_len)
                steps = np.sum(np.bitwise_and(peaks >= ind[0], peaks <= ind[-1]))
                cad.append( [acc[i][0], acc[i + w_len][0], steps, steps * fps / float(w_len)] )

            sps = np.array(np.array(cad)[:, 3], dtype=float)
            sps_x = np.arange(1, mag.shape[0] + 1 - w_len, w_step) + w_len / 2
			
			# get skeleton position at each magnitude peak
            body_pos = []
            thr = 30 # ms
            for i in range(len(skeletons)): # which skeleton? (re-id)

                body_pos.append([])
                for j in range(peaks.shape[0]):

                    diff = np.abs(ts_skeletons[i] - ts_acc[peaks[j]])
                    min_pos = np.argmin(diff)
                    min_val = diff[min_pos]

                    if min_val <= thr:
                        body_pos[i].append( skeletons[i][min_pos][1][0, [0,1,2]] )
                    else:
                        body_pos[i].append( None )

            # measure distance between peaks
            dist = []
            for i in range(len(skeletons)):

                dist.append([])
                for j in range(peaks.shape[0] - 1):
                    if (body_pos[i][j] is not None) and (body_pos[i][j + 1] is not None):
                        dist[i].append( np.sqrt(np.sum((body_pos[i][j + 1] - body_pos[i][j])**2)) )
                    else:
                        dist[i].append( np.NaN )

                dist[i] = np.array(dist[i])

            dist_x = peaks[:-1] + np.diff(peaks) + 1
			
            events = []
            for i in range(len(skeletons)):

                events.append([])

                ind1 = dist_x[dist[i] < 0.4]
                ind2 = sps_x[sps > 2]

                ind = np.intersect1d(ind1, ind2)

                jMax = np.minimum(int(ind.shape[0]), int(ind1.shape[0]))

                j = 0
                seq = []
                while j < jMax:

                    ind_start = int(np.nonzero(ind1 == ind[j])[0])
                    kMax = np.minimum(ind[j:].shape[0], ind1[ind_start:].shape[0])

                    seq.append([])
                    count = 0
                    for k in range(kMax):
                        if ind[j + k] == ind1[ind_start + k]:
                            seq[-1].append( ind[j + k] )
                            count = count + 1
                        else:
                            break

                    j = j + count

                events[i] = seq

            events_sum = []
            for i in range(len(events)):

                events_sum.append([])
                for j in range(len(events[i])):

                    start = events[i][j][0]
                    end = events[i][j][-1]

                    td = acc[end][0] - acc[start][0]

                    events_sum[i].append([acc[start][0], td.total_seconds()])

                try:
                    events_sum.remove([])
                except:
                    "No events"


            results[key][band_key] = events_sum


    return results
	
