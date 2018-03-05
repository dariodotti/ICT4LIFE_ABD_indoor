import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter


def bar_plot_occupancy_selectedAreas_over_time(data):
    last_col = np.array(data).shape[1]-1
    hs = data[:,last_col:]
    data = data[:,:last_col]

    ##definition of semantic areas
    ##TODO: retrieve the data from external file
    door_areas=[23,25,39,41,55,56,57,72,73]
    desks_areas= [35,37,45,51,53,61]

    working_h = [9,10,11,12,13,14,15,16,17]

    class_freq_per_hour = np.zeros([len(working_h),2])


    for i,hist_allcubes in enumerate(data):
        array_pos = int(hs[i])-working_h[0]

        for n_cube,value_singlecube in enumerate(hist_allcubes):
            if n_cube in door_areas:
                class_freq_per_hour[array_pos,0]= class_freq_per_hour[array_pos,0]+value_singlecube
            elif n_cube in desks_areas:
                class_freq_per_hour[array_pos,1]=class_freq_per_hour[array_pos,1]+value_singlecube

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.25
    ind = np.arange(9,9+len(working_h))
    rect1 = ax.bar(ind,class_freq_per_hour[:,0],width,color='red')
    rect2 = ax.bar(ind+width,class_freq_per_hour[:,1],width)

    ## add a legend
    ax.legend( (rect1[0], rect2[0]), ('Door area', 'Desks area'),fontsize=11 )

    # axes and labels
    ax.set_xlim(9-width,9+len(ind)+width)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(ind)

    plt.show()


def bar_plot_motion_over_time(data):
    last_col = np.array(data).shape[1]-1
    hs = data[:,last_col:]
    data = data[:,:last_col]

    ##definition of semantic areas
    ##TODO: retrieve the data from external file
    door_areas=[23,25,39,41,55,56,57,72,73]
    desks_areas= [35,37,45,51,53,61]

    working_h = [9,10,11,12,13,14,15,16,17]

    magn_per_hour = np.zeros([len(working_h),3])

    for i,hist_allcubes in enumerate(data):
        hist_allcubes = hist_allcubes.reshape((np.array(data).shape[1]/24,24))
        array_pos = int(hs[i])-working_h[0]

        for n_cube,value_singlecube in enumerate(hist_allcubes):
            value_singlecube = value_singlecube.reshape((8,3))
            magns_cube = value_singlecube.sum(axis= 0)


            magn_per_hour[array_pos,0] = magn_per_hour[array_pos,0]+magns_cube[0]
            magn_per_hour[array_pos,1] = magn_per_hour[array_pos,1]+magns_cube[1]
            magn_per_hour[array_pos,2] = magn_per_hour[array_pos,2]+magns_cube[2]


    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.25
    ind = np.arange(9,9+len(working_h))
    rect1 = ax.bar(ind,magn_per_hour[:,0],width,color='red')
    rect2 = ax.bar(ind+width,magn_per_hour[:,1],width)
    rect3 = ax.bar(ind+(width*2),magn_per_hour[:,2],width,color='green')

    ## add a legend
    ax.legend( (rect1[0], rect2[0],rect3[0]), ('stationary', 'slight mov', 'mov'),fontsize=11 )

    # axes and labels
    ax.set_xlim(9-width,9+len(ind)+width)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(ind)

    plt.show()


def pie_plot_motion_day(data,plot=0):
    #last_col = np.array(data).shape[1]-1
    #hs = data[:,last_col:]
    #data = data[:,:last_col]

    desks_areas = range(0,36)#[12,13,14]


    #motion divided in 3 groups
    motion = np.zeros((1,3))
    time_counter = 0
    for i,hist_allcubes in enumerate(data):
        hist_allcubes = hist_allcubes.reshape((np.array(data).shape[1]/24,24))

        for n_cube,value_singlecube in enumerate(hist_allcubes):
            value_singlecube = value_singlecube.reshape((8,3))
            magns_cube = value_singlecube.sum(axis= 0)

            if n_cube in desks_areas:
                motion[0,0] = motion[0,0]+magns_cube[0]
                motion[0,1] = motion[0,1]+magns_cube[1]
                motion[0,2] = motion[0,2]+magns_cube[2]


    if motion.sum(axis=1) == 0:
        print 'no motion to plot ', motion
    else:
        if plot != 0:
            #plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            labels = 'stationary', 'slight mov', 'fast mov'
            colors = ['yellowgreen', 'gold', 'lightskyblue']
            pie_slice_size = [float(i)/np.sum(motion[0]) for i in motion[0]]

            print motion

            ax.pie(pie_slice_size,labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)
            plt.axis('equal')
            #plt.show()

    return motion


def bar_plot_motion_in_region_over_long_time(motion_week):
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']


    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.5
    ind = np.arange(0,len(days))
    plt.bar(ind,motion_week,width)

    ax.set_xticklabels(days)
    ax.set_xticks(ind+0.2)


    plt.show()


def plot_ambient_sensor_ONOFF_pair_over_time(sensor_data):
    print 'plots ambient sensors over time'

    fig = plt.figure()
    ax = fig.add_subplot(111)

    time_in_seconds_off=[]
    time_in_seconds_on = []
    markers=[]
    colors=[]
    for e in sensor_data:
        raw_time=e.split(' ')[2]
        minutes = raw_time.split('-')[1]
        seconds = raw_time.split('-')[2]

        ##converts in seconds

        if e[9:12] == 'OFF':
            time_in_seconds_off.append((int(minutes)*60) + int(seconds))

            markers.append((5,2))
            colors.append('red')
        else:
            time_in_seconds_on.append((int(minutes)*60) + int(seconds))
            markers.append((5,2))
            colors.append('blue')

    width = 5
    fix_y= np.ones((len(time_in_seconds_off)))

    rect_open =ax.bar(np.array(time_in_seconds_off),fix_y,width,color='r')

    fix_y= np.ones((len(time_in_seconds_on)))
    rect_close = ax.bar(np.array(time_in_seconds_on)+width,fix_y,width,color='b')

    ax.set_ylim(0,1.2)

    ax.legend( (rect_open[0], rect_close[0]), ('Door open', 'Door close'),fontsize=11 )

    #ax.set_xticklabels()

    # for x,y,m,c in zip(time_in_seconds_off+time_in_seconds_on,[1,1,1,1,1,1,1,1,1,1,1],markers,colors):
    #     plt.scatter(x,y,marker=m,s=80,c=c)
    #
    plt.show()


def plot_mean_joints_displacement(mean_displacement_list):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ##TODO: joints from hetra are called by ID and not by name
    joints = ['head','neck','spine mid','spine base','spine shoulder','shoulder R','elbow R','wrist R','hand R','shoulder L','elbow L',\
              'wrist L','hand L','hip R', 'knee R','ankle R','foot R','hip L','knee L','ankle L','foot L']

    for mean_d,joint_name in zip(mean_displacement_list,joints):

        ax.scatter(1,mean_d)

        ax.annotate(joint_name, xy=(1,mean_d), xycoords='data',size=8)

    plt.show()


def plot_single_joint_displacement_vs_filtered_points(my_joint_raw,my_joint_filtered):

    ##plot as frequency
    plt.plot(my_joint_raw,color='b',label='raw points')
    plt.plot(my_joint_filtered,color='r',label='filtered points')
    plt.title('subject 7')
    plt.show()


def draw_joints_and_tracks(body_points, scene_patches, scene):
    color = (0, 0, 255)

    # draw line between joints
    thickness = 3
    line_color = (19, 19, 164)

    ##check patches are correct
    for i_rect, rect in enumerate(scene_patches):
        cv2.rectangle(scene, (int(rect.vertices[1][0]), int(rect.vertices[1][1])),
                      (int(rect.vertices[3][0]), int(rect.vertices[3][1])), (0, 0, 0))

        ## write number of patch on img
        cv2.putText(scene, str(i_rect), (int(rect.vertices[1][0]) + 10, int(rect.vertices[1][1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    for n_frame, traj_body_joints in enumerate(body_points):
        # n_frame = n_frame+1402

        # if n_frame < 4870:
        #     continue

        temp_img = scene.copy()

        # draw joints
        print n_frame

        # first position skipped cause there are other info stored
        try:
            # torso
            cv2.line(temp_img, (int(float(traj_body_joints[4, 0])), int(float(traj_body_joints[4, 1]))),
                     (int(float(traj_body_joints[3, 0])), int(float(traj_body_joints[3, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[3, 0])), int(float(traj_body_joints[3, 1]))),
                     (int(float(traj_body_joints[2, 0])), int(float(traj_body_joints[2, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[2, 0])), int(float(traj_body_joints[2, 1]))),
                     (int(float(traj_body_joints[1, 0])), int(float(traj_body_joints[1, 1]))), line_color, thickness)
            # shoulder
            cv2.line(temp_img, (int(float(traj_body_joints[21, 0])), int(float(traj_body_joints[21, 1]))),
                     (int(float(traj_body_joints[9, 0])), int(float(traj_body_joints[9, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[21, 0])), int(float(traj_body_joints[21, 1]))),
                     (int(float(traj_body_joints[5, 0])), int(float(traj_body_joints[5, 1]))), line_color, thickness)
            # hips
            cv2.line(temp_img, (int(float(traj_body_joints[1, 0])), int(float(traj_body_joints[1, 1]))),
                     (int(float(traj_body_joints[17, 0])), int(float(traj_body_joints[17, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[1, 0])), int(float(traj_body_joints[1, 1]))),
                     (int(float(traj_body_joints[13, 0])), int(float(traj_body_joints[13, 1]))), line_color, thickness)
            # right arm
            cv2.line(temp_img, (int(float(traj_body_joints[9, 0])), int(float(traj_body_joints[9, 1]))),
                     (int(float(traj_body_joints[10, 0])), int(float(traj_body_joints[10, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[10, 0])), int(float(traj_body_joints[10, 1]))),
                     (int(float(traj_body_joints[11, 0])), int(float(traj_body_joints[11, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[11, 0])), int(float(traj_body_joints[11, 1]))),
                     (int(float(traj_body_joints[12, 0])), int(float(traj_body_joints[12, 1]))), line_color, thickness)
            # left arm
            cv2.line(temp_img, (int(float(traj_body_joints[5, 0])), int(float(traj_body_joints[5, 1]))),
                     (int(float(traj_body_joints[6, 0])), int(float(traj_body_joints[6, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[6, 0])), int(float(traj_body_joints[6, 1]))),
                     (int(float(traj_body_joints[7, 0])), int(float(traj_body_joints[7, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[7, 0])), int(float(traj_body_joints[7, 1]))),
                     (int(float(traj_body_joints[8, 0])), int(float(traj_body_joints[8, 1]))), line_color, thickness)

            # left leg
            cv2.line(temp_img, (int(float(traj_body_joints[13, 0])), int(float(traj_body_joints[13, 1]))),
                     (int(float(traj_body_joints[14, 0])), int(float(traj_body_joints[14, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[14, 0])), int(float(traj_body_joints[14, 1]))),
                     (int(float(traj_body_joints[15, 0])), int(float(traj_body_joints[15, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[15, 0])), int(float(traj_body_joints[15, 1]))),
                     (int(float(traj_body_joints[16, 0])), int(float(traj_body_joints[16, 1]))), line_color, thickness)
            # right leg
            cv2.line(temp_img, (int(float(traj_body_joints[17, 0])), int(float(traj_body_joints[17, 1]))),
                     (int(float(traj_body_joints[18, 0])), int(float(traj_body_joints[18, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[18, 0])), int(float(traj_body_joints[18, 1]))),
                     (int(float(traj_body_joints[19, 0])), int(float(traj_body_joints[19, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[19, 0])), int(float(traj_body_joints[19, 1]))),
                     (int(float(traj_body_joints[20, 0])), int(float(traj_body_joints[20, 1]))), line_color, thickness)

            if n_frame > 0:
                for i, joint in enumerate(traj_body_joints):
                    if i == 0:
                        continue
                    cv2.circle(temp_img, (int(float(joint[0])), int(float(joint[1]))), 2, color, -1)
                    if i == 3 and n_frame > 0:
                        ##draw trajectories
                        cv2.circle(scene, (int(float(joint[0])), int(float(joint[1]))), 2, color, -1)
                    else:
                        ##draw joint
                        cv2.circle(temp_img, (int(float(joint[0])), int(float(joint[1]))), 2, color, -1)

            cv2.imshow('lab', temp_img)
            cv2.waitKey(1)

        except:
            print 'traj coordinates not available'
            continue

def plot_word_distribution(keys_labels,data):

    fig, ax = plt.subplots()

    bins = np.arange(0,len(keys_labels))
    width = 0.35       # the width of the bars

    confused_data = ax.bar(bins,data[0],width, color='b')

    labels = []
    
    for x in keys_labels:
        labels.append(str(x))

    ax.set_xticks(bins + width / 2)
    ax.set_xticklabels(labels)

    plt.show()
        
    
