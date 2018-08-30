import visualization as vis
import conf_file
from datetime import datetime, timedelta


def org_data_ID(as_data):

    sensor_pos = conf_file.get_ambient_sensor_position()

    #make dictionary with k the position and value the data
    sensors_ID = {}

    #initialize dictionary
    for k in sensor_pos.keys():
        sensors_ID.setdefault(k,[])

    for k in sensor_pos.keys():
        for i in xrange(0,len(as_data)):
            if as_data[i][:3] == sensor_pos[k]:
                sensors_ID[k].append(as_data[i])

    print sensors_ID['entrance']
    print sensors_ID['toilet']
    #print sensors_ID['bedroom']

    return sensors_ID


#def as_motion(as_data):
#
#    sensors_ID = org_data_ID(as_data)
#
#    #check reliability
#    #vis.plot_ambient_sensor_ONOFF_pair_over_time(sensor_ID)
#
#    #sensor_activation = {}
#    sensor_activation = {key: [] for key in sensors_ID.keys()}
#    for k in sensors_ID.keys():
#        ##general time stamp list
#        ##for every sensor create two lists: off and on and compute the duration of the stay in a certain room
#        off_list = []
#        on_list = []
#        for value in sensors_ID[k]:
#            line = value.split(' ')
#            c = line[0].encode('utf8')
#            if c[len(c)-2:] == 'FF':
#                off_list.append(line[2].encode('utf8').replace('-',':'))
#            elif c[len(c)-2:] == 'N-':
#                on_list.append(line[2].encode('utf8').replace('-',':'))
#
#        ##compute durationn of stay: NOW i assume the first event is open the door (TODO: think of robust solution if people do not close the doors)
#        #durations = []
#        #new_off_list = []
#        on_list.append(3)
#        on_list.append(9)
#        on_list.append(90)
#        on_list.append(90)
#
#
#        for i in range(0,len(on_list)-1,2):
#
#            durations = {'duration':'', 'beginning':''}
#            
#            end_t = datetime.strptime('15:37:00', '%H:%M:%S')
#            start_t= datetime.strptime('15:36:00', '%H:%M:%S')
#            durations['duration']= str(end_t-start_t)
#            durations['beginning']= '2018-01-18 15:36:00'#off_list[i]
#            #durations.append(str(end_t-start_t))
#            #new_off_list.append(off_list[i])
#
#            sensor_activation[k].append(durations)
#
#    return sensor_activation


def as_motion(sensors_ID):
    
    #check reliability
    #vis.plot_ambient_sensor_ONOFF_pair_over_time(sensor_ID)
            
    sensor_activation = {key: [] for key in sensors_ID.keys()}
    for k in sensors_ID.keys():
        ##general time stamp list
        
        for i_v in range(len(sensors_ID[k])-1):
          
            now_t=datetime.strptime(sensors_ID[k][i_v][1], '%Y-%m-%d %H:%M:%S')
            next_t=datetime.strptime(sensors_ID[k][i_v+1][1], '%Y-%m-%d %H:%M:%S')
            
            if (next_t-now_t) > timedelta(seconds=30):
                durations = {'duration':'', 'beginning':''}
                durations['beginning'] = now_t.strftime('%Y-%m-%d %H:%M:%S')
            
                sensor_activation[k].append(durations)
        
    return sensor_activation


def nr_visit_bathroom(as_data):

    sensors_ID = org_data_ID(as_data)
    object_events=sensors_ID['toilet']

    if len(object_events)%2 != 0:
        print 'odd number of events: '+ str(len(object_events))



    #check reliability
    #vis.plot_ambient_sensor_ONOFF_pair_over_time(object_events)

    event_timestamp = []
    for o in object_events:
        event_timestamp.append(o.split(' ')[2].encode('utf8').replace('-',':'))

    durations = []
    for t in range(0, len(event_timestamp) - 1, 2):
        end_t = datetime.strptime(event_timestamp[t + 1], '%H:%M:%S')
        start_t = datetime.strptime(event_timestamp[t], '%H:%M:%S')
        durations.append(str(end_t - start_t))

    toilet_event = {}
    toilet_event['toilet'] = [event_timestamp, durations]
    ##todo: HISTOGRAM OVER morning/afternoon/evening/night
    #print 'number visit bathroom'
    #print int(len(sensors_ID['toilet'])/2)

    return toilet_event