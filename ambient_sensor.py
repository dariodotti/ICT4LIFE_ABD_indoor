import visualization as vis
import conf_file


def org_data_ID(as_data):

    sensor_pos = conf_file.get_ambient_sensor_position()

    #make dictionary with k the position and value the data
    sensors_ID = {}

    #initialize dictionary
    for k in sensor_pos.keys():
        sensors_ID.setdefault(k,[])

    for k in sensor_pos.keys():
        for i in xrange(1,len(as_data)):
            if as_data[i][:3] == sensor_pos[k]:
                sensors_ID[k].append(as_data[i])

    #print sensors_ID['entrance']
    #print sensors_ID['toilet']
    #print sensors_ID['bedroom']

    return sensors_ID


def night_motion(as_data):
    sensors_ID = org_data_ID(as_data)

    #check reliability
    #vis.plot_ambient_sensor_ONOFF_pair_over_time(sensor_ID)

    ##TODO: hist each ID how many times are activated during night
    print 'ambient sensor night activation'
    print int(len(sensors_ID['toilet'])/2)+int(len(sensors_ID['bedroom'])/2)



def nr_visit_bathroom(as_data):

    sensors_ID = org_data_ID(as_data)
    object_events=sensors_ID['toilet']

    if len(object_events)%2 != 0:
        print 'odd number of events: '+ str(len(object_events))


    #check reliability
    #vis.plot_ambient_sensor_ONOFF_pair_over_time(object_events)

    ##todo: HISTOGRAM OVER morning/afternoon/evening/night
    print 'number visit bathroom'
    print int(len(sensors_ID['toilet'])/2)
