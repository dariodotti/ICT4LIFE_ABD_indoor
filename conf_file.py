import xml.etree.ElementTree as ET



avaliable_sensors = {}
ambient_sens_pos = {}

def parse_conf_file(file_path):
    #read values in conf file and save it in dictionaries

    tree = ET.parse(file_path)
    root = tree.getroot()

    for element in root:
        if element.tag == 'available_sensor':
            for child in element:
                avaliable_sensors[child.tag] = int(child.text)
        if element.tag == 'ambientSensor_position':
            for child in element:

                ambient_sens_pos[child.tag] = child.text




def get_available_sensors():
    if len(avaliable_sensors) == 0:
        print 'avalable sensors empty!'
    else:
        print 'available sensor'
        print avaliable_sensors

    return avaliable_sensors

def get_ambient_sensor_position():
    if len(ambient_sens_pos) == 0:
        print 'ambient sensor position empty!'
    return ambient_sens_pos