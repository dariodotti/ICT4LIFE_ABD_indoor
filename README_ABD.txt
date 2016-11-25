Abnormal Behavior Detection subsystem 
Author: Dario Dotti

The first version of the system focus only on non-realtime functionalities listed in the functionality table. 

INPUT: The system asks as input a configuration file that contains all the parameters that have to be modified often. The first parameters in the file is available_sensor that allows
you to run all the functionalies in the system even if you are missing sensor's data. For example if you do not have the abient sensors, you have to set it to 0 and the system will not call
the ambient sensor class.

OUTPUT: Depending on the functionality it plots the selected data, it labels an activity pattern using classifiers, or it save data into the database.


Classes:

nonrealtime_functionalities.py is the main class. First it loads all the parameters from the cofiguration file using the conf_file.py class, then it calls all the non-realtime 
functionaties (e.g. daily motion, night motion).

conf_file.py reads the parameters from the configuration file. 

database.py reads the sensors data recorded in a certain time interval from the database.

classifiers.py contains some supervised/ unsupervised classifiers from scikit library

visualization.py contains method to plot the data 


Classes only for tasks belonging to University of Maastricht:

kinect_global_trajectories.py
img_processing_kinect.py
ambient_sensor.py