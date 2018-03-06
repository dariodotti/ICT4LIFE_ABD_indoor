Abnormal Behavior Detection subsystem 
Authors: UM,UPM,CERTH 

This code has two main function depending on the use.

nonrealtime_functionalities.py 

It contains all the functions for off line behaviors analysis.  First the re-id code is run to match the kinect trajectory with the MS band, then all the 
method listed in the functionality table described in Task 3.3 are launched.
Finally, a summarization file is created for the Higher modules.

ARGUMENTS TO CALL THIS FUNCTION: 
PATH\nonrealtime_functionalities.py PATH\conf_file_abd_ict4life.xml "2018-03-02 23:00:00"

In this example, the file will analyze the 24h before the selected date.


####################

realtime_functionalities.py

It contains all the function that work in real-time, it continuously read the last 1 second of data from the db, and 
it run the functions to analyze it. If Abnormality is detected, a notification will be send to the phone connected to the MSband.

ARGUMENTS TO CALL THIS FUNCTION: 

PATH\realtime_functionalities.py PATH\conf_file_abd_ict4life.xml