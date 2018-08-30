@ECHO off
for %%d in (18 19 20 21 22 25 26 30 31) do ( 
echo Processing the summarizaon file for %%d of July 2018.
python nonrealtime_functionalities.py conf_file_abd_ict4life.xml "2018-07-%%d 22:00:00"
)
