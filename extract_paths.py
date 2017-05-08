from pymongo import MongoClient
import datetime
from time import mktime

hourIn = "00:00:00"
dateIn = "03-05-2017"
hourFin = "00:00:00"
dateFin = "03-05-2017"

fin = dateIn.split('-')
dateIn = datetime.datetime(int(fin[2]), int(fin[1]), int(fin[0]), 0, 0)
dateIn = mktime(datetime.datetime.timetuple(dateIn))
hourIn = hourIn.split(':')
dateIn = dateIn+int(hourIn[0])*60*60+int(hourIn[1])*60+int(hourIn[2]) 
ffin = dateFin.split('-')
dateFin = datetime.datetime(int(ffin[2]), int(ffin[1]), int(ffin[0]), 0, 0)
dateFin = mktime(datetime.datetime.timetuple(dateFin))
hourFin = hourFin.split(':')
dateFin = dateFin+int(hourFin[0])*60*60+int(hourFin[1])*60+int(hourFin[2])

client = MongoClient('localhost', 27017)
dbName = 'local'
collectionName = 'paths'
db = client[dbName][collectionName]
paths = db.find({"ts": {"$gt": dateIn, "$lt": dateFin}}).sort("ts")
ids = []
for data in paths:
	if data["id"] not in ids:
		ids.append(data["id"])
finalPaths = {}
for i in ids:
	dataPaths = db.find({"ts": {"$gt": dateIn, "$lt": dateFin}, "id": i}).sort("ts")
	finalPaths[i] = {'Kinect': [], 'Zenith': [], 'WSN': []}
	for data in dataPaths:
		if data['device'] == 'Kinect':
			finalPaths[i]['Kinect'].append([data['x'],data['y'],data['z'],data['ts'],data['confidence']])
		if data['device'] == 'Zenith':
			finalPaths[i]['Zenith'].append([data['x'],data['y'],data['z'],data['ts'],data['confidence']])
		if data['device'] == 'WSN':
			finalPaths[i]['WSN'].append([data['x'],data['y'],data['z'],data['ts'],data['confidence']])

print finalPaths