import numpy as np
import os
import time
import datetime
import inspect
import logging

scriptPath = os.path.dirname(os.path.realpath(__file__))
    

class myLogging(object):
    output_dir_path = ""
    infoString = ""
    infoFilePath = ""
    
    def setOutputDirPath(self, scriptName):
        scriptNameCapitalized = "".join(scriptName[:1].upper() + scriptName[1:])
        self.output_dir_path = scriptPath+"/../data/temp"+scriptNameCapitalized+"/"
        self.infoFilePath = self.output_dir_path+"info.txt"
    
    def setScriptName(self, scriptName):
        self.infoString = scriptName + "info:\n"
    
    def log(self,variableName,variableValue):
        self.infoString += "\t" + variableName + ":\n\t\t" + str(variableValue) + "\n"
        
    def writeLog(self):            
        infoFile = open(self.infoFilePath,"a")
        infoFile.write(self.infoString)
        infoFile.close()

myLog = myLogging()

def init(scriptName, overWriteOutput):
    myLog.setOutputDirPath(scriptName)
    myLog.setScriptName(scriptName)
    if os.path.isdir(myLog.output_dir_path):
        if not(overWriteOutput):
            logging.error("output directory ("+myLog.output_dir_path+") already exists")
            exit()
        else:
            infoFile = open(myLog.infoFilePath,"w")
            infoFile.close()
            
    else:
        os.makedirs(myLog.output_dir_path)
    myLog.log("starting time", datetime.datetime.now())
    
def log(variableName,variableValue):
    myLog.log(variableName, variableValue)

def writeLog():
    myLog.writeLog()






