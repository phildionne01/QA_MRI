#dcmsort.py
import numpy as np
import glob, os
import tkinter as tk 
from tkinter import filedialog
import pydicom as dcm
import shutil

os.system("") #This section of code is just to change the color that appears in the command window
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

print(style.YELLOW + 'Select the directory containing all your MR dicom data to sort'+ style.RESET)

root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()

# An input is requested and stored in a variable
#Dyn_vol_in = input ("Entrer le numéro de la dynamique voulu: ")

# Converts the string into a integer. If you need
# to convert the user input into decimal format,
# the float() function is used instead of int()
#Dyn_vol = int(Dyn_vol_in)

file_count=0
os.chdir(PathDicom)
for file in glob.glob('*.dcm'):
	file_count=file_count+1

print(style.YELLOW + 'DICOM file count =' + style.RESET, file_count)
runtime=file_count/182
file_count_max=file_count
print(style.YELLOW + 'Estimated runtime [min] =' + style.RESET, runtime)

print(style.YELLOW+'Directory nomenclature: '+style.GREEN+' ProtocolName_'+style.CYAN+'ScanDate_'+style.MAGENTA+'SeriesNumber_'+style.BLUE+'EchoTime_'+style.RED+'Dynamic_'+style.WHITE+'Phase/Magnitude'+style.RESET)
while file_count!=0:   	
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))
  	 
    RefDs0=dcm.read_file(lstFilesDCM[0])
    im_type=RefDs0.ImageType[2]
    parent_dir=PathDicom
    patient_id=RefDs0.PatientID
    series_num=RefDs0.SeriesNumber
    AcquisitionNumber=RefDs0.AcquisitionNumber
    date=RefDs0.PerformedProcedureStepStartDate
    
   # if AcquisitionNumber!=Dyn_vol:
   # 	delete
    
    directory=RefDs0.ProtocolName+'_'+str(RefDs0.PerformedProcedureStepStartDate)+'_'+str(RefDs0.SeriesNumber)+'_'+str(RefDs0.EchoTime)+'_'+str(RefDs0.AcquisitionNumber)+'_'+im_type
    TE=RefDs0.EchoTime

    daughter_dir=os.path.join(parent_dir, directory)
    os.mkdir(daughter_dir)
  
    for f in range(len(lstFilesDCM)):
       RefDs=dcm.read_file(lstFilesDCM[f])
       filename = RefDs.SOPInstanceUID

       if RefDs.ProtocolName==RefDs0.ProtocolName and RefDs.PatientID==patient_id and RefDs.PerformedProcedureStepStartDate==date and RefDs.SeriesNumber==series_num and RefDs.EchoTime==TE and RefDs.AcquisitionNumber==AcquisitionNumber and RefDs.ImageType[2]==im_type:
           shutil.move(parent_dir +'/MR.'+filename +'.dcm', daughter_dir +'/MR.'+filename +'.dcm')
           file_count=file_count-1
       elif RefDs.PatientID!=patient_id:
       	   print(style.RED + 'Warning! Same sequence on different patients.' + style.RESET)
       	   break
       elif RefDs.PerformedProcedureStepStartDate!=date:
           print(style.RED + 'Warning! Same patient scanned on two different dates.'+ style.RESET)
           break
       #elif RefDs.AcquisitionNumber!=Dyn_vol:
         #  shutil.rmtree(parent_dir +'/MR.'+filename +'.dcm', daughter_dir +'/MR.'+filename +'.dcm')
         #  file_count=file_count-1
         #  break
    

    
  