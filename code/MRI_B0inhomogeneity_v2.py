import numpy as np
import os
import pydicom as dcm
import matplotlib.pyplot as plt
import tkinter as tk 
from tkinter import filedialog
from scipy import ndimage
from skimage.restoration import unwrap_phase


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

print(style.YELLOW + 'Select the directory containing the phase of TE1'+ style.RESET)
root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

RefDs = dcm.read_file(lstFilesDCM[0])
if RefDs.Rows==RefDs.Columns-1:
  print(style.RED + 'Dicom export error. Will pad missing row of pixels with zeros' + style.RESET)
  ExportErrorFlag=1
else:
  ExportErrorFlag=0

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
#ArrayDicom = np.zeros(ConstPixelDims)
Bits=RefDs.BitsStored
ScanMode=RefDs.MRAcquisitionType
TE1=RefDs.EchoTime
print("MRI Scan mode:", ScanMode)

if RefDs.StationName =='AWP183025': 
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  #AcqNum = RefDs.AcquisitionNumber
  #seriesNum=RefDs.SeriesNumber
  #filename = RefDs.SOPInstanceUID
  #print('filename ...', filename)
  ArrayDicom[:,:,instNum-1]=RefDs.pixel_array
else:
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  #AcqNum = RefDs.AcquisitionNumber
  #seriesNum=RefDs.SeriesNumber
  #filename = RefDs.SOPInstanceUID
  #print('filename ...', filename)
  #SliceNumber=TotalSliceNumber-((AcqNum * TotalSliceNumber)-instNum)
  ArrayDicom[:,:,instNum-1]=RefDs.pixel_array
TE1phase=ArrayDicom
if ExportErrorFlag==1:
  TE1phase_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
  TE1phase_corr[1:,:,:]=TE1phase
  TE1phase=TE1phase_corr
print('MRI dataset size:....', np.shape(TE1phase))

print(style.YELLOW + 'Select the directory containing the phase of TE2'+ style.RESET)
root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

RefDs = dcm.read_file(lstFilesDCM[0])
if RefDs.Rows==RefDs.Columns-1:
  print(style.RED + 'Dicom export error. Will pad missing row of pixels with zeros' + style.RESET)
  ExportErrorFlag=1
else:
  ExportErrorFlag=0

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
Bits=RefDs.BitsStored
ScanMode=RefDs.MRAcquisitionType
TE2=RefDs.EchoTime
print("MRI Scan mode:", ScanMode)

if RefDs.StationName =='AWP183025': 
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  #AcqNum = RefDs.AcquisitionNumber
  #seriesNum=RefDs.SeriesNumber
  #filename = RefDs.SOPInstanceUID
  #print('filename ...', filename)
  ArrayDicom[:,:,instNum-1]=RefDs.pixel_array
else:
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  AcqNum = RefDs.AcquisitionNumber
  #seriesNum=RefDs.SeriesNumber
  #filename = RefDs.SOPInstanceUID
  #print('filename ...', filename)
  #SliceNumber=TotalSliceNumber-((AcqNum * TotalSliceNumber)-instNum)
  #print('instNum=', instNum)
  ArrayDicom[:,:,instNum-RefDs.Columns-1]=RefDs.pixel_array 
  #The Aera dicom files have instanceNumber that continues to iterate from 129 to 256 rather than restarting at 1
  #following the upgrade. Strange...
TE2phase=ArrayDicom
if ExportErrorFlag==1:
  TE2phase_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
  TE2phase_corr[1:,:,:]=TE2phase
  TE2phase=TE2phase_corr

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
print('MRI dataset size:....', np.shape(TE2phase))

print(style.YELLOW + 'Select the directory containing the magnitude of TE1'+ style.RESET)
root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

RefDs = dcm.read_file(lstFilesDCM[0])
if RefDs.Rows==RefDs.Columns-1:
  print(style.RED + 'Dicom export error. Will pad missing row of pixels with zeros' + style.RESET)
  ExportErrorFlag=1
else:
  ExportErrorFlag=0

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
Bits=RefDs.BitsStored
ScanMode=RefDs.MRAcquisitionType
print("MRI Scan mode:", ScanMode)

if RefDs.StationName =='AWP183025': 
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  #AcqNum = RefDs.AcquisitionNumber
  #seriesNum=RefDs.SeriesNumber
  #filename = RefDs.SOPInstanceUID
  #print('filename ...', filename)
  ArrayDicom[:,:,instNum-1]=RefDs.pixel_array
else:
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  #AcqNum = RefDs.AcquisitionNumber
  #seriesNum=RefDs.SeriesNumber
  #filename = RefDs.SOPInstanceUID
  #print('filename ...', filename)
  #SliceNumber=TotalSliceNumber-((AcqNum * TotalSliceNumber)-instNum)
  ArrayDicom[:,:,instNum-1]=RefDs.pixel_array 
TE1mag=ArrayDicom
if ExportErrorFlag==1:
  TE1mag_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
  TE1mag_corr[1:,:,:]=TE1mag
  TE1mag=TE1mag_corr

mask=TE1mag<200 #Can adjust this threshold which will influence the peak-to-peak calculation
# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
#plt.imshow(np.squeeze(TE1mag[:,:,64]), cmap=plt.cm.gray)
#plt.show()

print('MRI dataset size:....', np.shape(TE1mag))

if int(Bits)==12:
 nBits=4096
elif int(Bits)==16:
 nBits=65536

if ScanMode == "2D":
 print(style.YELLOW + "Scan mode was 2D, therefore 2D phase-unwrapping will be performed slice-by-slice" + style.RESET)
 unwrapped_series=np.zeros(np.shape(TE1phase))
 for i in range(TE1phase.shape[2]):
  arr_phase1=2*np.pi/nBits*np.float32(TE1phase[:,:,i])
  unwrapped_phas1[:,:,i]=unwrap_phase(arr_phase1)
  arr_phase2=2*np.pi/nBits*np.float32(TE2phase[:,:,i])
  unwrapped_phase2[:,:,i]=unwrap_phase(arr_phase2)

elif ScanMode == "3D":
 print(style.YELLOW + "Scan mode was 3D, therefore 3D phase-unwrapping will be performed" + style.RESET)
 arr_phase1=2*np.pi/nBits*np.float32(TE1phase)
 unwrapped_phase1=unwrap_phase(arr_phase1)
 arr_phase2=2*np.pi/nBits*np.float32(TE2phase)
 unwrapped_phase2=unwrap_phase(arr_phase2)
gamma=2.67522e8
DeltaTE=(TE2-TE1)/1000
f0=RefDs.ImagingFrequency
#print('f0=',f0)
#print('DeltaTE=', DeltaTE)
unity=np.ones(np.shape(arr_phase1))
DeltaPhi=unwrapped_phase2-unwrapped_phase1
#plt.imshow(np.squeeze(DeltaPhi[:,:,64]), cmap=plt.cm.gray)
#plt.show()

DeltaPhi[mask]=1
DeltaPhi_vec=DeltaPhi.flatten()
DeltaPhi_vec2=np.delete(DeltaPhi_vec, np.where(DeltaPhi_vec==1))
DeltaPhiMean=np.mean(DeltaPhi_vec2)
print('mean=', DeltaPhiMean)
if DeltaPhiMean>5: #Check that there is no 2pi offset in the map
  DeltaPhi=DeltaPhi-2*np.pi*unity
elif DeltaPhiMean<-5: #Check that there is no -2pi offset in the map
  DeltaPhi=DeltaPhi+2*np.pi*unity


#DeltaB0=DeltaPhi/(gamma*DeltaTE)/1.5*1e6
DeltaB0=DeltaPhi/(2*np.pi*42.576e6*DeltaTE)/(f0/42.576)*1e6
DeltaB0[mask]=-1
DeltaB0_vec=DeltaB0.flatten()
DeltaB0_vec2=np.delete(DeltaB0_vec, np.where(DeltaB0_vec==-1))

DeltaB0rms=np.sqrt(np.sum(DeltaB0_vec2**2)/DeltaB0_vec2.shape[0])
DeltaB0mean=np.mean(DeltaB0_vec2)
DeltaB0std=np.std(DeltaB0_vec2)
#DeltaB0pk2pk=DeltaB0std*4*2 #Assume 4 times the standard deviation on each side
DeltaB0pk2pk=np.max([np.percentile(DeltaB0_vec2,99.7), -np.percentile(DeltaB0_vec2,0.3)])

print('Number of Pixels=', DeltaB0_vec2.shape[0])
print('Delta B0 RMS[ppm]=', DeltaB0rms)
print('Delta B0 mean[ppm]=', DeltaB0mean)
print('Delta B0 std[ppm]=', DeltaB0std)
print('Delta B0 peak-to-peak[ppm]=', DeltaB0pk2pk)

print(style.RESET + 'Results stored in textfile of last folder opened')
floats=np.array([DeltaB0_vec2.shape[0], DeltaB0rms, DeltaB0mean, DeltaB0std, DeltaB0pk2pk])
names=np.array(['Number of Pixels=', 'Delta B0 RMS[ppm]=', 'Delta B0 mean[ppm]=', \
'Delta B0 std[ppm]=', 'Delta B0 peak-to-peak[ppm]='])

ab=np.zeros(names.size, dtype=[('var1', 'U27'), ('var2', float)])
ab['var1'] = names
ab['var2'] = floats
np.savetxt(PathDicom + '/B0_homogeneity_results.txt', ab, fmt='%20s %10.5f')

hist_bins=np.linspace(-9e-1,9e-1,751)
#n,_=np.histogram(Delta_B0,hist_bins)
fig, axs=plt.subplots(3,2)
axs[0,0].imshow(np.squeeze(TE1phase[:,:,64]), cmap=plt.cm.bone, vmin=0, vmax=4096)
axs[0,0].set_title('Phase image axial')
axs[0,1].imshow(np.squeeze(TE1phase[:,64,:]), cmap=plt.cm.bone, vmin=0, vmax=4096)
axs[0,1].set_title('Phase image sag')

axs[1,0].imshow(np.squeeze(unwrapped_phase1[:,:,64]), cmap=plt.cm.bone, vmin=0, vmax=4*np.pi)
axs[1,0].set_title('Unwrapped phase image axial')
axs[1,1].imshow(np.squeeze(unwrapped_phase1[:,64,:]), cmap=plt.cm.bone, vmin=0, vmax=4*np.pi)
axs[1,1].set_title('Unwrapped phase image sag')

axs[2,0].imshow(np.squeeze(DeltaB0[:,:,64]), cmap=plt.cm.bone, vmin=-5e-1, vmax=5e-1)
axs[2,0].set_title('B0 map axial')
axs[2,1].imshow(np.squeeze(DeltaB0[:,64,:]), cmap=plt.cm.bone, vmin=-5e-1, vmax=5e-1)
axs[2,1].set_title('B0 map sag')
fig=plt.gcf()
fig.set_size_inches(7,9)
plt.savefig(PathDicom +'/Images.png', dpi=300)
plt.show()
plt.close()

plt.hist(DeltaB0.flatten(), hist_bins)
plt.title('B0 inhomogeneity in ppm')
plt.savefig(PathDicom +'/DeltaB0hist.png')
plt.show()
