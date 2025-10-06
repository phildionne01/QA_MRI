import numpy as np
import os
import pydicom as dcm
import matplotlib.pyplot as plt
import tkinter as tk 
from tkinter import filedialog
from scipy import ndimage

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

print(style.YELLOW + 'Select the first image dataset'+ style.RESET)
root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

RefDs = dcm.read_file(lstFilesDCM[0])
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), int(len(lstFilesDCM)))
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype) 

if RefDs.Rows==RefDs.Columns-1:
  print(style.RED + 'Dicom export error. Will pad missing row of pixels with zeros' + style.RESET)
  ExportErrorFlag=1
else:
  ExportErrorFlag=0

for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  ArrayDicom[:,:,instNum-1]=RefDs.pixel_array

im1=ArrayDicom
if ExportErrorFlag==1:
  im1_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
  im1_corr[1:,:,:]=im1
  im1=im1_corr
print('MRI dataset size:....', np.shape(im1))

print(style.YELLOW + 'Select the second image dataset'+ style.RESET)
root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

RefDs = dcm.read_file(lstFilesDCM[0])
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), int(len(lstFilesDCM)))
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype) 

if RefDs.Rows==RefDs.Columns-1:
  print(style.RED + 'Dicom export error. Will pad missing row of pixels with zeros' + style.RESET)
  ExportErrorFlag=1
else:
  ExportErrorFlag=0

for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  ArrayDicom[:,:,instNum-1]=RefDs.pixel_array

im2=ArrayDicom
if ExportErrorFlag==1:
  im2_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
  im2_corr[1:,:,:]=im2
  im2=im2_corr
print('MRI dataset size:....', np.shape(im2))
'''
print(style.YELLOW + 'Select the third image dataset'+ style.RESET)
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

for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  ArrayDicom[:,:,instNum-1]=RefDs.pixel_array

im2=ArrayDicom
if ExportErrorFlag==1:
  im3_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
  im3_corr[1:,:,:]=im3
  im3=im3_corr
print('MRI dataset size:....', np.shape(im3))
'''
sl=100
fig, axs=plt.subplots(3,2)
axs[0,0].imshow(np.squeeze(im1[:,:,100]), cmap='gray', vmin=0, vmax=1000)
axs[0,0].set_title('fse2d_axial')
axs[1,0].imshow(np.squeeze(im1[:,256,:]), cmap='gray', vmin=0, vmax=1000)
axs[1,0].set_title('fse2d_sag')
axs[2,0].imshow(np.squeeze(im1[293,:,:]), cmap='gray', vmin=0, vmax=1000)
axs[2,0].set_title('fse2d_cor')
axs[0,1].imshow(np.squeeze(im2[:,:,100]), cmap='gray', vmin=0, vmax=1000)
axs[0,1].set_title('fse3d_axial')
axs[1,1].imshow(np.squeeze(im2[:,256,:]), cmap='gray', vmin=0, vmax=1000)
axs[1,1].set_title('fse3d_sag')
axs[2,1].imshow(np.squeeze(im2[293,:,:]), cmap='gray', vmin=0, vmax=1000)
axs[2,1].set_title('fse3d_cor')


#imC=axs[2,0].imshow(np.squeeze(im3[:,:,sl]), cmap='gray', vmin=0, vmax=4000)
#axs[2,0].set_title('Title im3')

fig=plt.gcf()
#fig.set_size_inches(7,9)
#fig.colorbar(imA, ax=axs[0,0])
#fig.colorbar(imB, ax=axs[0,1])
#fig.colorbar(imC, ax=axs[1,0])

plt.savefig(PathDicom +'/Images.svg', format='svg')
plt.savefig(PathDicom +'/Images.tiff', dpi=600, format='tiff')
plt.show()
plt.close()
