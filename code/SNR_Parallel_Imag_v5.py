import matplotlib.pyplot as plt
import pydicom as dcm
import numpy as np
import os
from roipoly import *
import tkinter as tk 
from tkinter import filedialog

#Turn on this switch to reuse the same ROIs:
reuse_masks=1

#sl=105 #use this slice for mprage aera
#sl=105 #use thus slice for mprage sola
#sl=33 #use this slice for space sola

# Group of Different functions for different styles
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

print(style.YELLOW + 'Select the directory containing the two dynamics without acceleration' + style.RESET)

root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
RefDs = dcm.read_file(lstFilesDCM[71])
if RefDs.Rows==RefDs.Columns-1:
  print(style.RED + 'Dicom export error. Will pad missing row of pixels with zeros' + style.RESET)
  ExportErrorFlag=1
else:
  ExportErrorFlag=0

#To correct a dicom bug where the slices arrive in reverse order.
if (RefDs.SliceLocation < 0) and (RefDs.ProtocolName[0:8]=='t2_space'):
  sl=33
elif (RefDs.SliceLocation < 0) and (RefDs.ProtocolName[0:8]=='t1_mprag'):
  sl=105
elif (RefDs.SliceLocation > 0) and (RefDs.ProtocolName[0:8]=='t2_space'):
  sl=55
elif (RefDs.SliceLocation > 0) and (RefDs.ProtocolName[0:8]=='t1_mprag'):
  sl=71

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), int(len(lstFilesDCM)/2), 2)
TotalSliceNumber = int(len(lstFilesDCM)/2)

ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
if RefDs.StationName =='AWP183025': 
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  AcqNum = RefDs.AcquisitionNumber
  #seriesNum=RefDs.SeriesNumber
  #filename = RefDs.SOPInstanceUID
  #print('filename ...', filename)
  ArrayDicom[:,:,instNum-1, AcqNum-1]=RefDs.pixel_array

else: 
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  AcqNum = RefDs.AcquisitionNumber
  #seriesNum=RefDs.SeriesNumber
  #filename = RefDs.SOPInstanceUID
  #print('filename ...', filename)
  SliceNumber=TotalSliceNumber-((AcqNum * TotalSliceNumber)-instNum)
  ArrayDicom[:,:,SliceNumber-1,AcqNum-1]=RefDs.pixel_array 

image_ref1=np.float32(np.squeeze(ArrayDicom[:,:,:,0]))
image_ref2=np.float32(np.squeeze(ArrayDicom[:,:,:,1]))

if ExportErrorFlag==1:
	image_ref1_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
	image_ref2_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
	image_ref1_corr[1:,:,:]=image_ref1
	image_ref2_corr[1:,:,:]=image_ref2
	image_ref1=image_ref1_corr
	image_ref2=image_ref2_corr

noise_image_ref=(image_ref1-image_ref2)
#print(RefDs)

print('MRI ref dataset size:....', np.shape(ArrayDicom))

print(style.YELLOW + 'Select the directory containing the two dynamics with acceleration' + style.RESET)

root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
RefDs = dcm.read_file(lstFilesDCM[0])
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), int(len(lstFilesDCM)/2), 2)
TotalSliceNumber = int(len(lstFilesDCM)/2)
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

RefDs=dcm.read_file(lstFilesDCM[0])
if RefDs.Rows==RefDs.Columns-1:
  print(style.RED + 'Dicom export error. Will pad missing row of pixels with zeros' + style.RESET)
  ExportErrorFlag=1
else:
  ExportErrorFlag=0

print('Sequence:', RefDs.ProtocolName[0:6])
print('MRI scanner [IRM-sim = AWP183025, IRM-curie = AWP142447]:', RefDs.StationName)

if RefDs.StationName =='AWP183025': 
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  AcqNum = RefDs.AcquisitionNumber
  ArrayDicom[:,:,instNum-1,AcqNum-1]=RefDs.pixel_array
else:
 for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  instNum = RefDs.InstanceNumber
  AcqNum = RefDs.AcquisitionNumber
  SliceNumber=TotalSliceNumber-((AcqNum * TotalSliceNumber)-instNum)
  ArrayDicom[:,:,SliceNumber-1,AcqNum-1]=RefDs.pixel_array 

print('MRI noise dataset size:....', np.shape(ArrayDicom))

image_accel1=np.float32(np.squeeze(ArrayDicom[:,:,:,0]))
image_accel2=np.float32(np.squeeze(ArrayDicom[:,:,:,1]))

if ExportErrorFlag==1:
	image_accel1_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
	image_accel2_corr=np.zeros((int(RefDs.Rows)+1, int(RefDs.Columns), len(lstFilesDCM)))
	image_accel1_corr[1:,:,:]=image_accel1
	image_accel2_corr[1:,:,:]=image_accel2
	image_accel1=image_accel1_corr
	image_accel2=image_accel2_corr

noise_image_accel=(image_accel1-image_accel2)

fig, axs=plt.subplots(1,2)
axs[0].imshow(np.squeeze(image_ref1[:,:,sl]))
axs[0].set_title('average image')
axs[1].imshow(np.squeeze(noise_image_ref[:,:,sl]))
axs[1].set_title('noise image')
plt.show()

im1=np.squeeze(image_ref1[:,:,sl]+image_ref2[:,:,sl])/2
im1_noise=np.squeeze(noise_image_ref[:,:,sl])
im2=np.squeeze(image_accel1[:,:,sl]+image_accel2[:,:,sl])/2
im2_noise=np.squeeze(noise_image_accel[:,:,sl])



if reuse_masks==0:
 plt.imshow(im1, cmap=plt.cm.gray) #Draw a first ROI
 print('Draw an ROI at the centre of the phantom')
 ROI1 = RoiPoly(color='r')
 ROI1.show_figure()
 centre=ROI1.get_mean_and_std(im1)
 mask1=ROI1.get_mask(im1)

 plt.imshow(im1, cmap=plt.cm.gray) #Draw the second ROI
 ROI1.display_roi()
 ROI1.display_mean(im1)
 print('Draw an ROI to the left of the phantom')
 ROI2 = RoiPoly(color='b')
 ROI2.show_figure()
 left=ROI2.get_mean_and_std(im1)
 mask2=ROI2.get_mask(im1)

 plt.imshow(im1, cmap=plt.cm.gray) #Draw the third ROI
 ROI1.display_roi()
 ROI1.display_mean(im1)
 ROI2.display_roi()
 ROI2.display_mean(im1)
 print('Draw a ROI to the right of the phantom')
 ROI3 = RoiPoly(color='g')
 ROI3.show_figure()
 right=ROI3.get_mean_and_std(im1)
 mask3=ROI3.get_mask(im1)

 plt.imshow(im1, cmap=plt.cm.gray) #Draw the fourth ROI
 ROI1.display_roi()
 ROI1.display_mean(im1)
 ROI2.display_roi()
 ROI2.display_mean(im1)
 ROI3.display_roi()
 ROI3.display_mean(im1)
 print('Draw a ROI above the top edge of the phantom')
 ROI4 = RoiPoly(color='m')
 ROI4.show_figure()
 top=ROI4.get_mean_and_std(im1)
 mask4=ROI4.get_mask(im1)


 plt.imshow(im1, cmap=plt.cm.gray) #Draw the fifth ROI
 ROI1.display_mean(im1)
 ROI1.display_roi()
 ROI2.display_mean(im1)
 ROI2.display_roi()
 ROI3.display_mean(im1)
 ROI3.display_roi()
 ROI4.display_mean(im1)
 ROI4.display_roi()
 print('Draw a ROI below the bottom edge of the phantom')
 ROI5 = RoiPoly(color='c')
 ROI5.show_figure()
 bot=ROI5.get_mean_and_std(im1)
 mask5=ROI5.get_mask(im1)

 plt.imshow(im1, cmap=plt.cm.gray) #Show all the final ROIs, displaying the ave and standard deviations
 ROI1.display_mean(im1)
 ROI1.display_roi()
 ROI2.display_mean(im1)
 ROI2.display_roi()
 ROI3.display_mean(im1)
 ROI3.display_roi()
 ROI4.display_mean(im1)
 ROI4.display_roi()
 ROI5.display_mean(im1)
 ROI5.display_roi()

 plt.savefig(PathDicom + '/ROIs_SNR.tiff', dpi=600, format='tiff')
 plt.savefig(PathDicom + '/ROIs_SNR.svg', format='svg')
 plt.show()

 if RefDs.ProtocolName[0:8]=='t2_space':
   np.savetxt('SNRmask1_space.txt', mask1, fmt='%d')
   np.savetxt('SNRmask2_space.txt', mask2, fmt='%d')
   np.savetxt('SNRmask3_space.txt', mask3, fmt='%d')
   np.savetxt('SNRmask4_space.txt', mask4, fmt='%d')
   np.savetxt('SNRmask5_space.txt', mask5, fmt='%d')

 else:
   np.savetxt('SNRmask1_mprage.txt', mask1, fmt='%d')
   np.savetxt('SNRmask2_mprage.txt', mask2, fmt='%d')
   np.savetxt('SNRmask3_mprage.txt', mask3, fmt='%d')
   np.savetxt('SNRmask4_mprage.txt', mask4, fmt='%d')
   np.savetxt('SNRmask5_mprage.txt', mask5, fmt='%d')

else:

 if RefDs.ProtocolName[0:8]=='t2_space':
   mask1=np.loadtxt('SNRmask1_space.txt', dtype=int)
   mask2=np.loadtxt('SNRmask2_space.txt', dtype=int)
   mask3=np.loadtxt('SNRmask3_space.txt', dtype=int)
   mask4=np.loadtxt('SNRmask4_space.txt', dtype=int)
   mask5=np.loadtxt('SNRmask5_space.txt', dtype=int)
 else:
   mask1=np.loadtxt('SNRmask1_mprage.txt', dtype=int)
   mask2=np.loadtxt('SNRmask2_mprage.txt', dtype=int)
   mask3=np.loadtxt('SNRmask3_mprage.txt', dtype=int)
   mask4=np.loadtxt('SNRmask4_mprage.txt', dtype=int)
   mask5=np.loadtxt('SNRmask5_mprage.txt', dtype=int)

im1_mean_roi1=im1*mask1
im1_mean_vec1=im1_mean_roi1.flatten()
im1_mean1=np.mean(im1_mean_vec1[im1_mean_vec1!=0])

im1_mean_roi2=im1*mask2
im1_mean_vec2=im1_mean_roi2.flatten()
im1_mean2=np.mean(im1_mean_vec2[im1_mean_vec2!=0])

im1_mean_roi3=im1*mask3
im1_mean_vec3=im1_mean_roi3.flatten()
im1_mean3=np.mean(im1_mean_vec3[im1_mean_vec3!=0])

im1_mean_roi4=im1*mask4
im1_mean_vec4=im1_mean_roi4.flatten()
im1_mean4=np.mean(im1_mean_vec4[im1_mean_vec4!=0])

im1_mean_roi5=im1*mask5
im1_mean_vec5=im1_mean_roi5.flatten()
im1_mean5=np.mean(im1_mean_vec5[im1_mean_vec5!=0])


im1_noise_roi1=im1_noise*mask1
im1_noise_vec1=im1_noise_roi1.flatten()
im1_std_dev1=np.std(im1_noise_vec1[im1_noise_vec1!=0])/np.sqrt(2)

im2_mean_roi1=im2*mask1
im2_mean_vec1=im2_mean_roi1.flatten()
im2_mean1=np.mean(im2_mean_vec1[im2_mean_vec1!=0])

im2_noise_roi1=im2_noise*mask1
im2_noise_vec1=im2_noise_roi1.flatten()
im2_std_dev1=np.std(im2_noise_vec1[im2_noise_vec1!=0])/np.sqrt(2)


im1_noise_roi2=im1_noise*mask2
im1_noise_vec2=im1_noise_roi2.flatten()
im1_std_dev2=np.std(im1_noise_vec2[im1_noise_vec2!=0])/np.sqrt(2)

im2_mean_roi2=im2*mask2
im2_mean_vec2=im2_mean_roi2.flatten()
im2_mean2=np.mean(im2_mean_vec2[im2_mean_vec2!=0])

im2_noise_roi2=im2_noise*mask2
im2_noise_vec2=im2_noise_roi2.flatten()
im2_std_dev2=np.std(im2_noise_vec2[im2_noise_vec2!=0])/np.sqrt(2)


im1_noise_roi3=im1_noise*mask3
im1_noise_vec3=im1_noise_roi3.flatten()
im1_std_dev3=np.std(im1_noise_vec3[im1_noise_vec3!=0])/np.sqrt(2)

im2_mean_roi3=im2*mask3
im2_mean_vec3=im2_mean_roi3.flatten()
im2_mean3=np.mean(im2_mean_vec3[im2_mean_vec3!=0])

im2_noise_roi3=im2_noise*mask3
im2_noise_vec3=im2_noise_roi3.flatten()
im2_std_dev3=np.std(im2_noise_vec3[im2_noise_vec3!=0])/np.sqrt(2)

im1_noise_roi4=im1_noise*mask4
im1_noise_vec4=im1_noise_roi4.flatten()
im1_std_dev4=np.std(im1_noise_vec4[im1_noise_vec4!=0])/np.sqrt(2)

im2_mean_roi4=im2*mask4
im2_mean_vec4=im2_mean_roi4.flatten()
im2_mean4=np.mean(im2_mean_vec4[im2_mean_vec4!=0])

im2_noise_roi4=im2_noise*mask4
im2_noise_vec4=im2_noise_roi4.flatten()
im2_std_dev4=np.std(im2_noise_vec4[im2_noise_vec4!=0])/np.sqrt(2)


im1_noise_roi5=im1_noise*mask5
im1_noise_vec5=im1_noise_roi5.flatten()
im1_std_dev5=np.std(im1_noise_vec5[im1_noise_vec5!=0])/np.sqrt(2)

im2_mean_roi5=im2*mask5
im2_mean_vec5=im2_mean_roi5.flatten()
im2_mean5=np.mean(im2_mean_vec5[im2_mean_vec5!=0])

im2_noise_roi5=im2_noise*mask5
im2_noise_vec5=im2_noise_roi5.flatten()
im2_std_dev5=np.std(im2_noise_vec5[im2_noise_vec5!=0])/np.sqrt(2)

SNR_ref1=im1_mean1/im1_std_dev1
print(style.YELLOW +'SNR ref ROI1='+ style.GREEN, SNR_ref1)

SNR_accel1=im2_mean1/im2_std_dev1
print(style.YELLOW +'SNR accel ROI1='+ style.GREEN, SNR_accel1)

SNR_ratio1=SNR_ref1/SNR_accel1
print(style.YELLOW +'SNR_ratio2='+ style.GREEN, SNR_ratio1)

#g_factor1=SNR_ref1/SNR_accel1/(np.sqrt(2))
#print(style.YELLOW +'g_factor1='+ style.GREEN, g_factor1)

SNR_ref2=im1_mean2/im1_std_dev2
print(style.YELLOW +'SNR ref ROI2='+ style.GREEN, SNR_ref2)

SNR_accel2=im2_mean2/im2_std_dev2
print(style.YELLOW +'SNR accel ROI2='+ style.GREEN, SNR_accel2)

SNR_ratio2=SNR_ref2/SNR_accel2
print(style.YELLOW +'SNR_ratio2='+ style.GREEN, SNR_ratio2)

#g_factor2=SNR_ref2/SNR_accel2/(np.sqrt(2))
#print(style.YELLOW +'g_factor2='+ style.GREEN, g_factor2)

SNR_ref3=im1_mean3/im1_std_dev3
print(style.YELLOW +'SNR ref ROI3='+ style.GREEN, SNR_ref3)

SNR_accel3=im2_mean3/im2_std_dev3
print(style.YELLOW +'SNR accel ROI3='+ style.GREEN, SNR_accel3)

SNR_ratio3=SNR_ref3/SNR_accel3
print(style.YELLOW +'SNR_ratio3='+ style.GREEN, SNR_ratio3)

#g_factor3=SNR_ref3/SNR_accel3/(np.sqrt(2))
#print(style.YELLOW +'g_factor3='+ style.GREEN, g_factor3)

SNR_ref4=im1_mean4/im1_std_dev4
print(style.YELLOW +'SNR ref ROI4='+ style.GREEN, SNR_ref4)

SNR_accel4=im2_mean4/im2_std_dev4
print(style.YELLOW +'SNR accel ROI4='+ style.GREEN, SNR_accel4)

SNR_ratio4=SNR_ref4/SNR_accel4
print(style.YELLOW +'SNR_ratio4='+ style.GREEN, SNR_ratio4)

#g_factor4=SNR_ref4/SNR_accel4/(np.sqrt(2))
#print(style.YELLOW +'g_factor4='+ style.GREEN, g_factor4)

SNR_ref5=im1_mean5/im1_std_dev5
print(style.YELLOW +'SNR ref ROI5='+ style.GREEN, SNR_ref5)

SNR_accel5=im2_mean5/im2_std_dev5
print(style.YELLOW +'SNR accel ROI5='+ style.GREEN, SNR_accel5)

SNR_ratio5=SNR_ref5/SNR_accel5
print(style.YELLOW +'SNR_ratio5='+ style.GREEN, SNR_ratio5)

#g_factor5=SNR_ref5/SNR_accel5/(np.sqrt(2))
#print(style.YELLOW +'g_factor5='+ style.GREEN, g_factor5)

print(style.RESET + 'Data stored in textfile of last folder opened')
floats=np.array([SNR_ref1, SNR_accel1, SNR_ratio1, SNR_ref2, SNR_accel2, SNR_ratio2, SNR_ref3,\
SNR_accel3, SNR_ratio3, SNR_ref4, SNR_accel4, SNR_ratio4, SNR_ref5, SNR_accel5, SNR_ratio5])
names=np.array(['Centre_SNR_ref=', 'Centre_SNR_accel=', 'Centre_ratio=', 'Left_SNR_ref=', 'Left_SNR_accel=', 'Left_ratio=', 'Right_SNR_ref=', 'Right_SNR_accel=', \
'Right_ratio=', 'Top_SNR_ref=', 'Top_SNR_accel=', 'Top_ratio=', 'Bottom_SNR_ref=', 'Bottom_SNR_accel=', 'Bottom_ratio='])

ab=np.zeros(names.size, dtype=[('var1', 'U20'), ('var2', float)])
ab['var1'] = names
ab['var2'] = floats
np.savetxt(PathDicom + '/ROI_SNR_results.txt', ab, fmt='%20s %10.5f')

fig, axs=plt.subplots(2,2)
fig.set_size_inches(5.2,5.5)
imA=axs[0,0].imshow(im1_noise, cmap=plt.cm.gray, vmin=-15, vmax=15)
axs[0,0].set_title('noise image no accel')
imB=axs[0,1].imshow(im1, cmap=plt.cm.gray)
axs[0,1].set_title('image no accel')
imC=axs[1,0].imshow(im2_noise, cmap=plt.cm.gray, vmin=-15, vmax=15)
axs[1,0].set_title('noise image accel')
imD=axs[1,1].imshow(im2, cmap=plt.cm.gray)
axs[1,1].set_title('image accel')
fig=plt.gcf()
fig.colorbar(imA, ax=axs[0,0])
fig.colorbar(imB, ax=axs[0,1])
fig.colorbar(imC, ax=axs[1,0])
fig.colorbar(imD, ax=axs[1,1])
plt.savefig(PathDicom + '/Images.tiff', dpi=600, format='tiff')
plt.savefig(PathDicom + '/Images.svg', format='svg')
plt.show()
plt.close()