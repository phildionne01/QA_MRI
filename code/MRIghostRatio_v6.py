import matplotlib.pyplot as plt
import pydicom as dcm
import numpy as np
import os
from roipoly import *
import tkinter as tk 
from tkinter import filedialog

reuse_masks=0

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

print(style.YELLOW + 'Select the directory containing the correct dataset to analyze' + style.RESET)

root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

RefDs = dcm.read_file(lstFilesDCM[0])
if RefDs.StationName =='AWP183025':
  sl=20 #I don't know why the slices don't get ordered the same for both MRI scanners
else:
  sl=20

for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])

  filename = RefDs.SOPInstanceUID
  #print('filename ...', filename)

  instNum = RefDs.InstanceNumber
  #print('instNum ...', instNum)
  
  if int(instNum) == sl:
#     file_path = PathDicom +'/MR'+ filename + '.dcm'
     x = lstFilesDCM[f].split('\\')
     file = x[len(x)-1]  # last item
     file_path = PathDicom +'/'+ file
     print('file_pathEPI=', file_path)
     #ds=dcm.read_file(filenameDCM)
     im=RefDs.pixel_array
# Get ref file


# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
'''
# The array is sized based on 'ConstPixelDims'
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dcm.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  
print('MRI dataset size:....', np.shape(ArrayDicom))
im=np.squeeze(ArrayDicom[:, :, 9])
#plt.figure(dpi=300)
#plt.axes().set_aspect('equal', 'datalim')
#plt.set_cmap(plt.gray())
#plt.pcolormesh(x, y, np.flipud(ArrayDicom[:, :, 1]))
'''


if reuse_masks==0:
 plt.imshow(im, cmap=plt.cm.gray) #Draw a first ROI
 print('Draw an ROI at the centre of the phantom')
 ROI1 = RoiPoly(color='r')
 ROI1.show_figure()
 centre=ROI1.get_mean_and_std(im)
 mask1=ROI1.get_mask(im)

 plt.imshow(im, cmap=plt.cm.gray) #Draw the second ROI
 ROI1.display_roi()
 ROI1.display_mean(im)
 print('Draw an ROI to the left of the phantom')
 ROI2 = RoiPoly(color='b')
 ROI2.show_figure()
 left=ROI2.get_mean_and_std(im)
 mask2=ROI2.get_mask(im)

 plt.imshow(im, cmap=plt.cm.gray) #Draw the third ROI
 ROI1.display_roi()
 ROI1.display_mean(im)
 ROI2.display_roi()
 ROI2.display_mean(im)
 print('Draw a ROI to the right of the phantom')
 ROI3 = RoiPoly(color='g')
 ROI3.show_figure()
 right=ROI3.get_mean_and_std(im)
 mask3=ROI3.get_mask(im)

 plt.imshow(im, cmap=plt.cm.gray) #Draw the fourth ROI
 ROI1.display_roi()
 ROI1.display_mean(im)
 ROI2.display_roi()
 ROI2.display_mean(im)
 ROI3.display_roi()
 ROI3.display_mean(im)
 print('Draw a ROI above the top edge of the phantom')
 ROI4 = RoiPoly(color='m')
 ROI4.show_figure()
 top=ROI4.get_mean_and_std(im)
 mask4=ROI4.get_mask(im)


 plt.imshow(im, cmap=plt.cm.gray) #Draw the fifth ROI
 ROI1.display_mean(im)
 ROI1.display_roi()
 ROI2.display_mean(im)
 ROI2.display_roi()
 ROI3.display_mean(im)
 ROI3.display_roi()
 ROI4.display_mean(im)
 ROI4.display_roi()
 print('Draw a ROI below the bottom edge of the phantom')
 ROI5 = RoiPoly(color='c')
 ROI5.show_figure()
 bot=ROI5.get_mean_and_std(im)
 mask5=ROI5.get_mask(im)

 plt.imshow(im, cmap=plt.cm.gray) #Show all the final ROIs, displaying the ave and standard deviations
 ROI1.display_mean(im)
 ROI1.display_roi()
 ROI2.display_mean(im)
 ROI2.display_roi()
 ROI3.display_mean(im)
 ROI3.display_roi()
 ROI4.display_mean(im)
 ROI4.display_roi()
 ROI5.display_mean(im)
 ROI5.display_roi()

 plt.savefig(PathDicom + '/ROIs.png')
 plt.show()

 np.savetxt('mask1.txt', mask1, fmt='%d')
 np.savetxt('mask2.txt', mask2, fmt='%d')
 np.savetxt('mask3.txt', mask3, fmt='%d')
 np.savetxt('mask4.txt', mask4, fmt='%d')
 np.savetxt('mask5.txt', mask5, fmt='%d')

 GhostingRatio=np.abs((top[0]+bot[0])-(left[0]+right[0]))/(2*centre[0])*100
 print(style.YELLOW + 'Mean signal=', centre[0])
 print('Top signal=', top[0])
 print('Bottom signal=', bot[0])
 print('Left signal=', left[0])
 print('Right signal=', right[0])
 print('Ghosting radio=', GhostingRatio)

 floats=np.array([centre[0], top[0], bot[0], left[0], right[0], GhostingRatio])
 names=np.array(['Mean_signal=', 'Top_signal=', 'Bottom_signal=', 'Left_signal=', 'Right_signal=', 'GhostingRatio='])
 ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
 ab['var1'] = names
 ab['var2'] = floats
 np.savetxt(PathDicom + '/GhostingRatio.txt', ab, fmt='%20s %10.5f') 

elif reuse_masks==1:
 mask1=np.loadtxt('mask1.txt', dtype=int)
 mask2=np.loadtxt('mask2.txt', dtype=int)
 mask3=np.loadtxt('mask3.txt', dtype=int)
 mask4=np.loadtxt('mask4.txt', dtype=int)
 mask5=np.loadtxt('mask5.txt', dtype=int)
 im1=im*mask1
 im1=im1.flatten()
 centre=np.mean(im1[im1!=0])

 im2=im*mask2
 im2=im2.flatten()
 left=np.mean(im2[im2!=0])
 
 im3=im*mask3
 im3=im3.flatten()
 right=np.mean(im3[im3!=0])

 im4=im*mask4
 im4=im4.flatten()
 top=np.mean(im4[im4!=0])
 
 im5=im*mask5
 im5=im5.flatten()
 bot=np.mean(im5[im5!=0])

 GhostingRatio=np.abs((top+bot)-(left+right))/(2*centre)*100
 print(style.YELLOW + 'Mean signal=', centre)
 print('Top signal=', top)
 print('Bottom signal=', bot)
 print('Left signal=', left)
 print('Right signal=', right)
 print('Ghosting radio=', GhostingRatio)
 
 floats=np.array([centre, top, bot, left, right, GhostingRatio])
 names=np.array(['Mean_signal=', 'Top_signal=', 'Bottom_signal=', 'Left_signal=', 'Right_signal=', 'GhostingRatio='])
 ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
 ab['var1'] = names
 ab['var2'] = floats
 np.savetxt(PathDicom + '/GhostingRatio.txt', ab, fmt='%20s %10.5f')

if GhostingRatio <= 3: #The threshold for this test is set at 3%
	print(style.WHITE + 'QA test status...:'+ style.GREEN + 'pass' + style.RESET)
else:
	print(style.WHITE + 'QA test status...:'+ style.RED + 'fail'+ style.RESET)

