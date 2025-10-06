import numpy as np
import os
import pydicom as dcm
import matplotlib.pyplot as plt
import tkinter as tk 
from tkinter import filedialog
from scipy import ndimage
from scipy.fftpack import fft, fftshift, fftfreq

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

print(style.YELLOW + 'Select the directory containing the correct dataset to analyze'+ style.RESET)

root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
# Get ref file
AcqNum = int(0)
Im_num=int(len(lstFilesDCM))

for f in range(Im_num):
  RefDs = dcm.read_file(lstFilesDCM[f])
  if f==0:
    x=RefDs.Rows
    y=RefDs.Columns
    console=RefDs.StationName
  #print('filename ...', filename)
  #print('instNum ...', instNum)
  if AcqNum < int(RefDs.AcquisitionNumber):
     AcqNum = int(RefDs.AcquisitionNumber)

TotalSliceNumber = int(Im_num / AcqNum)
ArrayDicom=np.zeros((x,y,TotalSliceNumber,AcqNum))

# loop through all the DICOM files
print('TotalSliceNumber=', TotalSliceNumber)
print('AcquisitionNumber max=', AcqNum)
print('Im_num=', Im_num)

#if console == 'AWP183025':
# sl=18
# for filenameDCM in lstFilesDCM:
#    # read the file
#     ds = dcm.read_file(filenameDCM)
#     InstanceNum=int(ds.InstanceNumber)
#     AcquisitionNumber=int(ds.AcquisitionNumber)
#     SliceNumber=InstanceNum
#     ArrayDicom[:, :, SliceNumber-1, AcquisitionNumber-1] = ds.pixel_array
#else:
sl=18
for filenameDCM in lstFilesDCM:
    # read the file
     ds = dcm.read_file(filenameDCM)
     InstanceNum=int(ds.InstanceNumber)
     AcquisitionNumber=int(ds.AcquisitionNumber)
     SliceNumber=TotalSliceNumber-((AcquisitionNumber * TotalSliceNumber)-InstanceNum)
     ArrayDicom[:, :, SliceNumber-1, AcquisitionNumber-1] = ds.pixel_array
    
print('MRI dataset size:....', np.shape(ArrayDicom))

arr0=np.squeeze(ArrayDicom[:,:,sl,:])
skip=2 # Skip the first few images in the series. Must be an even number to work.
s0=np.shape(arr0)
arr=arr0[:,:,skip:]

s=np.shape(arr)
x=s[0]
y=s[1]
z=s[2]
print("Number in original time series =",s0[2])
print("Number in new time series =",z)
signal_im=np.mean(arr, axis=2)
temp_even=np.zeros((x,y,np.uint16(z/2)))
temp_odd=np.zeros((x,y,np.uint16(z/2)))

for j in range(z): #Separate even and odd images in the time series
  if np.fmod(j,2)==0:
   i=np.uint16(j/2)
   temp_even[:,:,i]=arr[:,:,j]
  elif np.fmod(j,2)!=0: 
   i=np.uint16((j-1)/2)
   temp_odd[:,:,i]=arr[:,:,j]

sum_even=np.sum(temp_even, axis=2)
sum_odd=np.sum(temp_odd, axis=2)
static_noise_im=sum_even-sum_odd # Subtract even and odd to obtain noise image

#################################################################################
#This section performs the Glover analysis
fig, axs=plt.subplots(1,3)
axs[0].imshow(static_noise_im, cmap=plt.cm.gray)
axs[0].set_title('static noise image')
axs[1].imshow(sum_even, cmap=plt.cm.gray)
axs[1].set_title('sum even im')
axs[2].imshow(sum_odd, cmap=plt.cm.gray)
axs[2].set_title('sum odd im')
fig.savefig(PathDicom + '/Noise&sum.png')
#plt.show()
plt.close(fig)

x_val=np.float32(range(z))
y_val=np.squeeze(arr[0,0,:])
polyval=np.zeros(np.shape(arr))
for j in range(x):
 for i in range(y):
  y_val=np.squeeze(arr[j,i,:])
  p=np.polyfit(x_val,y_val,2) #Apply a polynomial fit to each pixel series
  poly=np.poly1d(p)
  polyval[j,i,:]=poly(x_val)

residual=arr-polyval
temp_fluc_noise_im=np.std(residual, axis=2) #Temporal fluctuation noise image
SFNR_im=np.real(np.float32(signal_im)/np.float32(temp_fluc_noise_im))

fig, axs=plt.subplots(1,3)
axs[0].imshow(arr[:,:,0], cmap=plt.cm.gray)
axs[0].set_title('arr image')
axs[1].imshow(polyval[:,:,0], cmap=plt.cm.gray)
axs[1].set_title('polyval im')
axs[2].imshow(SFNR_im, cmap=plt.cm.gray, vmin=0, vmax=400)
axs[2].set_title('SFNR im')
fig.savefig(PathDicom + '/PolyfitImage.png')
#plt.show()
plt.close(fig)

x_centre=np.uint16(np.round(x/2))
y_centre=np.uint16(np.round(y/2))

SFNR_ROI=SFNR_im[x_centre-10:x_centre+10, y_centre-10:y_centre+10]
SFNR_val=np.mean(SFNR_ROI.flatten())
print(style.YELLOW +'The Signal-to-Fluctuation-Noise Ratio (SFNR) summary value is:'+ style.GREEN, SFNR_val)

varROI=static_noise_im[x_centre-10:x_centre+10, y_centre-10:y_centre+10]
var_val=np.var(varROI.flatten())

signalROI=signal_im[x_centre-10:x_centre+10, y_centre-10:y_centre+10]
mean_val=np.mean(signalROI.flatten())
SNR=mean_val/np.sqrt(var_val/z)
print(style.YELLOW +'SNR ='+ style.GREEN, SNR)

ave_series=np.zeros(z)
for j in range(z):
 ave_series[j]=np.mean(arr[x_centre-10:x_centre+10, y_centre-10:y_centre+10,j].flatten())

poly_roi=np.poly1d(np.polyfit(x_val,ave_series,2))
poly_vec=poly_roi(x_val)
roi_residuals=poly_vec-ave_series
mean_signal_intensity=np.mean(ave_series)
SD_residuals=np.std(roi_residuals)
Percent_fluc=100*(SD_residuals/mean_signal_intensity)
Drift=(np.max(poly_vec)-np.min(poly_vec))/mean_signal_intensity*100

fft_residuals=fftshift(fft(roi_residuals))
vec=fftfreq(np.uint16(z))
print(style.YELLOW + "Percent Drift = " + style.GREEN, Drift)
print(style.YELLOW + "Percent fluctuation =" + style.GREEN, Percent_fluc)
print(style.RESET)

fig, ax=plt.subplots(2,1)
ax[0].plot(x_val, ave_series, x_val, poly_roi(x_val), label="Ave signal time series")
ax[0].set(xlabel="Volume #", ylabel="Ave intensity", title="ROI Signal Drift")
ax[1].plot(vec[0:np.uint16(z/2)],np.abs(fft_residuals[np.uint16(z/2):]),label="Noise magnitude spectrum")
ax[1].set(xlabel="freq", ylabel="magnitude", title="Noise magnitude spectrum")
ax[1].grid()
fig.savefig(PathDicom + '/Curve_fit&fft.png')
#plt.show()
plt.close(fig)

################################################################################
#This section performs the Weisskoff analysis
print("Click to select the ROI location")
plt.imshow(arr[:,:,0], cmap=plt.cm.bone)
x=np.uint16(plt.ginput(1))
c=x[0,0] 
r=x[0,1]
plt.close()

plt.imshow(arr[:,:,0], cmap=plt.cm.bone)
plt.plot(x[0][0], x[0][1], 'rs')
plt.show()

ROI_ini=np.zeros((21,21,s[2]))
mean_ini=np.zeros((s[2]))
std_dev_ini=np.zeros((s[2]))

#arr0=arr.copy()
w=10 #For ROI size,
#arr0[r-w:r+w+1,c-w:c+w+1,0]=1000 This is just to check the ROI location
#plt.imshow(arr0[:,:,0])
#plt.show()

for j in range(0,s[2]):
 ROI_ini[:,:,j]=arr[r-w:r+w+1,c-w:c+w+1,j]
 noise_vec=static_noise_im[r-w:r+w+1,c-w:c+w+1].flatten()
 mean_ini[j]=np.mean(ROI_ini[:,:,j].flatten())
 std_dev_ini[j]=np.std(noise_vec)

SNR0=np.sum(mean_ini)/(np.sum(std_dev_ini)/np.sqrt(z))
print("SNR0="+style.GREEN,SNR0)

ROI_1pix=arr[r:r+1,c:c+1,:]
ROI_2pix=arr[r-1:r+1,c-1:c+1,:]
ROI_3pix=arr[r-1:r+2,c-1:c+2,:]
ROI_4pix=arr[r-2:r+2,c-2:c+2,:]
ROI_5pix=arr[r-2:r+3,c-2:c+3,:]
ROI_6pix=arr[r-3:r+3,c-3:c+3,:]
ROI_7pix=arr[r-3:r+4,c-3:c+4,:]
ROI_8pix=arr[r-4:r+4,c-4:c+4,:]
ROI_9pix=arr[r-4:r+5,c-4:c+5,:]
ROI_10pix=arr[r-5:r+5,c-5:c+5,:]
ROI_11pix=arr[r-5:r+6,c-5:c+6,:]
ROI_12pix=arr[r-6:r+6,c-6:c+6,:]
ROI_13pix=arr[r-6:r+7,c-6:c+7,:]
ROI_14pix=arr[r-7:r+7,c-7:c+7,:]
ROI_15pix=arr[r-7:r+8,c-7:c+8,:]
ROI_16pix=arr[r-8:r+8,c-8:c+8,:]
ROI_17pix=arr[r-8:r+9,c-8:c+9,:]
ROI_18pix=arr[r-9:r+9,c-9:c+9,:]
ROI_19pix=arr[r-9:r+10,c-9:c+10,:]
ROI_20pix=arr[r-10:r+10,c-10:c+10,:]
#ROI_21pix=arr[r-10:r+11,c-10:c+11,:]

ROI2=np.zeros((s[2],1))
ROI3=np.zeros((s[2],1))
ROI4=np.zeros((s[2],1))
ROI5=np.zeros((s[2],1))
ROI6=np.zeros((s[2],1))
ROI7=np.zeros((s[2],1))
ROI8=np.zeros((s[2],1))
ROI9=np.zeros((s[2],1))
ROI10=np.zeros((s[2],1))
ROI11=np.zeros((s[2],1))
ROI12=np.zeros((s[2],1))
ROI13=np.zeros((s[2],1))
ROI14=np.zeros((s[2],1))
ROI15=np.zeros((s[2],1))
ROI16=np.zeros((s[2],1))
ROI17=np.zeros((s[2],1))
ROI18=np.zeros((s[2],1))
ROI19=np.zeros((s[2],1))
ROI20=np.zeros((s[2],1))
#ROI21=np.zeros((s[2],1))

for j in range(0,s[2]):
 ROI2[j]=np.mean(ROI_2pix[:,:,j].flatten())
 ROI3[j]=np.mean(ROI_3pix[:,:,j].flatten())
 ROI4[j]=np.mean(ROI_4pix[:,:,j].flatten())
 ROI5[j]=np.mean(ROI_5pix[:,:,j].flatten())
 ROI6[j]=np.mean(ROI_6pix[:,:,j].flatten())
 ROI7[j]=np.mean(ROI_7pix[:,:,j].flatten())
 ROI8[j]=np.mean(ROI_8pix[:,:,j].flatten())
 ROI9[j]=np.mean(ROI_9pix[:,:,j].flatten())
 ROI10[j]=np.mean(ROI_10pix[:,:,j].flatten())
 ROI11[j]=np.mean(ROI_11pix[:,:,j].flatten())
 ROI12[j]=np.mean(ROI_12pix[:,:,j].flatten())
 ROI13[j]=np.mean(ROI_13pix[:,:,j].flatten())
 ROI14[j]=np.mean(ROI_14pix[:,:,j].flatten())
 ROI15[j]=np.mean(ROI_15pix[:,:,j].flatten())
 ROI16[j]=np.mean(ROI_16pix[:,:,j].flatten())
 ROI17[j]=np.mean(ROI_17pix[:,:,j].flatten())
 ROI18[j]=np.mean(ROI_18pix[:,:,j].flatten())
 ROI19[j]=np.mean(ROI_19pix[:,:,j].flatten())
 ROI20[j]=np.mean(ROI_20pix[:,:,j].flatten())
 #ROI21[j]=np.mean(ROI_21pix[:,:,j].flatten())

F=np.zeros((20,1))
F[0]=np.std(ROI_1pix)/np.mean(ROI_1pix)
F[1]=np.std(ROI2)/np.mean(ROI2)
F[2]=np.std(ROI3)/np.mean(ROI3)
F[3]=np.std(ROI4)/np.mean(ROI4)
F[4]=np.std(ROI5)/np.mean(ROI5)
F[5]=np.std(ROI6)/np.mean(ROI6)
F[6]=np.std(ROI7)/np.mean(ROI7)
F[7]=np.std(ROI8)/np.mean(ROI8)
F[8]=np.std(ROI9)/np.mean(ROI9)
F[9]=np.std(ROI10)/np.mean(ROI10)
F[10]=np.std(ROI11)/np.mean(ROI11)
F[11]=np.std(ROI12)/np.mean(ROI12)
F[12]=np.std(ROI13)/np.mean(ROI13)
F[13]=np.std(ROI14)/np.mean(ROI14)
F[14]=np.std(ROI15)/np.mean(ROI15)
F[15]=np.std(ROI16)/np.mean(ROI16)
F[16]=np.std(ROI17)/np.mean(ROI17)
F[17]=np.std(ROI18)/np.mean(ROI18)
F[18]=np.std(ROI19)/np.mean(ROI19)
F[19]=np.std(ROI20)/np.mean(ROI20)
#F[20]=np.std(ROI21)/np.mean(ROI21)

n=np.arange(1,21)
F_theory=1/(n*SNR0)
Coil_corr=F[0]/F_theory[0] #Get a coil correction factor
F=F/Coil_corr #Normalize by a coil correction factor to avoid bias
RDC=F[0]/F[-1]
print(style.RESET+"RDC="+style.GREEN,RDC)
print(style.RESET)

fig, ax=plt.subplots()
ax.loglog(n,100*F,label="Measured")
ax.loglog(n,100*F_theory,label="Theoretical")
ax.set(xlabel="ROI Width (pixels)", ylabel="100 * CV", title='Weisskoff Analysis')
ax.legend()
fig.savefig(PathDicom + '/Weisskoff_Plot.png')
plt.close(fig)

floats=np.array([SFNR_val, SNR, Percent_fluc, Drift, RDC, SNR0])
names=np.array(['SFNR=','SNR=', 'Percent_fluctuation=', 'Percent_drift=', 'RDC=', 'SNR_Weisskoff='])
ab=np.zeros(names.size, dtype=[('var1', 'U20'), ('var2', float)])
ab['var1'] = names
ab['var2'] = floats
np.savetxt(PathDicom + '/Output.txt', ab, fmt='%20s %10.4f')