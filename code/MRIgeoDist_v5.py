import itk
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk 
from tkinter import filedialog
from scipy import ndimage
import pydicom as dcm


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



print(style.YELLOW +'Select the directory containing the SE dataset to analyze'+ style.RESET)

root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

RefDs = dcm.read_file(lstFilesDCM[0])
if RefDs.StationName=='AWP183025':
    sl=20
else:
    sl=20

for f in range(len(lstFilesDCM)):
  RefDs = dcm.read_file(lstFilesDCM[f])
  filename = RefDs.SOPInstanceUID
  instNum = RefDs.InstanceNumber
  
  if int(instNum) == sl:
#     file_path = PathDicom +'/MR'+ filename + '.dcm'
      x = lstFilesDCM[f].split('\\')
      file = x[len(x)-1]  # last item
      file_path = PathDicom +'/'+ file
      print('file_path=', file_path)

itk_SEimage = itk.imread(file_path)
SEview = itk.array_view_from_image(itk_SEimage)

inputImage = file_path
outputImage = 'SE_edges.dcm'
variance = 3
lowerThreshold = 10 #10-80
upperThreshold = 40

InputPixelType = itk.F
OutputPixelType = itk.UC
Dimension = 2

InputImageType = itk.Image[InputPixelType, Dimension]
OutputImageType = itk.Image[OutputPixelType, Dimension]

reader = itk.ImageFileReader[InputImageType].New()
reader.SetFileName(inputImage)

reader2=sitk.ImageFileReader()
reader2.SetFileName(inputImage)
reader2.LoadPrivateTagsOn()
reader2.ReadImageInformation()
reader2.GetMetaDataKeys()

#for k in reader2.GetMetaDataKeys():
#   v = reader2.GetMetaData(k)
#   print(f"({k}) = = \"{v}\"")

spacing=reader2.GetMetaData('0028|0030')
x=spacing.split('\\')
SEPixelSpacingX=float(x[0])
SEPixelSpacingY=float(x[1])

#Call an itk edge-detection filter to detect the edges of the phantom
cannyFilter = itk.CannyEdgeDetectionImageFilter[
    InputImageType,
    InputImageType].New()
cannyFilter.SetInput(reader.GetOutput())
cannyFilter.SetVariance(variance)
cannyFilter.SetLowerThreshold(lowerThreshold)
cannyFilter.SetUpperThreshold(upperThreshold)

rescaler = itk.RescaleIntensityImageFilter[
    InputImageType,
    OutputImageType].New()
rescaler.SetInput(cannyFilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

writer = itk.ImageFileWriter[OutputImageType].New()
writer.SetFileName(outputImage)
writer.SetInput(rescaler.GetOutput())
writer.Update()

SEarray=np.squeeze(itk.array_from_image(itk_SEimage))
#print(np.shape(SEarray))
centroidSE=ndimage.measurements.center_of_mass(SEarray)

#Repeat the same procedure for the EPI image and compare to the SE image
print(style.YELLOW + 'Select the directory containing the EPI dataset to analyze' + style.RESET)


root = tk.Tk() 
root.withdraw() 
PathDicom = filedialog.askdirectory()
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

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
      print('file_path=', file_path)


itk_EPIimage = itk.imread(file_path)
# Run filters on itk.Image

# View only of itk.Image, data is not copied
EPIview = itk.array_view_from_image(itk_EPIimage)

inputImage = file_path
outputImage = 'EPI_edges.dcm'
variance = 3
lowerThreshold = 10
upperThreshold = 40

InputPixelType = itk.F
OutputPixelType = itk.UC
Dimension = 2

InputImageType = itk.Image[InputPixelType, Dimension]
OutputImageType = itk.Image[OutputPixelType, Dimension]

reader = itk.ImageFileReader[InputImageType].New()
reader.SetFileName(inputImage)

reader2=sitk.ImageFileReader()
reader2.SetFileName(inputImage)
reader2.LoadPrivateTagsOn()
reader2.ReadImageInformation()
reader2.GetMetaDataKeys()
spacing=reader2.GetMetaData('0028|0030')
x=spacing.split('\\')
EPIPixelSpacingX=float(x[0])
EPIPixelSpacingY=float(x[1])

#Call an itk edge-detection filter to detect the edges of the phantom
cannyFilter = itk.CannyEdgeDetectionImageFilter[
    InputImageType,
    InputImageType].New()
cannyFilter.SetInput(reader.GetOutput())
cannyFilter.SetVariance(variance)
cannyFilter.SetLowerThreshold(lowerThreshold)
cannyFilter.SetUpperThreshold(upperThreshold)

rescaler = itk.RescaleIntensityImageFilter[
    InputImageType,
    OutputImageType].New()
rescaler.SetInput(cannyFilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

writer = itk.ImageFileWriter[OutputImageType].New()
writer.SetFileName(outputImage)
writer.SetInput(rescaler.GetOutput())
writer.Update()

EPIarray=np.squeeze(itk.array_from_image(itk_EPIimage))
#plt.imshow(np.squeeze(EPIarray), cmap=plt.cm.bone)
#plt.show()
#print(np.shape(SEarray))

centroidEPI=ndimage.measurements.center_of_mass(EPIarray)
#print(centroidEPI)


itk_SEimageEdges = itk.imread('SE_edges.dcm')
itk_EPIimageEdges = itk.imread('EPI_edges.dcm')

#fig, axs=plt.subplots(1,2)
#axs[0].imshow(np.squeeze(itk_SEimageEdges))
#axs[0].set_title('SE edges')
#axs[1].imshow(np.squeeze(itk_EPIimageEdges))
#axs[1].set_title('EPI edges')
#plt.show()



SEviewEdges = itk.array_view_from_image(itk_SEimageEdges)
SEarrayEdges=itk.array_from_image(itk_SEimageEdges)

EPIviewEdges = itk.array_view_from_image(itk_EPIimageEdges)
EPIarrayEdges=itk.array_from_image(itk_EPIimageEdges)


i=int(round(centroidSE[0]))
j=int(round(centroidSE[1]))

index1=np.nonzero(SEarrayEdges[:,i,:])
ind1=list(index1[1])
index2=np.nonzero(SEarrayEdges[:,:,j])
ind2=list(index2[1])
#print(ind1)
#print(index1)

print(style.WHITE + 'Image centroid =', centroidSE)
SEdistX=(ind1[-1]-ind1[0])*SEPixelSpacingX
SEdistY=(ind2[-1]-ind2[0])*SEPixelSpacingY
print(style.WHITE + 'SE image FWHM along X= ', SEdistX, 'mm')
print(style.WHITE + 'SE image FWHM along Y= ', SEdistY, 'mm')

i=int(round(centroidEPI[0]))
j=int(round(centroidEPI[1]))

index1=np.nonzero(EPIarrayEdges[:,i,:])
ind1=list(index1[1])
index2=np.nonzero(EPIarrayEdges[:,:,j])
ind2=list(index2[1])


print('Image centroid =', centroidEPI)
EPIdistX=(ind1[-1]-ind1[0])*EPIPixelSpacingX
EPIdistY=(ind2[-1]-ind2[0])*EPIPixelSpacingY
print(style.WHITE + 'EPI image FWHM along X= ', EPIdistX, 'mm')
print(style.WHITE + 'EPI image FWHM along Y= ', EPIdistY, 'mm')

XDiff=EPIdistX-SEdistX
YDiff=EPIdistY-SEdistY
dist=np.sqrt((centroidSE[0]-centroidEPI[0])**2+(centroidSE[1]-centroidEPI[1])**2)

tol=2.00 #Choice of tolerance for EPI distortions

if np.abs(XDiff) <= tol:
	print(style.GREEN + 'Difference in FWHM along x =', XDiff, 'mm' + style.RESET)
else:
	print(style.RED + 'Difference in FWHM along x =', XDiff, 'mm' + style.RESET)
if np.abs(YDiff) <= tol:	
	print(style.GREEN + 'Difference in FWHM along y =', YDiff, 'mm' + style.RESET)
else:
	print(style.RED + 'Difference in FWHM along y =', YDiff, 'mm' + style.RESET)

if np.abs(dist) <= tol/2:
	print(style.GREEN + 'Distance in centroids =', dist, 'mm' + style.RESET)
else:
	print(style.RED + 'Distance in centroids =', dist, 'mm' + style.RESET)

fig, axs=plt.subplots(2,2)
axs[0,0].imshow(np.squeeze(SEview), cmap=plt.cm.gray)
axs[0,0].set_title('2D SE image')
axs[1,0].imshow(np.squeeze(SEviewEdges), cmap=plt.cm.gray)
axs[1,0].set_title('2D SE image edges')
axs[0,1].imshow(np.squeeze(EPIview), cmap=plt.cm.gray)
axs[0,1].set_title('2D EPI image')
axs[0,1].set_xlabel('Delta centroids = %s mm'% str(dist))
axs[1,1].imshow(np.squeeze(EPIviewEdges), cmap=plt.cm.gray)
axs[1,1].set_title('2D EPI image edges')
fig=plt.gcf()
fig.set_size_inches(6,8)

print(style.YELLOW + 'Select a directory to save the output'+ style.RESET)
path=filedialog.askdirectory()
plt.savefig(path + '/Images&Edges.png', dpi=300)
plt.show()

floats=np.array([SEdistX, SEdistY, EPIdistX, EPIdistY, XDiff, YDiff, dist])
names=np.array(['SE freq FWHM=', 'SE phase FWHM=', 'EPI freq FWHM=', 'EPI phase FWHM=', 'Geometric dist freq=', 'Geometric dist phase=', 'Diff in centroids='])
ab=np.zeros(names.size, dtype=[('var1', 'U22'), ('var2', float)])
ab['var1'] = names
ab['var2'] = floats
np.savetxt(path + '/EPIgeoDist.txt', ab, fmt='%22s %10.5f')