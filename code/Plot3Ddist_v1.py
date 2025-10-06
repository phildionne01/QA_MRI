import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
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


SixGradientDirections = 0 #Change this switch to run a single plot instead of all 6 directions
#For the case of 6 different gradient directions, the files must have the following nomenclature:
#mass_centers_deform_world_ax_LR.txt, mass_centers_trans_world_ax_LR.txt, etc...

#For a single gradient direction with or without correction, the files must have the names:
#mass_centers_deform_world.txt, mass_centers_trans_world.txt, mass_centers_deform_world_ND.txt
#mass_centers_trans_world_ND.txt,

print(style.YELLOW + 'Select the directory containing the two txt files with the distortion coordinates' + style.RESET)

root = tk.Tk() 
root.withdraw() 
Path = filedialog.askdirectory()


if SixGradientDirections==1:

  coordinates_image_ax_LR = np.loadtxt(Path + '/mass_centers_deform_world_ax_LR.txt')
  coordinates_model_ax_LR = np.loadtxt(Path + '/mass_centers_trans_world_ax_LR.txt')

  coordinates_image_ax_RL = np.loadtxt(Path + '/mass_centers_deform_world_ax_RL.txt')
  coordinates_model_ax_RL = np.loadtxt(Path + '/mass_centers_trans_world_ax_RL.txt')

  coordinates_image_cor_FH = np.loadtxt(Path + '/mass_centers_deform_world_cor_FH.txt')
  coordinates_model_cor_FH = np.loadtxt(Path + '/mass_centers_trans_world_cor_FH.txt')

  coordinates_image_cor_HF = np.loadtxt(Path + '/mass_centers_deform_world_cor_HF.txt')
  coordinates_model_cor_HF = np.loadtxt(Path + '/mass_centers_trans_world_cor_HF.txt')

  coordinates_image_sag_AP = np.loadtxt(Path + '/mass_centers_deform_world_sag_AP.txt')
  coordinates_model_sag_AP = np.loadtxt(Path + '/mass_centers_trans_world_sag_AP.txt')

  coordinates_image_sag_PA = np.loadtxt(Path + '/mass_centers_deform_world_sag_PA.txt')
  coordinates_model_sag_PA = np.loadtxt(Path + '/mass_centers_trans_world_sag_PA.txt')




  coordinates_image_dict_ax_LR = { x[3] : x[:3] for x in coordinates_image_ax_LR }
  coordinates_model_dict_ax_LR = { x[3] : x[:3] for x in coordinates_model_ax_LR }

  coordinates_image_dict_ax_RL = { x[3] : x[:3] for x in coordinates_image_ax_RL }
  coordinates_model_dict_ax_RL = { x[3] : x[:3] for x in coordinates_model_ax_RL }

  coordinates_image_dict_cor_FH = { x[3] : x[:3] for x in coordinates_image_cor_FH }
  coordinates_model_dict_cor_FH = { x[3] : x[:3] for x in coordinates_model_cor_FH }

  coordinates_image_dict_cor_HF = { x[3] : x[:3] for x in coordinates_image_cor_HF }
  coordinates_model_dict_cor_HF = { x[3] : x[:3] for x in coordinates_model_cor_HF }

  coordinates_image_dict_sag_AP = { x[3] : x[:3] for x in coordinates_image_sag_AP }
  coordinates_model_dict_sag_AP = { x[3] : x[:3] for x in coordinates_model_sag_AP }
 
  coordinates_image_dict_sag_PA = { x[3] : x[:3] for x in coordinates_image_sag_PA }
  coordinates_model_dict_sag_PA = { x[3] : x[:3] for x in coordinates_model_sag_PA }



  x_axis_ax_LR=list()
  y_axis_ax_LR=list()
  for label in coordinates_image_dict_ax_LR:
	  image_coord_ax_LR = coordinates_image_dict_ax_LR[label]
	  model_coord_ax_LR = coordinates_model_dict_ax_LR[label]
	  point_distance_ax_LR = np.linalg.norm(image_coord_ax_LR-model_coord_ax_LR)
	  isocenter_distance_ax_LR = np.linalg.norm(model_coord_ax_LR)
	  x_axis_ax_LR += [isocenter_distance_ax_LR]
	  y_axis_ax_LR += [point_distance_ax_LR]

  x_axis_ax_RL=list()
  y_axis_ax_RL=list()
  for label in coordinates_image_dict_ax_RL:
	  image_coord_ax_RL = coordinates_image_dict_ax_RL[label]
	  model_coord_ax_RL = coordinates_model_dict_ax_RL[label]
	  point_distance_ax_RL = np.linalg.norm(image_coord_ax_RL-model_coord_ax_RL)
	  isocenter_distance_ax_RL = np.linalg.norm(model_coord_ax_RL)
	  x_axis_ax_RL += [isocenter_distance_ax_RL]
	  y_axis_ax_RL += [point_distance_ax_RL]

  x_axis_cor_FH=list()
  y_axis_cor_FH=list()
  for label in coordinates_image_dict_cor_FH:
	  image_coord_cor_FH = coordinates_image_dict_cor_FH[label]
	  model_coord_cor_FH = coordinates_model_dict_cor_FH[label]
	  point_distance_cor_FH = np.linalg.norm(image_coord_cor_FH-model_coord_cor_FH)
	  isocenter_distance_cor_FH = np.linalg.norm(model_coord_cor_FH)
	  x_axis_cor_FH += [isocenter_distance_cor_FH]
	  y_axis_cor_FH += [point_distance_cor_FH]

  x_axis_cor_HF=list()
  y_axis_cor_HF=list()
  for label in coordinates_image_dict_cor_HF:
	  image_coord_cor_HF = coordinates_image_dict_cor_HF[label]
	  model_coord_cor_HF = coordinates_model_dict_cor_HF[label]
	  point_distance_cor_HF = np.linalg.norm(image_coord_cor_HF-model_coord_cor_HF)
	  isocenter_distance_cor_HF = np.linalg.norm(model_coord_cor_HF)
	  x_axis_cor_HF += [isocenter_distance_cor_HF]
	  y_axis_cor_HF += [point_distance_cor_HF]


  x_axis_sag_AP=list()
  y_axis_sag_AP=list()
  for label in coordinates_image_dict_sag_AP:
	  image_coord_sag_AP = coordinates_image_dict_sag_AP[label]
	  model_coord_sag_AP = coordinates_model_dict_sag_AP[label]
	  point_distance_sag_AP = np.linalg.norm(image_coord_sag_AP-model_coord_sag_AP)
	  isocenter_distance_sag_AP = np.linalg.norm(model_coord_sag_AP)
	  x_axis_sag_AP += [isocenter_distance_sag_AP]
	  y_axis_sag_AP += [point_distance_sag_AP]

  x_axis_sag_PA=list()
  y_axis_sag_PA=list()
  for label in coordinates_image_dict_sag_PA:
	  image_coord_sag_PA = coordinates_image_dict_sag_PA[label]
	  model_coord_sag_PA = coordinates_model_dict_sag_PA[label]
	  point_distance_sag_PA = np.linalg.norm(image_coord_sag_PA-model_coord_sag_PA)
	  isocenter_distance_sag_PA = np.linalg.norm(model_coord_sag_PA)
	  x_axis_sag_PA += [isocenter_distance_sag_PA]
	  y_axis_sag_PA += [point_distance_sag_PA]


  fig, axs=plt.subplots(3,2)
  fig=plt.gcf()

  plt.style.use('seaborn-whitegrid')
  axs[0,0].scatter(x_axis_ax_LR, y_axis_ax_LR, s=0.8)
  axs[0,0].set_title('gre_ax_LR')
  axs[0,0].set_xlabel('Distance from iso [mm]')
  axs[0,0].set_ylabel('Distortion mm')
  axs[0,0].set_xlim(right=300)
  axs[0,0].set_ylim(top=15)
  axs[0,0].grid(True)

  axs[0,1].scatter(x_axis_ax_RL, y_axis_ax_RL, s=0.8)
  axs[0,1].set_title('gre_ax_RL')
  axs[0,1].set_xlabel('Distance from iso [mm]')
  axs[0,1].set_ylabel('Distortion mm')
  axs[0,1].set_xlim(right=300)
  axs[0,1].set_ylim(top=15)
  axs[0,1].grid(True)

  axs[1,0].scatter(x_axis_cor_HF, y_axis_cor_HF, s=0.8)
  axs[1,0].set_title('gre_cor_HF')
  axs[1,0].set_ylabel('Distortion mm')
  axs[1,0].set_xlabel('Distance from iso [mm]')
  axs[1,0].set_xlim(right=300)
  axs[1,0].set_ylim(top=15)
  axs[1,0].grid(True)

  axs[1,1].scatter(x_axis_cor_FH, y_axis_cor_FH, s=0.8)
  axs[1,1].set_title('gre_cor_FH')
  axs[1,1].set_ylabel('Distortion mm')
  axs[1,1].set_xlabel('Distance from iso [mm]')
  axs[1,1].set_xlim(right=300)
  axs[1,1].set_ylim(top=15)
  axs[1,1].grid(True)

  axs[2,0].scatter(x_axis_sag_AP, y_axis_sag_AP, s=0.8)
  axs[2,0].set_title('gre_sag_AP')
  axs[2,0].set_ylabel('Distortion mm')
  axs[2,0].set_xlabel('Distance from iso [mm]')
  axs[2,0].set_xlim(right=300)
  axs[2,0].set_ylim(top=15)
  axs[2,0].grid(True)

  axs[2,1].scatter(x_axis_sag_PA, y_axis_sag_PA, s=0.8)
  axs[2,1].set_title('gre_sag_PA')
  axs[2,1].set_ylabel('Distortion mm')
  axs[2,1].set_xlabel('Distance from iso [mm]')
  axs[2,1].set_xlim(right=300)
  axs[2,1].set_ylim(top=15)
  axs[2,1].grid(True)

  fig.set_size_inches(9,12.5)
  #plt.savefig(Path + '/DistortionScatterPlots_6directions.png')
  plt.savefig(Path + '/DistortionScatterPlots_6directions.tiff', dpi=600, format='tiff')
  plt.savefig(Path + '/DistortionScatterPlots_6directions.svg', format='svg')  
  plt.show()

  #Obtain statistics and print in txt file for ax_LR direction
  x=np.array(x_axis_ax_LR)
  y=np.array(y_axis_ax_LR)
  dist10cm=y[x<100]
  dist15cm=y[np.where(np.logical_and(x>=100, x<150))]
  dist20cm=y[np.where(np.logical_and(x>=150, x<200))]
  dist25cm=y[np.where(np.logical_and(x>=200, x<250))]
  dist30cm=y[x>=250]

  mean10=np.mean(dist10cm) 
  mean15=np.mean(dist15cm) 
  mean20=np.mean(dist20cm)
  mean25=np.mean(dist25cm) 
  mean30=np.mean(dist30cm)
  max10=np.max(dist10cm) 
  max15=np.max(dist15cm) 
  max20=np.max(dist20cm) 
  max25=np.max(dist25cm) 
  max30=np.max(dist30cm)
  std10=np.std(dist10cm) 
  std15=np.std(dist15cm) 
  std20=np.std(dist20cm) 
  std25=np.std(dist25cm) 
  std30=np.std(dist30cm)  

  floats=np.array([mean10, std10, max10, mean15, std15, max15, mean20, std20, max20, mean25, std25, max25, mean30, std30, max30])
  names=np.array(['mean<100mm =', 'StdDev=', 'max[mm]=','mean 100-150mm=', 'StdDev=', 'max[mm]=', 'mean 150-200mm=', 'StdDev=', 'max[mm]=', 'mean 200-250mm=', 'StdDev=', 'max[mm]=', 'mean>250mm', 'StdDev=', 'max[mm]=' ])
  ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
  ab['var1'] = names
  ab['var2'] = floats
  np.savetxt(Path + '/DistortionStats_ax_LR.txt', ab, fmt='%20s %10.5f')

  #Obtain statistics and print in txt file for ax_RL direction
  x=np.array(x_axis_ax_RL)
  y=np.array(y_axis_ax_RL)
  dist10cm=y[x<100]
  dist15cm=y[np.where(np.logical_and(x>=100, x<150))]
  dist20cm=y[np.where(np.logical_and(x>=150, x<200))]
  dist25cm=y[np.where(np.logical_and(x>=200, x<250))]
  dist30cm=y[x>=250]

  mean10=np.mean(dist10cm) 
  mean15=np.mean(dist15cm) 
  mean20=np.mean(dist20cm)
  mean25=np.mean(dist25cm) 
  mean30=np.mean(dist30cm)
  max10=np.max(dist10cm) 
  max15=np.max(dist15cm) 
  max20=np.max(dist20cm) 
  max25=np.max(dist25cm) 
  max30=np.max(dist30cm)
  std10=np.std(dist10cm) 
  std15=np.std(dist15cm) 
  std20=np.std(dist20cm) 
  std25=np.std(dist25cm) 
  std30=np.std(dist30cm)

  floats=np.array([mean10, std10, max10, mean15, std15, max15, mean20, std20, max20, mean25, std25, max25, mean30, std30, max30])
  names=np.array(['mean<100mm =', 'StdDev=', 'max[mm]=','mean 100-150mm=', 'StdDev=', 'max[mm]=', 'mean 150-200mm=', 'StdDev=', 'max[mm]=', 'mean 200-250mm=', 'StdDev=', 'max[mm]=', 'mean>250mm', 'StdDev=', 'max[mm]=' ])
  ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
  ab['var1'] = names
  ab['var2'] = floats
  np.savetxt(Path + '/DistortionStats_ax_RL.txt', ab, fmt='%20s %10.5f')

  #Obtain statistics and print in txt file for cor_FH direction
  x=np.array(x_axis_cor_FH)
  y=np.array(y_axis_cor_FH)
  dist10cm=y[x<100]
  dist15cm=y[np.where(np.logical_and(x>=100, x<150))]
  dist20cm=y[np.where(np.logical_and(x>=150, x<200))]
  dist25cm=y[np.where(np.logical_and(x>=200, x<250))]
  dist30cm=y[x>=250]

  mean10=np.mean(dist10cm) 
  mean15=np.mean(dist15cm) 
  mean20=np.mean(dist20cm)
  mean25=np.mean(dist25cm) 
  mean30=np.mean(dist30cm)
  max10=np.max(dist10cm) 
  max15=np.max(dist15cm) 
  max20=np.max(dist20cm) 
  max25=np.max(dist25cm) 
  max30=np.max(dist30cm)
  std10=np.std(dist10cm) 
  std15=np.std(dist15cm) 
  std20=np.std(dist20cm) 
  std25=np.std(dist25cm) 
  std30=np.std(dist30cm)

  floats=np.array([mean10, std10, max10, mean15, std15, max15, mean20, std20, max20, mean25, std25, max25, mean30, std30, max30])
  names=np.array(['mean<100mm =', 'StdDev=', 'max[mm]=','mean 100-150mm=', 'StdDev=', 'max[mm]=', 'mean 150-200mm=', 'StdDev=', 'max[mm]=', 'mean 200-250mm=', 'StdDev=', 'max[mm]=', 'mean>250mm', 'StdDev=', 'max[mm]=' ])
  ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
  ab['var1'] = names
  ab['var2'] = floats
  np.savetxt(Path + '/DistortionStats_cor_FH.txt', ab, fmt='%20s %10.5f')

  #Obtain statistics and print in txt file for cor_HF direction
  x=np.array(x_axis_cor_HF)
  y=np.array(y_axis_cor_HF)
  dist10cm=y[x<100]
  dist15cm=y[np.where(np.logical_and(x>=100, x<150))]
  dist20cm=y[np.where(np.logical_and(x>=150, x<200))]
  dist25cm=y[np.where(np.logical_and(x>=200, x<250))]
  dist30cm=y[x>=250]

  mean10=np.mean(dist10cm) 
  mean15=np.mean(dist15cm) 
  mean20=np.mean(dist20cm)
  mean25=np.mean(dist25cm) 
  mean30=np.mean(dist30cm)
  max10=np.max(dist10cm) 
  max15=np.max(dist15cm) 
  max20=np.max(dist20cm) 
  max25=np.max(dist25cm) 
  max30=np.max(dist30cm)
  std10=np.std(dist10cm) 
  std15=np.std(dist15cm) 
  std20=np.std(dist20cm) 
  std25=np.std(dist25cm) 
  std30=np.std(dist30cm)

  floats=np.array([mean10, std10, max10, mean15, std15, max15, mean20, std20, max20, mean25, std25, max25, mean30, std30, max30])
  names=np.array(['mean<100mm =', 'StdDev=','max[mm]=','mean 100-150mm=', 'StdDev=', 'max[mm]=', 'mean 150-200mm=', 'StdDev=', 'max[mm]=', 'mean 200-250mm=', 'StdDev=', 'max[mm]=', 'mean>250mm', 'StdDev=', 'max[mm]=' ])
  ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
  ab['var1'] = names
  ab['var2'] = floats
  np.savetxt(Path + '/DistortionStats_cor_HF.txt', ab, fmt='%20s %10.5f')

  #Obtain statistics and print in txt file for sag_AP direction
  x=np.array(x_axis_sag_AP)
  y=np.array(y_axis_sag_AP)
  dist10cm=y[x<100]
  dist15cm=y[np.where(np.logical_and(x>=100, x<150))]
  dist20cm=y[np.where(np.logical_and(x>=150, x<200))]
  dist25cm=y[np.where(np.logical_and(x>=200, x<250))]
  dist30cm=y[x>=250]

  mean10=np.mean(dist10cm) 
  mean15=np.mean(dist15cm) 
  mean20=np.mean(dist20cm)
  mean25=np.mean(dist25cm) 
  mean30=np.mean(dist30cm)
  max10=np.max(dist10cm) 
  max15=np.max(dist15cm) 
  max20=np.max(dist20cm) 
  max25=np.max(dist25cm) 
  max30=np.max(dist30cm)
  std10=np.std(dist10cm) 
  std15=np.std(dist15cm) 
  std20=np.std(dist20cm) 
  std25=np.std(dist25cm) 
  std30=np.std(dist30cm)

  floats=np.array([mean10, std10, max10, mean15, std15, max15, mean20, std20, max20, mean25, std25, max25, mean30, std30, max30])
  names=np.array(['mean<100mm =', 'StdDev=','max[mm]=','mean 100-150mm=', 'StdDev=','max[mm]=', 'mean 150-200mm=', 'StdDev=','max[mm]=', 'mean 200-250mm=', 'StdDev=','max[mm]=', 'mean>250mm', 'StdDev=','max[mm]=' ])
  ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
  ab['var1'] = names
  ab['var2'] = floats
  np.savetxt(Path + '/DistortionStats_sag_AP.txt', ab, fmt='%20s %10.5f')

  #Obtain statistics and print in txt file for sag_PA direction
  x=np.array(x_axis_sag_PA)
  y=np.array(y_axis_sag_PA)
  dist10cm=y[x<100]
  dist15cm=y[np.where(np.logical_and(x>=100, x<150))]
  dist20cm=y[np.where(np.logical_and(x>=150, x<200))]
  dist25cm=y[np.where(np.logical_and(x>=200, x<250))]
  dist30cm=y[x>=250]

  mean10=np.mean(dist10cm) 
  mean15=np.mean(dist15cm) 
  mean20=np.mean(dist20cm)
  mean25=np.mean(dist25cm) 
  mean30=np.mean(dist30cm)
  max10=np.max(dist10cm) 
  max15=np.max(dist15cm) 
  max20=np.max(dist20cm) 
  max25=np.max(dist25cm) 
  max30=np.max(dist30cm)
  std10=np.std(dist10cm) 
  std15=np.std(dist15cm) 
  std20=np.std(dist20cm) 
  std25=np.std(dist25cm) 
  std30=np.std(dist30cm)

  floats=np.array([mean10, std10, max10, mean15, std15, max15, mean20, std20, max20, mean25, std25, max25, mean30, std30, max30])
  names=np.array(['mean<100mm =', 'StdDev=','max[mm]=','mean 100-150mm=', 'StdDev=','max[mm]=', 'mean 150-200mm=', 'StdDev=','max[mm]=', 'mean 200-250mm=', 'StdDev=','max[mm]=', 'mean>250mm', 'StdDev=','max[mm]=' ])
  ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
  ab['var1'] = names
  ab['var2'] = floats
  np.savetxt(Path + '/DistortionStats_sag_PA.txt', ab, fmt='%20s %10.5f')

else:

  coordinates_image = np.loadtxt(Path + '/mass_centers_deform_world.txt')
  coordinates_model = np.loadtxt(Path + '/mass_centers_trans_world.txt')

  coordinates_image_ND = np.loadtxt(Path + '/mass_centers_deform_world_ND.txt')
  coordinates_model_ND = np.loadtxt(Path + '/mass_centers_trans_world_ND.txt')

  coordinates_image_dict = { x[3] : x[:3] for x in coordinates_image }
  coordinates_model_dict = { x[3] : x[:3] for x in coordinates_model }

  coordinates_image_dict_ND = { x[3] : x[:3] for x in coordinates_image_ND }
  coordinates_model_dict_ND = { x[3] : x[:3] for x in coordinates_model_ND }

  x_axis=list()
  y_axis=list()
  for label in coordinates_image_dict:
	  image_coord = coordinates_image_dict[label]
	  model_coord = coordinates_model_dict[label]
	  point_distance = np.linalg.norm(image_coord-model_coord)
	  isocenter_distance = np.linalg.norm(model_coord)
	  x_axis += [isocenter_distance]
	  y_axis += [point_distance]

  x_axis_ND=list()
  y_axis_ND=list()
  for label in coordinates_image_dict_ND:
	  image_coord_ND = coordinates_image_dict_ND[label]
	  model_coord_ND = coordinates_model_dict_ND[label]
	  point_distance_ND = np.linalg.norm(image_coord_ND-model_coord_ND)
	  isocenter_distance_ND = np.linalg.norm(model_coord_ND)
	  x_axis_ND += [isocenter_distance_ND]
	  y_axis_ND += [point_distance_ND]

  fig, axs=plt.subplots(2,1)
  plt.style.use('seaborn-whitegrid')
  axs[0].scatter(x_axis, y_axis, s=1.2)
  axs[0].set_title('Corrected Gradient 3D Distortion control points')
  axs[0].set_xlabel('Distance from iso [mm]')
  axs[0].set_ylabel('Distortion mm')
  axs[0].set_xlim(right=300)
  axs[0].set_ylim(top=50)
  axs[0].grid(True)

  axs[1].scatter(x_axis_ND, y_axis_ND, s=1.2)
  axs[1].set_title('Uncorrected Gradient 3D Distortion control points')
  axs[1].set_xlabel('Distance from iso [mm]')
  axs[1].set_ylabel('Distortion mm')
  axs[1].set_xlim(right=300)
  axs[1].set_ylim(top=50)
  axs[1].grid(True)
  fig=plt.gcf()	
  fig.set_size_inches(6,9)
  plt.savefig(Path + '/DistortionScatterPlots_Corr&Uncorr.png')
  plt.savefig(Path + '/DistortionScatterPlots_Corr&Uncorr.tiff', dpi=600, format='tiff')
  plt.savefig(Path + '/DistortionScatterPlots_Corr&Uncorr.svg',  format='svg')
  plt.show()

  x=np.array(x_axis)
  y=np.array(y_axis)
  dist10cm=y[x<100]
  dist15cm=y[np.where(np.logical_and(x>=100, x<150))]
  dist20cm=y[np.where(np.logical_and(x>=150, x<200))]
  dist25cm=y[np.where(np.logical_and(x>=200, x<250))]
  dist30cm=y[x>=250]

  print('mean < 100mm =', np.mean(dist10cm),', StdDev=', np.std(dist10cm),', max[mm]=', np.max(dist10cm))
  print('mean 100-150mm =', np.mean(dist15cm),', StdDev=', np.std(dist15cm),', max[mm]=', np.max(dist15cm))
  print('mean 150-200mm =', np.mean(dist20cm),', StdDev=', np.std(dist20cm),', max[mm]=', np.max(dist20cm))
  print('mean 200-250mm =', np.mean(dist25cm),', StdDev=', np.std(dist25cm),', max[mm]=', np.max(dist25cm))
  print('mean > 250mm =', np.mean(dist30cm),', StdDev=', np.std(dist30cm),', max[mm]=', np.max(dist30cm))

  mean10=np.mean(dist10cm) 
  mean15=np.mean(dist15cm) 
  mean20=np.mean(dist20cm)
  mean25=np.mean(dist25cm) 
  mean30=np.mean(dist30cm)
  max10=np.max(dist10cm) 
  max15=np.max(dist15cm) 
  max20=np.max(dist20cm) 
  max25=np.max(dist25cm) 
  max30=np.max(dist30cm)
  std10=np.std(dist10cm) 
  std15=np.std(dist15cm) 
  std20=np.std(dist20cm) 
  std25=np.std(dist25cm) 
  std30=np.std(dist30cm)

  floats=np.array([mean10, std10, max10,  mean15, std15, max15, mean20, std20, max20, mean25, std25, max25, mean30, std30, max30])
  names=np.array(['mean<100mm =', 'StdDev=', 'max[mm]=','mean 100-150mm=', 'StdDev=','max[mm]=', 'mean 150-200mm=','StdDev=', 'max[mm]=', 'mean 200-250mm=', 'StdDev=', 'max[mm]=', 'mean>250mm', 'StdDev=', 'max[mm]='])
  ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
  ab['var1'] = names
  ab['var2'] = floats
  np.savetxt(Path + '/DistortionStats_corr.txt', ab, fmt='%20s %10.5f')

  x=np.array(x_axis_ND)
  y=np.array(y_axis_ND)
  dist10cm=y[x<100]
  dist15cm=y[np.where(np.logical_and(x>=100, x<150))]
  dist20cm=y[np.where(np.logical_and(x>=150, x<200))]
  dist25cm=y[np.where(np.logical_and(x>=200, x<250))]
  dist30cm=y[x>=250]

  print('mean < 100mm =', np.mean(dist10cm), ', StdDev=', np.std(dist10cm), ', max[mm]=', np.max(dist10cm))
  print('mean 100-150mm =', np.mean(dist15cm),', StdDev=', np.std(dist15cm), ', max[mm]=', np.max(dist15cm))
  print('mean 150-200mm =', np.mean(dist20cm),', StdDev=', np.std(dist20cm), ', max[mm]=', np.max(dist20cm))
  print('mean 200-250mm =', np.mean(dist25cm),', StdDev=', np.std(dist25cm), ', max[mm]=', np.max(dist25cm))
  print('mean > 250mm =', np.mean(dist30cm),', StdDev=', np.std(dist30cm),', max[mm]=', np.max(dist30cm))

  mean10=np.mean(dist10cm) 
  mean15=np.mean(dist15cm) 
  mean20=np.mean(dist20cm)
  mean25=np.mean(dist25cm) 
  mean30=np.mean(dist30cm)
  max10=np.max(dist10cm) 
  max15=np.max(dist15cm) 
  max20=np.max(dist20cm) 
  max25=np.max(dist25cm) 
  max30=np.max(dist30cm)
  std10=np.std(dist10cm) 
  std15=np.std(dist15cm) 
  std20=np.std(dist20cm) 
  std25=np.std(dist25cm) 
  std30=np.std(dist30cm)

  floats=np.array([mean10, std10, max10, mean15, std15, max15, mean20, std20, max20, mean25, std25, max25, mean30, std30, max30])
  names=np.array(['mean<100mm =', 'StdDev=','max[mm]=','mean 100-150mm=', 'StdDev=','max[mm]=', 'mean 150-200mm=', 'StdDev=','max[mm]=', 'mean 200-250mm=', 'StdDev=','max[mm]=', 'mean>250mm', 'StdDev=','max[mm]=' ])
  ab=np.zeros(names.size, dtype=[('var1', 'U14'), ('var2', float)])
  ab['var1'] = names
  ab['var2'] = floats
  np.savetxt(Path + '/DistortionStats_uncorr.txt', ab, fmt='%20s %10.5f')