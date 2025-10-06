import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.restoration import unwrap_phase
from mri_utils import DicomReader, get_bit_depth_max_value, save_results_to_file

# Initialize reader with configuration
config_file = 'code/qa_sequences_config.json'
reader = DicomReader(config_file)

# Load all required sequences automatically
sequences = reader.load_qa_sequences(None, 'b0_inhomogeneity')

# Check if all required sequences were loaded
required_sequences = ['te1_phase', 'te2_phase', 'te1_magnitude']
missing_sequences = [seq for seq in required_sequences if seq not in sequences]

if missing_sequences:
    print(f"{reader.style.RED}Error: Missing sequences: {', '.join(missing_sequences)}{reader.style.RESET}")
    print(f"{reader.style.RED}Available sequences: {list(sequences.keys())}{reader.style.RESET}")
    exit(1)

# Extract data from loaded sequences
TE1phase, metadata_te1, _ = sequences['te1_phase']
TE2phase, metadata_te2, _ = sequences['te2_phase'] 
TE1mag, _, _ = sequences['te1_magnitude']

print(f"{reader.style.GREEN}Successfully loaded all sequences{reader.style.RESET}")
print(f"TE1 phase shape: {TE1phase.shape}")
print(f"TE2 phase shape: {TE2phase.shape}")
print(f"TE1 magnitude shape: {TE1mag.shape}")

# Get acquisition parameters
TE1 = metadata_te1['echo_time']
TE2 = metadata_te2['echo_time']
Bits = metadata_te1['bits_stored']
ScanMode = metadata_te1['acquisition_type']

mask = TE1mag < 200  # Can adjust this threshold which will influence the peak-to-peak calculation
nBits = get_bit_depth_max_value(Bits)

if ScanMode == "2D":
    print(reader.style.YELLOW + "Scan mode was 2D, therefore 2D phase-unwrapping will be performed slice-by-slice" + reader.style.RESET)
    unwrapped_phase1 = np.zeros(np.shape(TE1phase))
    unwrapped_phase2 = np.zeros(np.shape(TE2phase))
    for i in range(TE1phase.shape[2]):
        arr_phase1 = 2 * np.pi / nBits * np.float32(TE1phase[:, :, i])
        unwrapped_phase1[:, :, i] = unwrap_phase(arr_phase1)
        arr_phase2 = 2 * np.pi / nBits * np.float32(TE2phase[:, :, i])
        unwrapped_phase2[:, :, i] = unwrap_phase(arr_phase2)
    # For 2D, use the full arrays for subsequent processing
    arr_phase1 = 2 * np.pi / nBits * np.float32(TE1phase)
    arr_phase2 = 2 * np.pi / nBits * np.float32(TE2phase)

elif ScanMode == "3D":
    print(reader.style.YELLOW + "Scan mode was 3D, therefore 3D phase-unwrapping will be performed" + reader.style.RESET)
    arr_phase1 = 2 * np.pi / nBits * np.float32(TE1phase)
    unwrapped_phase1 = unwrap_phase(arr_phase1)
    arr_phase2 = 2 * np.pi / nBits * np.float32(TE2phase)
    unwrapped_phase2 = unwrap_phase(arr_phase2)

gamma = 2.67522e8
DeltaTE = (TE2 - TE1) / 1000
f0 = metadata_te1['imaging_frequency']

unity = np.ones(np.shape(arr_phase1))
DeltaPhi = unwrapped_phase2 - unwrapped_phase1

DeltaPhi[mask] = 1
DeltaPhi_vec = DeltaPhi.flatten()
DeltaPhi_vec2 = np.delete(DeltaPhi_vec, np.where(DeltaPhi_vec == 1))
DeltaPhiMean = np.mean(DeltaPhi_vec2)
print('mean=', DeltaPhiMean)
if DeltaPhiMean > 5:  # Check that there is no 2pi offset in the map
    DeltaPhi = DeltaPhi - 2 * np.pi * unity
elif DeltaPhiMean < -5:  # Check that there is no -2pi offset in the map
    DeltaPhi = DeltaPhi + 2 * np.pi * unity

DeltaB0 = DeltaPhi / (2 * np.pi * 42.576e6 * DeltaTE) / (f0 / 42.576) * 1e6
DeltaB0[mask] = -1
DeltaB0_vec = DeltaB0.flatten()
DeltaB0_vec2 = np.delete(DeltaB0_vec, np.where(DeltaB0_vec == -1))

DeltaB0rms = np.sqrt(np.sum(DeltaB0_vec2**2) / DeltaB0_vec2.shape[0])
DeltaB0mean = np.mean(DeltaB0_vec2)
DeltaB0std = np.std(DeltaB0_vec2)
DeltaB0pk2pk = np.max([np.percentile(DeltaB0_vec2, 99.7), -np.percentile(DeltaB0_vec2, 0.3)])

print('Number of Pixels=', DeltaB0_vec2.shape[0])
print('Delta B0 RMS[ppm]=', DeltaB0rms)
print('Delta B0 mean[ppm]=', DeltaB0mean)
print('Delta B0 std[ppm]=', DeltaB0std)
print('Delta B0 peak-to-peak[ppm]=', DeltaB0pk2pk)

print(reader.style.RESET + 'Results stored in textfile of selected directory')
results = {
    'Number of Pixels=': DeltaB0_vec2.shape[0],
    'Delta B0 RMS[ppm]=': DeltaB0rms,
    'Delta B0 mean[ppm]=': DeltaB0mean,
    'Delta B0 std[ppm]=': DeltaB0std,
    'Delta B0 peak-to-peak[ppm]=': DeltaB0pk2pk
}

# Save to the base directory selected by user
base_dir = reader.select_dicom_directory("Select directory to save results")
save_results_to_file(results, base_dir, 'B0_homogeneity_results.txt')

hist_bins = np.linspace(-9e-1, 9e-1, 751)
fig, axs = plt.subplots(3, 2)
axs[0, 0].imshow(np.squeeze(TE1phase[:, :, 64]), cmap=plt.cm.bone, vmin=0, vmax=4096)
axs[0, 0].set_title('Phase image axial')
axs[0, 1].imshow(np.squeeze(TE1phase[:, 64, :]), cmap=plt.cm.bone, vmin=0, vmax=4096)
axs[0, 1].set_title('Phase image sag')

axs[1, 0].imshow(np.squeeze(unwrapped_phase1[:, :, 64]), cmap=plt.cm.bone, vmin=0, vmax=4*np.pi)
axs[1, 0].set_title('Unwrapped phase image axial')
axs[1, 1].imshow(np.squeeze(unwrapped_phase1[:, 64, :]), cmap=plt.cm.bone, vmin=0, vmax=4*np.pi)
axs[1, 1].set_title('Unwrapped phase image sag')

axs[2, 0].imshow(np.squeeze(DeltaB0[:, :, 64]), cmap=plt.cm.bone, vmin=-5e-1, vmax=5e-1)
axs[2, 0].set_title('B0 map axial')
axs[2, 1].imshow(np.squeeze(DeltaB0[:, 64, :]), cmap=plt.cm.bone, vmin=-5e-1, vmax=5e-1)
axs[2, 1].set_title('B0 map sag')
fig = plt.gcf()
fig.set_size_inches(7, 9)
plt.savefig(base_dir + '/Images.png', dpi=300)
plt.show()
plt.close()

plt.hist(DeltaB0.flatten(), hist_bins)
plt.title('B0 inhomogeneity in ppm')
plt.savefig(base_dir + '/DeltaB0hist.png')
plt.show()