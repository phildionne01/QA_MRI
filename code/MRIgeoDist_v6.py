import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy import ndimage
import cv2
import pydicom as dcm
from mri_utils import DicomReader

# Initialize DICOM reader with configuration
config_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'qa_sequences_config.json')
dicom_reader = DicomReader(config_file=config_file)
style = dicom_reader.style

# Edge detection parameters
EDGE_LOWER_THRESHOLD = 60
EDGE_UPPER_THRESHOLD = 150
SLICE_NUMBER = 20  # Default slice number
DISTORTION_TOLERANCE = 2.00  # mm


def detect_edges(image_array: np.ndarray,
                lower_threshold: int = EDGE_LOWER_THRESHOLD,
                upper_threshold: int = EDGE_UPPER_THRESHOLD) -> np.ndarray:
    """
    Apply Canny edge detection to image array using OpenCV.

    Args:
        image_array: Input image array
        lower_threshold: Lower threshold for edge detection
        upper_threshold: Upper threshold for edge detection

    Returns:
        Edge-detected image array
    """
    # Normalize to 8-bit for OpenCV
    image_8bit = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply Canny edge detection
    edges = cv2.Canny(image_8bit, lower_threshold, upper_threshold)

    return edges


def get_pixel_spacing_from_metadata(metadata: dict) -> tuple:
    """
    Extract pixel spacing from metadata dictionary.

    Args:
        metadata: Metadata dictionary from DicomReader

    Returns:
        Tuple of (pixel_spacing_x, pixel_spacing_y) in mm
    """
    pixel_spacing = metadata['pixel_spacing']
    if pixel_spacing:
        return float(pixel_spacing[0]), float(pixel_spacing[1])
    else:
        raise ValueError("No pixel spacing found in metadata")


def compute_fwhm(image: np.ndarray, edge_image: np.ndarray,
                pixel_spacing_x: float, pixel_spacing_y: float) -> tuple:
    """
    Compute Full Width at Half Maximum (FWHM) from edge-detected image.

    Args:
        image: Original image array
        edge_image: Edge-detected image array
        pixel_spacing_x: Pixel spacing in x direction (mm)
        pixel_spacing_y: Pixel spacing in y direction (mm)

    Returns:
        Tuple of (centroid, fwhm_x, fwhm_y)
    """
    # Compute centroid
    centroid = ndimage.center_of_mass(image)

    # Get integer indices
    i = int(round(centroid[0]))
    j = int(round(centroid[1]))

    # Find edges along x and y axes through the centroid
    # For 2D images, access directly
    index1 = np.nonzero(edge_image[:, j])
    ind1 = list(index1[0])
    index2 = np.nonzero(edge_image[i, :])
    ind2 = list(index2[0])

    # Calculate FWHM
    fwhm_x = (ind1[-1] - ind1[0]) * pixel_spacing_x
    fwhm_y = (ind2[-1] - ind2[0]) * pixel_spacing_y

    return centroid, fwhm_x, fwhm_y


def visualize_results(se_view: np.ndarray, se_edges: np.ndarray,
                     epi_view: np.ndarray, epi_edges: np.ndarray,
                     centroid_distance: float, output_dir: str):
    """
    Create and save visualization of images and edges.

    Args:
        se_view: Turbo spin echo image
        se_edges: TSE edges
        epi_view: EPI image
        epi_edges: EPI edges
        centroid_distance: Distance between centroids
        output_dir: Directory to save output
    """
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(np.squeeze(se_view), cmap=plt.cm.gray)
    axs[0, 0].set_title('2D TSE image')
    axs[1, 0].imshow(np.squeeze(se_edges), cmap=plt.cm.gray)
    axs[1, 0].set_title('2D TSE image edges')
    axs[0, 1].imshow(np.squeeze(epi_view), cmap=plt.cm.gray)
    axs[0, 1].set_title('2D EPI image')
    axs[0, 1].set_xlabel('Delta centroids = %.3f mm' % centroid_distance)
    axs[1, 1].imshow(np.squeeze(epi_edges), cmap=plt.cm.gray)
    axs[1, 1].set_title('2D EPI image edges')
    fig = plt.gcf()
    fig.set_size_inches(6, 8)

    plt.savefig(os.path.join(output_dir, 'Images&Edges.png'), dpi=300)
    plt.show()


def save_results(output_dir: str, se_fwhm_x: float, se_fwhm_y: float,
                epi_fwhm_x: float, epi_fwhm_y: float,
                x_diff: float, y_diff: float, centroid_dist: float):
    """
    Save analysis results to text file.

    Args:
        output_dir: Directory to save results
        se_fwhm_x: TSE FWHM along x
        se_fwhm_y: TSE FWHM along y
        epi_fwhm_x: EPI FWHM along x
        epi_fwhm_y: EPI FWHM along y
        x_diff: Difference in FWHM along x
        y_diff: Difference in FWHM along y
        centroid_dist: Distance between centroids
    """
    floats = np.array([se_fwhm_x, se_fwhm_y, epi_fwhm_x, epi_fwhm_y,
                      x_diff, y_diff, centroid_dist])
    names = np.array(['TSE freq FWHM=', 'TSE phase FWHM=', 'EPI freq FWHM=',
                     'EPI phase FWHM=', 'Geometric dist freq=',
                     'Geometric dist phase=', 'Diff in centroids='])
    ab = np.zeros(names.size, dtype=[('var1', 'U22'), ('var2', float)])
    ab['var1'] = names
    ab['var2'] = floats
    np.savetxt(os.path.join(output_dir, 'EPIgeoDist.txt'), ab, fmt='%22s %10.5f')


def main():
    """Main execution function for geometric distortion analysis."""

    # Load sequences using configuration
    # print(style.YELLOW + 'Select base directory containing geometric distortion sequences' + style.RESET)
    base_directory = dicom_reader.select_dicom_directory(
        'Select base directory containing sequences'
    )

    # Load both TSE and EPI sequences automatically
    loaded_sequences = dicom_reader.load_qa_sequences(base_directory, 'geometric_distortion')

    if len(loaded_sequences) != 2:
        print(style.RED + f'Error: Expected 2 sequences (TSE and EPI), found {len(loaded_sequences)}' + style.RESET)
        return

    # Extract turbo spin echo and EPI data
    if 'turbo_spin_echo' not in loaded_sequences or 'epi' not in loaded_sequences:
        print(style.RED + 'Error: Could not find required turbo_spin_echo and epi sequences' + style.RESET)
        print(f'Found sequences: {list(loaded_sequences.keys())}')
        return

    se_data, se_metadata, se_export_error = loaded_sequences['turbo_spin_echo']
    epi_data, epi_metadata, epi_export_error = loaded_sequences['epi']

    print(style.GREEN + f'Successfully loaded TSE data: {se_data.shape}' + style.RESET)
    print(style.GREEN + f'Successfully loaded EPI data: {epi_data.shape}' + style.RESET)

    # Select slice to analyze - use config value if available
    sl = dicom_reader.get_slice_number_for_test('geometric_distortion')
    if sl is None:
        sl = SLICE_NUMBER  # Fallback to default
    if sl >= se_data.shape[2]:
        sl = se_data.shape[2] // 2  # Use middle slice if default is out of range
    print(style.WHITE + f'Analyzing slice {sl}' + style.RESET)

    # Extract selected slice
    se_slice = se_data[:, :, sl]
    epi_slice = epi_data[:, :, epi_metadata['rows'] - sl - 1] if epi_export_error else epi_data[:, :, sl]

    # Get pixel spacing from metadata
    se_pixel_spacing_x, se_pixel_spacing_y = get_pixel_spacing_from_metadata(se_metadata)
    epi_pixel_spacing_x, epi_pixel_spacing_y = get_pixel_spacing_from_metadata(epi_metadata)

    print(style.CYAN + f'TSE pixel spacing: {se_pixel_spacing_x:.3f} x {se_pixel_spacing_y:.3f} mm' + style.RESET)
    print(style.CYAN + f'EPI pixel spacing: {epi_pixel_spacing_x:.3f} x {epi_pixel_spacing_y:.3f} mm' + style.RESET)

    # Perform edge detection
    print(style.CYAN + 'Performing edge detection on TSE image...' + style.RESET)
    se_edges = detect_edges(se_slice)

    print(style.CYAN + 'Performing edge detection on EPI image...' + style.RESET)
    epi_edges = detect_edges(epi_slice)

    # Compute FWHM for TSE
    se_centroid, se_fwhm_x, se_fwhm_y = compute_fwhm(
        se_slice, se_edges, se_pixel_spacing_x, se_pixel_spacing_y
    )

    print(style.WHITE + f'TSE image centroid = {se_centroid}')
    print(style.WHITE + f'TSE image FWHM along X = {se_fwhm_x:.3f} mm')
    print(style.WHITE + f'TSE image FWHM along Y = {se_fwhm_y:.3f} mm' + style.RESET)

    # Compute FWHM for EPI
    epi_centroid, epi_fwhm_x, epi_fwhm_y = compute_fwhm(
        epi_slice, epi_edges, epi_pixel_spacing_x, epi_pixel_spacing_y
    )

    print(style.WHITE + f'EPI image centroid = {epi_centroid}')
    print(style.WHITE + f'EPI image FWHM along X = {epi_fwhm_x:.3f} mm')
    print(style.WHITE + f'EPI image FWHM along Y = {epi_fwhm_y:.3f} mm' + style.RESET)

    # Compute differences
    x_diff = epi_fwhm_x - se_fwhm_x
    y_diff = epi_fwhm_y - se_fwhm_y
    centroid_dist = np.sqrt((se_centroid[0] - epi_centroid[0])**2 +
                           (se_centroid[1] - epi_centroid[1])**2)

    # Display results with color coding
    if np.abs(x_diff) <= DISTORTION_TOLERANCE:
        print(style.GREEN + f'Difference in FWHM along x = {x_diff:.3f} mm' + style.RESET)
    else:
        print(style.RED + f'Difference in FWHM along x = {x_diff:.3f} mm' + style.RESET)

    if np.abs(y_diff) <= DISTORTION_TOLERANCE:
        print(style.GREEN + f'Difference in FWHM along y = {y_diff:.3f} mm' + style.RESET)
    else:
        print(style.RED + f'Difference in FWHM along y = {y_diff:.3f} mm' + style.RESET)

    if np.abs(centroid_dist) <= DISTORTION_TOLERANCE / 2:
        print(style.GREEN + f'Distance in centroids = {centroid_dist:.3f} mm' + style.RESET)
    else:
        print(style.RED + f'Distance in centroids = {centroid_dist:.3f} mm' + style.RESET)

    # Select output directory
    print(style.YELLOW + 'Select a directory to save the output' + style.RESET)
    root = tk.Tk()
    root.withdraw()
    output_dir = filedialog.askdirectory()

    # Visualize and save results
    visualize_results(se_slice, se_edges, epi_slice, epi_edges,
                     centroid_dist, output_dir)
    save_results(output_dir, se_fwhm_x, se_fwhm_y, epi_fwhm_x, epi_fwhm_y,
                x_diff, y_diff, centroid_dist)

    print(style.GREEN + f'\nAnalysis complete! Results saved to {output_dir}' + style.RESET)


if __name__ == '__main__':
    main()
