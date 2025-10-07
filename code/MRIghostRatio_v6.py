import matplotlib.pyplot as plt
import numpy as np
import cv2
from mri_utils import DicomReader, ConsoleStyle, save_results_to_file


def detect_phantom_and_create_rois(image, center_roi_radius_fraction=0.8, background_roi_size=None):
    """
    Automatically detect the phantom in the image and create ROIs.

    Args:
        image: 2D numpy array of the MRI slice
        center_roi_radius_fraction: Fraction of phantom radius for center ROI (default: 0.8)
        background_roi_size: Size of background ROI boxes in pixels (auto-calculated if None)

    Returns:
        Dictionary containing masks for center, top, bottom, left, right ROIs
    """
    # Normalize image to 8-bit for OpenCV processing
    img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img_normalized, (5, 5), 0)

    # Use Otsu's thresholding to segment the phantom
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (should be the phantom)
    if not contours:
        raise ValueError("No phantom contour detected in the image")

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
    cx, cy, radius = int(cx), int(cy), int(radius)

    # Calculate ROI sizes based on phantom size
    center_roi_radius = int(radius * center_roi_radius_fraction)
    if background_roi_size is None:
        # Make background ROI size proportional to phantom radius (about 15% of phantom radius)
        background_roi_size = max(10, int(radius * 0.15))

    # Create masks for ROIs
    masks = {}
    img_height, img_width = image.shape

    # Center ROI - circular in the center of the phantom
    masks['center'] = np.zeros(image.shape, dtype=bool)
    y_coords, x_coords = np.ogrid[:img_height, :img_width]
    center_circle = (x_coords - cx)**2 + (y_coords - cy)**2 <= center_roi_radius**2
    masks['center'] = center_circle

    # Background ROIs - rectangular, positioned outside the phantom
    # Margin to ensure ROIs are clearly outside the phantom
    margin = int(radius * 0.15)  # 15% margin beyond phantom edge
    half_roi = background_roi_size // 2

    # Top ROI - rectangular
    top_y_center = max(cy - radius - margin - half_roi, half_roi)
    top_y_start = max(top_y_center - half_roi, 0)
    top_y_end = min(top_y_center + half_roi, img_height)
    masks['top'] = np.zeros(image.shape, dtype=bool)
    masks['top'][top_y_start:top_y_end, cx-half_roi:cx+half_roi] = True

    # Bottom ROI - rectangular
    bottom_y_center = min(cy + radius + margin + half_roi, img_height - half_roi)
    bottom_y_start = max(bottom_y_center - half_roi, 0)
    bottom_y_end = min(bottom_y_center + half_roi, img_height)
    masks['bottom'] = np.zeros(image.shape, dtype=bool)
    masks['bottom'][bottom_y_start:bottom_y_end, cx-half_roi:cx+half_roi] = True

    # Left ROI - rectangular
    left_x_center = max(cx - radius - margin - half_roi, half_roi)
    left_x_start = max(left_x_center - half_roi, 0)
    left_x_end = min(left_x_center + half_roi, img_width)
    masks['left'] = np.zeros(image.shape, dtype=bool)
    masks['left'][cy-half_roi:cy+half_roi, left_x_start:left_x_end] = True

    # Right ROI - rectangular
    right_x_center = min(cx + radius + margin + half_roi, img_width - half_roi)
    right_x_start = max(right_x_center - half_roi, 0)
    right_x_end = min(right_x_center + half_roi, img_width)
    masks['right'] = np.zeros(image.shape, dtype=bool)
    masks['right'][cy-half_roi:cy+half_roi, right_x_start:right_x_end] = True

    return masks, (cx, cy, radius)


def calculate_ghosting_ratio(image, masks):
    """
    Calculate ghosting ratio from image and ROI masks.

    Args:
        image: 2D numpy array of the MRI slice
        masks: Dictionary containing masks for center, top, bottom, left, right

    Returns:
        Dictionary containing mean signals and ghosting ratio
    """
    # Calculate mean signal in each ROI
    centre = np.mean(image[masks['center']])
    top = np.mean(image[masks['top']])
    bottom = np.mean(image[masks['bottom']])
    left = np.mean(image[masks['left']])
    right = np.mean(image[masks['right']])

    # Calculate ghosting ratio
    ghosting_ratio = np.abs((top + bottom) - (left + right)) / (2 * centre) * 100

    results = {
        'centre': centre,
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right,
        'ghosting_ratio': ghosting_ratio
    }

    return results


def visualize_rois(image, masks, phantom_info, results, title="ROI Visualization"):
    """
    Visualize the image with ROI overlays and results.

    Args:
        image: 2D numpy array of the MRI slice
        masks: Dictionary containing masks for all ROIs
        phantom_info: Tuple of (center_x, center_y, radius)
        results: Dictionary containing calculated results
        title: Plot title

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')

    # Draw phantom circle
    cx, cy, radius = phantom_info
    circle = plt.Circle((cx, cy), radius, color='yellow', fill=False, linewidth=2, label='Phantom')
    ax.add_patch(circle)

    # Draw ROIs with different colors
    colors = {'center': 'red', 'top': 'magenta', 'bottom': 'cyan', 'left': 'blue', 'right': 'green'}
    # Map mask keys to result keys
    result_key_map = {'center': 'centre', 'top': 'top', 'bottom': 'bottom', 'left': 'left', 'right': 'right'}

    for roi_name, color in colors.items():
        # Find bounding box of the ROI
        y_coords, x_coords = np.where(masks[roi_name])
        if len(y_coords) > 0:
            result_key = result_key_map[roi_name]

            if roi_name == 'center':
                # Draw circle for center ROI
                y_center = (y_coords.min() + y_coords.max()) / 2
                x_center = (x_coords.min() + x_coords.max()) / 2
                roi_radius = (x_coords.max() - x_coords.min()) / 2
                center_circle = plt.Circle((x_center, y_center), roi_radius,
                                          edgecolor=color, facecolor='none', linewidth=2,
                                          label=f'{roi_name.capitalize()}: {results[result_key]:.1f}')
                ax.add_patch(center_circle)
            else:
                # Draw rectangle for background ROIs
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    edgecolor=color, facecolor='none', linewidth=2,
                                    label=f'{roi_name.capitalize()}: {results[result_key]:.1f}')
                ax.add_patch(rect)

    ax.set_title(f'{title}\nGhosting Ratio: {results["ghosting_ratio"]:.2f}%', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.axis('off')
    plt.tight_layout()

    return fig


def process_sequence(dicom_array, slice_number, sequence_name, output_dir,
                     center_roi_radius_fraction=0.25, background_roi_size=None, show_plot=True):
    """
    Process a single sequence to calculate ghosting ratio.

    Args:
        dicom_array: 3D numpy array of DICOM images
        slice_number: Slice index to analyze
        sequence_name: Name of the sequence (for output)
        output_dir: Directory to save results
        center_roi_radius_fraction: Fraction of phantom radius for center ROI
        background_roi_size: Size of background ROI boxes (auto if None)
        show_plot: Whether to display the plot interactively (default: True)

    Returns:
        Dictionary containing results
    """
    # Extract the specified slice
    image = dicom_array[:, :, slice_number]

    # Detect phantom and create ROIs
    masks, phantom_info = detect_phantom_and_create_rois(image, center_roi_radius_fraction, background_roi_size)

    # Calculate ghosting ratio
    results = calculate_ghosting_ratio(image, masks)

    # Visualize and save
    fig = visualize_rois(image, masks, phantom_info, results,
                        title=f"{sequence_name} - Slice {slice_number}")
    fig.savefig(f'{output_dir}/{sequence_name}_ROIs.png', dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Save numerical results
    save_data = {
        f'{sequence_name}_Mean_signal': results['centre'],
        f'{sequence_name}_Top_signal': results['top'],
        f'{sequence_name}_Bottom_signal': results['bottom'],
        f'{sequence_name}_Left_signal': results['left'],
        f'{sequence_name}_Right_signal': results['right'],
        f'{sequence_name}_GhostingRatio': results['ghosting_ratio']
    }

    return results, save_data


def main():
    import os
    style = ConsoleStyle()

    # Load config file
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'qa_sequences_config.json')
    reader = DicomReader(config_file=config_path)

    # Ask if user wants to display plots for QA
    print(style.CYAN + "Do you want to display the ROI plots for QA purposes? (y/n): " + style.RESET, end='')
    show_plots = input().strip().lower() in ['y', 'yes', '']

    # Select directory containing DICOM data
    print(style.YELLOW + 'Select the directory containing the DICOM datasets (EPI and TSE)' + style.RESET)
    base_dir = reader.select_dicom_directory()

    if not base_dir:
        print(style.RED + "No directory selected. Exiting." + style.RESET)
        return

    # Use config-based sequence detection
    print(style.CYAN + "\nSearching for ghost ratio sequences using configuration..." + style.RESET)
    sequence_paths = reader.find_sequences_in_directory(base_dir, 'ghost_ratio')

    # Initialize storage for both sequences
    sequences = {}

    # Map config keys to display names
    sequence_name_map = {'epi': 'EPI', 'tse': 'TSE'}

    # Load sequences found by config
    for seq_key, seq_path in sequence_paths.items():
        seq_name = sequence_name_map.get(seq_key, seq_key.upper())
        print(style.CYAN + f"Loading {seq_name} sequence..." + style.RESET)
        seq_files = reader.find_dicom_files(seq_path)
        if seq_files:
            seq_array, seq_metadata, seq_error = reader.load_dicom_series(seq_files)
            sequences[seq_name] = {'array': seq_array, 'metadata': seq_metadata, 'path': seq_path}
            print(style.GREEN + f"{seq_name} loaded: {seq_array.shape}" + style.RESET)
        else:
            print(style.RED + f"No {seq_name} DICOM files found" + style.RESET)

    if not sequences:
        print(style.RED + "No sequences loaded. Exiting." + style.RESET)
        return

    # Determine slice number based on station name (from original logic)
    slice_number = 20  # Default
    if 'EPI' in sequences:
        station = sequences['EPI']['metadata'].get('station_name', '')
        if station == 'AWP183025':
            slice_number = 20
        else:
            slice_number = 20

    print(style.CYAN + f"Analyzing slice: {slice_number}" + style.RESET)

    # Process each sequence
    all_results = {}
    all_save_data = {}

    for seq_name, seq_data in sequences.items():
        print(style.YELLOW + f"\nProcessing {seq_name} sequence..." + style.RESET)

        results, save_data = process_sequence(
            seq_data['array'],
            slice_number,
            seq_name,
            seq_data['path'],
            center_roi_radius_fraction=0.8,
            background_roi_size=None,  # Auto-calculated based on phantom size
            show_plot=show_plots
        )

        all_results[seq_name] = results
        all_save_data.update(save_data)

        # Print results
        print(style.YELLOW + f'\n{seq_name} Results:')
        print(f'  Mean signal: {results["centre"]:.2f}')
        print(f'  Top signal: {results["top"]:.2f}')
        print(f'  Bottom signal: {results["bottom"]:.2f}')
        print(f'  Left signal: {results["left"]:.2f}')
        print(f'  Right signal: {results["right"]:.2f}')
        print(f'  Ghosting Ratio: {results["ghosting_ratio"]:.2f}%' + style.RESET)

        # Check pass/fail (threshold: 3%)
        if results["ghosting_ratio"] <= 3:
            print(style.WHITE + f'{seq_name} QA test status: ' + style.GREEN + 'PASS' + style.RESET)
        else:
            print(style.WHITE + f'{seq_name} QA test status: ' + style.RED + 'FAIL' + style.RESET)

    # Compare sequences if both are available
    if len(sequences) == 2:
        print(style.CYAN + "\n=== Sequence Comparison ===" + style.RESET)
        epi_ratio = all_results['EPI']['ghosting_ratio']
        tse_ratio = all_results['TSE']['ghosting_ratio']
        difference = abs(epi_ratio - tse_ratio)

        print(f"EPI Ghosting Ratio: {epi_ratio:.2f}%")
        print(f"TSE Ghosting Ratio: {tse_ratio:.2f}%")
        print(f"Difference: {difference:.2f}%")

        all_save_data['Comparison_Difference'] = difference

        if difference < 1.0:
            print(style.GREEN + "Sequences show consistent ghosting levels (difference < 1%)" + style.RESET)
        elif difference < 2.0:
            print(style.YELLOW + "Moderate difference between sequences (1-2%)" + style.RESET)
        else:
            print(style.RED + "Significant difference between sequences (> 2%)" + style.RESET)

    # Save combined results
    save_results_to_file(all_save_data, base_dir, 'GhostingRatio_Combined.txt')

    print(style.GREEN + f"\n✓ Analysis complete! Results saved to {base_dir}" + style.RESET)


if __name__ == "__main__":
    main()
