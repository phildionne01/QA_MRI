import numpy as np
import os
import json
import pydicom as dcm
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class ConsoleStyle:
    """Console color styling for output messages."""
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


class DicomReader:
    """Handles DICOM directory selection, file discovery, and reading operations."""

    def __init__(self, config_file: str = None):
        os.system("")  # Enable color output in command window
        self.style = ConsoleStyle()
        self.config = self._load_config(config_file) if config_file else None

    def select_dicom_directory(self, prompt: str = "Select DICOM directory") -> str:
        """
        Open file dialog to select DICOM directory.

        Args:
            prompt: Message to display before directory selection

        Returns:
            Selected directory path
        """
        print(self.style.YELLOW + prompt + self.style.RESET)
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory()
        return path

    def find_dicom_files(self, directory: str) -> List[str]:
        """
        Find all DICOM files in directory and subdirectories.

        Args:
            directory: Directory path to search

        Returns:
            List of DICOM file paths
        """
        dicom_files = []
        for dir_name, subdir_list, file_list in os.walk(directory):
            for filename in file_list:
                if ".dcm" in filename.lower():
                    dicom_files.append(os.path.join(dir_name, filename))
        return dicom_files

    def check_export_error(self, ref_dataset: dcm.Dataset) -> bool:
        """
        Check for DICOM export error (missing row of pixels).

        Args:
            ref_dataset: Reference DICOM dataset

        Returns:
            True if export error detected, False otherwise
        """
        if ref_dataset.Rows == ref_dataset.Columns - 1:
            print(self.style.RED + 'DICOM export error detected. Will pad missing row of pixels with zeros' + self.style.RESET)
            return True
        return False

    def get_dicom_metadata(self, dicom_file: str) -> dict:
        """
        Extract common metadata from DICOM file.

        Args:
            dicom_file: Path to DICOM file

        Returns:
            Dictionary containing metadata
        """
        dataset = dcm.dcmread(dicom_file)
        metadata = {
            'rows': int(dataset.Rows),
            'columns': int(dataset.Columns),
            'bits_stored': dataset.BitsStored,
            'station_name': getattr(dataset, 'StationName', ''),
            'acquisition_type': getattr(dataset, 'MRAcquisitionType', ''),
            'echo_time': getattr(dataset, 'EchoTime', None),
            'imaging_frequency': getattr(dataset, 'ImagingFrequency', None),
            'pixel_spacing': getattr(dataset, 'PixelSpacing', None),
            'slice_location': getattr(dataset, 'SliceLocation', None),
            'protocol_name': getattr(dataset, 'ProtocolName', ''),
            'instance_number': getattr(dataset, 'InstanceNumber', None),
            'acquisition_number': getattr(dataset, 'AcquisitionNumber', None),
            'series_number': getattr(dataset, 'SeriesNumber', None)
        }
        return metadata

    def load_dicom_series(self, dicom_files: List[str],
                         station_specific: bool = True) -> Tuple[np.ndarray, dict, bool]:
        """
        Load DICOM series into numpy array with proper slice ordering.

        Args:
            dicom_files: List of DICOM file paths
            station_specific: Whether to apply station-specific ordering logic

        Returns:
            Tuple of (dicom_array, metadata, export_error_flag)
        """
        if not dicom_files:
            raise ValueError("No DICOM files provided")

        # Get reference dataset and metadata
        ref_dataset = dcm.dcmread(dicom_files[0])
        export_error = self.check_export_error(ref_dataset)
        metadata = self.get_dicom_metadata(dicom_files[0])

        # Create array dimensions
        dims = (metadata['rows'], metadata['columns'], len(dicom_files))
        dicom_array = np.zeros(dims, dtype=ref_dataset.pixel_array.dtype)

        print(f"MRI Scan mode: {metadata['acquisition_type']}")

        # Load pixel data with proper indexing
        instance_mapping = {}
        for i, file_path in enumerate(dicom_files):
            dataset = dcm.dcmread(file_path)
            inst_num = dataset.InstanceNumber
            instance_mapping[inst_num] = i

        # Sort by instance number and load data sequentially
        sorted_instances = sorted(instance_mapping.keys())

        if len(sorted_instances) != len(dicom_files):
            print(f"{self.style.YELLOW}Warning: Instance number mismatch. Expected {len(dicom_files)}, got {len(sorted_instances)}{self.style.RESET}")

        for slice_idx, inst_num in enumerate(sorted_instances):
            if slice_idx >= dicom_array.shape[2]:
                print(f"{self.style.RED}Warning: Too many slices for array size. Skipping instance {inst_num}{self.style.RESET}")
                break
            file_path = dicom_files[instance_mapping[inst_num]]
            dataset = dcm.dcmread(file_path)
            dicom_array[:, :, slice_idx] = dataset.pixel_array

        # Handle export error by padding with zeros
        if export_error:
            corrected_shape = (metadata['rows'] + 1, metadata['columns'], len(dicom_files))
            corrected_array = np.zeros(corrected_shape)
            corrected_array[1:, :, :] = dicom_array
            dicom_array = corrected_array

        print(f'MRI dataset size: {np.shape(dicom_array)}')
        return dicom_array, metadata, export_error

    def load_multi_acquisition_series(self, dicom_files: List[str]) -> Tuple[np.ndarray, dict]:
        """
        Load DICOM series with multiple acquisitions (e.g., for EPI stability analysis).

        Args:
            dicom_files: List of DICOM file paths

        Returns:
            Tuple of (4D dicom_array, metadata)
        """
        if not dicom_files:
            raise ValueError("No DICOM files provided")

        # Determine acquisition structure
        max_acq_num = 0
        metadata = {}

        for file_path in dicom_files:
            dataset = dcm.dcmread(file_path)
            if max_acq_num < int(dataset.AcquisitionNumber):
                max_acq_num = int(dataset.AcquisitionNumber)
                if not metadata:
                    metadata = self.get_dicom_metadata(file_path)

        total_slices = len(dicom_files) // max_acq_num
        dims = (metadata['rows'], metadata['columns'], total_slices, max_acq_num)
        dicom_array = np.zeros(dims)

        print(f'Total slices: {total_slices}')
        print(f'Max acquisition number: {max_acq_num}')
        print(f'Total images: {len(dicom_files)}')

        # Load data with proper indexing
        for file_path in dicom_files:
            dataset = dcm.dcmread(file_path)
            inst_num = int(dataset.InstanceNumber)
            acq_num = int(dataset.AcquisitionNumber)
            slice_num = total_slices - ((acq_num * total_slices) - inst_num)
            dicom_array[:, :, slice_num - 1, acq_num - 1] = dataset.pixel_array

        print(f'MRI dataset size: {np.shape(dicom_array)}')
        return dicom_array, metadata

    def _load_config(self, config_file: str) -> Dict:
        """Load sequence configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"{self.style.RED}Error loading config file {config_file}: {e}{self.style.RESET}")
            return {}

    def _group_files_by_property(self, files: List[str], property_func) -> Dict:
        """
        Group DICOM files by a specific property.

        Args:
            files: List of DICOM file paths
            property_func: Function that takes a dataset and returns grouping key

        Returns:
            Dictionary mapping property values to file lists
        """
        groups = defaultdict(list)
        for file_path in files:
            try:
                dataset = dcm.dcmread(file_path)
                key = property_func(dataset)
                groups[key].append(file_path)
            except Exception as e:
                print(f"{self.style.RED}Error reading {file_path}: {e}{self.style.RESET}")
        return dict(groups)

    def _separate_files_by_echo(self, files: List[str], target_echo: int) -> List[str]:
        """
        Separate files by echo time and return files for target echo.

        Args:
            files: List of DICOM file paths
            target_echo: Target echo number (1-indexed)

        Returns:
            List of files for target echo
        """
        echo_groups = self._group_files_by_property(
            files,
            lambda ds: getattr(ds, 'EchoTime', 0)
        )

        sorted_echoes = sorted(echo_groups.keys())
        if len(sorted_echoes) < target_echo:
            print(f"{self.style.RED}Not enough echoes. Found {len(sorted_echoes)}, need {target_echo}{self.style.RESET}")
            return []

        target_echo_time = sorted_echoes[target_echo - 1]
        print(f"{self.style.GREEN}Found TE{target_echo} with {len(echo_groups[target_echo_time])} files (TE={target_echo_time}ms){self.style.RESET}")
        return echo_groups[target_echo_time]

    def _find_directories_with_dicoms(self, base_directory: str) -> Dict[str, str]:
        """
        Find all directories containing DICOM files.

        Args:
            base_directory: Base directory to search

        Returns:
            Dictionary mapping directory names to paths
        """
        dicom_dirs = {}
        for root, dirs, files in os.walk(base_directory):
            if any(f.lower().endswith('.dcm') for f in files):
                dir_name = os.path.basename(root).lower()
                dicom_dirs[dir_name] = root
        return dicom_dirs

    def _match_directory_to_patterns(self, dir_name: str, patterns: List[str]) -> bool:
        """Check if directory name matches any pattern."""
        return any(pattern.lower() in dir_name for pattern in patterns)

    def find_sequences_in_directory(self, base_directory: str, qa_test: str) -> Dict[str, str]:
        """
        Automatically find sequence directories based on configuration.

        Args:
            base_directory: Base directory containing all sequences
            qa_test: Name of the QA test

        Returns:
            Dictionary mapping sequence roles to directory paths
        """
        if not self.config or qa_test not in self.config:
            print(f"{self.style.RED}No configuration found for QA test: {qa_test}{self.style.RESET}")
            return {}

        test_config = self.config[qa_test]['sequences']
        sequence_paths = {}

        print(f"{self.style.BLUE}Searching for sequences in: {base_directory}{self.style.RESET}")

        # Find all directories with DICOM files
        dicom_dirs = self._find_directories_with_dicoms(base_directory)

        # Match directories to sequence roles
        for role, seq_config in test_config.items():
            for dir_name, dir_path in dicom_dirs.items():
                if role not in sequence_paths:
                    if self._match_directory_to_patterns(dir_name, seq_config['sequence_patterns']):
                        sequence_paths[role] = dir_path
                        print(f"{self.style.GREEN}Found {role}: {dir_path}{self.style.RESET}")
                        break

        # Report missing sequences
        missing = set(test_config.keys()) - set(sequence_paths.keys())
        if missing:
            print(f"{self.style.YELLOW}Warning: Could not find sequences for: {', '.join(missing)}{self.style.RESET}")

        return sequence_paths

    def _load_sequence_with_echo_separation(self, files: List[str], seq_config: dict) -> Tuple[np.ndarray, dict, bool]:
        """
        Load a sequence, separating by echo time if needed.

        Args:
            files: List of DICOM file paths
            seq_config: Sequence configuration dictionary

        Returns:
            Tuple of (data, metadata, export_error)
        """
        # Check if echo separation is needed
        if 'echo_number' in seq_config:
            target_echo = seq_config['echo_number']
            files = self._separate_files_by_echo(files, target_echo)
            if not files:
                raise ValueError(f"No files found for echo {target_echo}")

        return self.load_dicom_series(files)

    def get_slice_number_for_test(self, qa_test: str) -> Optional[int]:
        """
        Get the recommended slice number for a QA test from config.

        Args:
            qa_test: Name of the QA test

        Returns:
            Slice number if specified in config, None otherwise
        """
        if self.config and qa_test in self.config:
            return self.config[qa_test].get('slice_number', None)
        return None

    def load_qa_sequences(self, base_directory: str, qa_test: str) -> Dict[str, Tuple[np.ndarray, dict, bool]]:
        """
        Load all required sequences for a QA test automatically.
        Handles separated sequences, multi-echo, and dual-echo structures.

        Args:
            base_directory: Base directory containing all sequences
            qa_test: Name of the QA test

        Returns:
            Dictionary mapping sequence roles to loaded data
        """
        if not base_directory:
            base_directory = self.select_dicom_directory(f"Select base directory for {qa_test} QA test")

        if not self.config or qa_test not in self.config:
            print(f"{self.style.RED}No configuration found for QA test: {qa_test}{self.style.RESET}")
            return {}

        test_config = self.config[qa_test]

        # Display slice number if configured
        slice_num = self.get_slice_number_for_test(qa_test)
        if slice_num is not None:
            print(f"{self.style.CYAN}Configured slice number for {qa_test}: {slice_num}{self.style.RESET}")

        # Check for dual-echo structure (special case: phase/magnitude folders)
        if test_config.get('dual_echo_structure', False):
            return self._load_dual_echo_structure(base_directory, qa_test)

        # Standard loading: find sequence directories
        sequence_paths = self.find_sequences_in_directory(base_directory, qa_test)
        loaded_sequences = {}

        for role, path in sequence_paths.items():
            print(f"{self.style.CYAN}Loading {role} from {path}{self.style.RESET}")
            files = self.find_dicom_files(path)
            if files:
                try:
                    seq_config = test_config['sequences'][role]
                    data, metadata, export_error = self._load_sequence_with_echo_separation(files, seq_config)
                    loaded_sequences[role] = (data, metadata, export_error)
                except Exception as e:
                    print(f"{self.style.RED}Error loading {role}: {e}{self.style.RESET}")
            else:
                print(f"{self.style.RED}No DICOM files found in {path}{self.style.RESET}")

        return loaded_sequences

    def _load_dual_echo_structure(self, base_directory: str, qa_test: str) -> Dict[str, Tuple[np.ndarray, dict, bool]]:
        """
        Load dual-echo sequences where phase/magnitude folders each contain multiple echoes.

        Args:
            base_directory: Base directory containing phase and magnitude folders
            qa_test: Name of the QA test

        Returns:
            Dictionary mapping sequence roles to loaded data
        """
        test_config = self.config[qa_test]

        print(f"{self.style.BLUE}Loading dual-echo sequence structure...{self.style.RESET}")

        # Find phase and magnitude folders by ImageType
        dicom_dirs = self._find_directories_with_dicoms(base_directory)
        found_folders = {}

        for dir_name, dir_path in dicom_dirs.items():
            # Check if matches base sequence patterns
            matches_any = False
            for folder_type, folder_config in test_config['base_sequences'].items():
                if self._match_directory_to_patterns(dir_name, folder_config['sequence_patterns']):
                    matches_any = True
                    break

            if not matches_any:
                continue

            # Determine if phase or magnitude by examining ImageType
            files = self.find_dicom_files(dir_path)
            if files:
                try:
                    dataset = dcm.dcmread(files[0])
                    if hasattr(dataset, 'ImageType') and len(dataset.ImageType) > 2:
                        image_type = dataset.ImageType[2]
                        if image_type == 'P':
                            found_folders['phase_folder'] = dir_path
                            print(f"{self.style.GREEN}Found phase_folder: {dir_path}{self.style.RESET}")
                        elif image_type == 'M':
                            found_folders['magnitude_folder'] = dir_path
                            print(f"{self.style.GREEN}Found magnitude_folder: {dir_path}{self.style.RESET}")
                except Exception as e:
                    print(f"{self.style.RED}Error reading {files[0]}: {e}{self.style.RESET}")

        # Load required sequences
        loaded_sequences = {}
        for seq_name, seq_config in test_config['sequences'].items():
            source_folder = seq_config['source_folder']

            if source_folder not in found_folders:
                print(f"{self.style.RED}Source folder {source_folder} not found for {seq_name}{self.style.RESET}")
                continue

            folder_path = found_folders[source_folder]
            print(f"{self.style.CYAN}Loading {seq_name} from {folder_path}{self.style.RESET}")

            files = self.find_dicom_files(folder_path)
            if files:
                try:
                    data, metadata, export_error = self._load_sequence_with_echo_separation(files, seq_config)
                    loaded_sequences[seq_name] = (data, metadata, export_error)
                except Exception as e:
                    print(f"{self.style.RED}Error loading {seq_name}: {e}{self.style.RESET}")

        return loaded_sequences

    def sort_mixed_sequence_directory(self, source_directory: str,
                                    create_subfolders: bool = True) -> Dict[str, List[str]]:
        """
        Sort mixed DICOM files by protocol, echo time, and image type.

        Args:
            source_directory: Directory containing mixed DICOM files
            create_subfolders: Whether to create sorted subdirectories

        Returns:
            Dictionary mapping sorted categories to file lists
        """
        dicom_files = self.find_dicom_files(source_directory)
        if not dicom_files:
            print(f"{self.style.RED}No DICOM files found in {source_directory}{self.style.RESET}")
            return {}

        print(f"{self.style.BLUE}Sorting {len(dicom_files)} DICOM files...{self.style.RESET}")

        # Group files by sorting criteria
        def get_sort_key(dataset):
            protocol = getattr(dataset, 'ProtocolName', 'Unknown')
            series = getattr(dataset, 'SeriesNumber', 0)
            echo = getattr(dataset, 'EchoTime', 0)
            date = getattr(dataset, 'PerformedProcedureStepStartDate', 'Unknown')
            img_type = dataset.ImageType[2] if hasattr(dataset, 'ImageType') and len(dataset.ImageType) > 2 else 'Unknown'
            return f"{protocol}_{date}_{series}_{echo}_{img_type}"

        sorted_groups = self._group_files_by_property(dicom_files, get_sort_key)

        # Create subdirectories if requested
        if create_subfolders:
            import shutil
            for sort_key, file_list in sorted_groups.items():
                subfolder_path = os.path.join(source_directory, sort_key)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                    print(f"{self.style.GREEN}Created: {subfolder_path}{self.style.RESET}")

                for file_path in file_list:
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(subfolder_path, filename)
                    if not os.path.exists(dest_path):
                        shutil.copy2(file_path, dest_path)

        print(f"{self.style.GREEN}Sorted into {len(sorted_groups)} groups{self.style.RESET}")
        return sorted_groups


def get_bit_depth_max_value(bits_stored: int) -> int:
    """
    Get maximum value for given bit depth.

    Args:
        bits_stored: Number of bits stored in DICOM

    Returns:
        Maximum value for the bit depth
    """
    if bits_stored == 12:
        return 4096
    elif bits_stored == 16:
        return 65536
    else:
        return 2 ** bits_stored


def save_results_to_file(results: dict, output_path: str, filename: str = 'analysis_results.txt'):
    """
    Save analysis results to text file.

    Args:
        results: Dictionary of result names and values
        output_path: Directory to save results
        filename: Output filename
    """
    names = np.array(list(results.keys()))
    values = np.array(list(results.values()))

    structured_array = np.zeros(names.size, dtype=[('var1', 'U50'), ('var2', float)])
    structured_array['var1'] = names
    structured_array['var2'] = values

    output_file = os.path.join(output_path, filename)
    np.savetxt(output_file, structured_array, fmt='%30s %10.5f')
    print(f'Results saved to: {output_file}')
