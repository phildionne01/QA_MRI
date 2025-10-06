import numpy as np
import os
import json
import pydicom as dcm
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple, Optional, Union, Dict


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
        # First, collect all instance numbers to create a mapping
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
                if not metadata:  # Store metadata from first file
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
    
    def find_sequences_in_directory(self, base_directory: str, qa_test: str) -> Dict[str, str]:
        """
        Automatically find sequence directories based on configuration.
        
        Args:
            base_directory: Base directory containing all sequences for the test session
            qa_test: Name of the QA test (e.g., 'b0_inhomogeneity')
            
        Returns:
            Dictionary mapping sequence roles to directory paths
        """
        if not self.config or qa_test not in self.config:
            print(f"{self.style.RED}No configuration found for QA test: {qa_test}{self.style.RESET}")
            return {}
        
        test_config = self.config[qa_test]['sequences']
        sequence_paths = {}
        
        print(f"{self.style.BLUE}Searching for sequences in: {base_directory}{self.style.RESET}")
        
        # Walk through all subdirectories to find matching sequences
        for root, dirs, files in os.walk(base_directory):
            # Check if this directory contains DICOM files
            if any(f.lower().endswith('.dcm') for f in files):
                dir_name = os.path.basename(root).lower()
                
                # Try to match this directory to a sequence role
                for role, seq_config in test_config.items():
                    if role not in sequence_paths:  # Only assign if not already found
                        for pattern in seq_config['sequence_patterns']:
                            if pattern.lower() in dir_name:
                                sequence_paths[role] = root
                                print(f"{self.style.GREEN}Found {role}: {root}{self.style.RESET}")
                                break
        
        # Report missing sequences
        missing = set(test_config.keys()) - set(sequence_paths.keys())
        if missing:
            print(f"{self.style.YELLOW}Warning: Could not find sequences for: {', '.join(missing)}{self.style.RESET}")
        
        return sequence_paths
    
    def load_qa_sequences(self, base_directory: str, qa_test: str) -> Dict[str, Tuple[np.ndarray, dict, bool]]:
        """
        Load all required sequences for a QA test automatically.
        Handles both separated sequences and mixed multi-echo sequences.
        
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
        
        # Check if this test uses dual-echo structure (phase/magnitude folders)
        if test_config.get('dual_echo_structure', False):
            print(f"{self.style.BLUE}Loading dual-echo sequence structure...{self.style.RESET}")
            return self.load_dual_echo_sequences(base_directory, qa_test)
        
        # Check if this test uses mixed sequences
        if 'mixed_sequence_patterns' in test_config:
            print(f"{self.style.BLUE}Checking for mixed multi-echo sequences...{self.style.RESET}")
            
            # Look for mixed sequence directory
            mixed_dir = None
            for root, dirs, files in os.walk(base_directory):
                if any(f.lower().endswith('.dcm') for f in files):
                    dir_name = os.path.basename(root).lower()
                    for pattern in test_config['mixed_sequence_patterns']:
                        if pattern.lower() in dir_name:
                            mixed_dir = root
                            print(f"{self.style.GREEN}Found mixed sequence directory: {mixed_dir}{self.style.RESET}")
                            break
                    if mixed_dir:
                        break
            
            # If mixed directory found, use multi-echo loading
            if mixed_dir:
                return self.load_multi_echo_sequence(mixed_dir, qa_test)
        
        # Fallback to standard separated sequence loading
        print(f"{self.style.BLUE}Loading separated sequences...{self.style.RESET}")
        sequence_paths = self.find_sequences_in_directory(base_directory, qa_test)
        loaded_sequences = {}
        
        for role, path in sequence_paths.items():
            print(f"{self.style.CYAN}Loading {role} from {path}{self.style.RESET}")
            files = self.find_dicom_files(path)
            if files:
                try:
                    data, metadata, export_error = self.load_dicom_series(files)
                    loaded_sequences[role] = (data, metadata, export_error)
                except Exception as e:
                    print(f"{self.style.RED}Error loading {role}: {e}{self.style.RESET}")
            else:
                print(f"{self.style.RED}No DICOM files found in {path}{self.style.RESET}")
        
        return loaded_sequences
    
    def load_dual_echo_sequences(self, base_directory: str, qa_test: str) -> Dict[str, Tuple[np.ndarray, dict, bool]]:
        """
        Load dual-echo sequences where phase and magnitude folders each contain both echoes.
        
        Args:
            base_directory: Base directory containing phase and magnitude folders
            qa_test: Name of the QA test
            
        Returns:
            Dictionary mapping sequence roles to loaded data
        """
        test_config = self.config[qa_test]
        base_sequences = test_config['base_sequences']
        
        # Find phase and magnitude folders by examining DICOM ImageType
        found_folders = {}
        for root, dirs, files in os.walk(base_directory):
            dicom_files = [f for f in files if f.lower().endswith('.dcm')]
            if dicom_files:
                dir_name = os.path.basename(root).lower()
                
                # Check if directory name matches any sequence pattern
                matches_pattern = False
                for folder_type, folder_config in base_sequences.items():
                    for pattern in folder_config['sequence_patterns']:
                        if pattern.lower() in dir_name:
                            matches_pattern = True
                            break
                    if matches_pattern:
                        break
                
                if matches_pattern:
                    # Examine a sample DICOM file to determine image type
                    try:
                        sample_file = os.path.join(root, dicom_files[0])
                        dataset = dcm.dcmread(sample_file)
                        
                        if hasattr(dataset, 'ImageType') and len(dataset.ImageType) > 2:
                            image_type_tag = dataset.ImageType[2]
                            
                            if image_type_tag == 'P' and 'phase_folder' not in found_folders:
                                found_folders['phase_folder'] = root
                                print(f"{self.style.GREEN}Found phase_folder: {root} (ImageType: {image_type_tag}){self.style.RESET}")
                            elif image_type_tag == 'M' and 'magnitude_folder' not in found_folders:
                                found_folders['magnitude_folder'] = root
                                print(f"{self.style.GREEN}Found magnitude_folder: {root} (ImageType: {image_type_tag}){self.style.RESET}")
                            else:
                                print(f"{self.style.YELLOW}Unknown or duplicate ImageType: {image_type_tag} in {root}{self.style.RESET}")
                        else:
                            print(f"{self.style.YELLOW}No ImageType found in {sample_file}{self.style.RESET}")
                    except Exception as e:
                        print(f"{self.style.RED}Error reading {sample_file}: {e}{self.style.RESET}")
        
        if len(found_folders) != len(base_sequences):
            missing = set(base_sequences.keys()) - set(found_folders.keys())
            print(f"{self.style.RED}Missing folders: {', '.join(missing)}{self.style.RESET}")
            return {}
        
        # Separate echoes within each folder and load required sequences
        loaded_sequences = {}
        sequences_config = test_config['sequences']
        
        for seq_name, seq_config in sequences_config.items():
            source_folder_type = seq_config['source_folder']
            target_echo = seq_config['echo_number']
            
            if source_folder_type not in found_folders:
                print(f"{self.style.RED}Source folder {source_folder_type} not found for {seq_name}{self.style.RESET}")
                continue
            
            folder_path = found_folders[source_folder_type]
            print(f"{self.style.CYAN}Loading {seq_name} (TE{target_echo}) from {folder_path}{self.style.RESET}")
            
            # Get all files and separate by echo time
            all_files = self.find_dicom_files(folder_path)
            if not all_files:
                print(f"{self.style.RED}No DICOM files found in {folder_path}{self.style.RESET}")
                continue
            
            # Group files by echo time
            echo_groups = {}
            for file_path in all_files:
                try:
                    dataset = dcm.dcmread(file_path)
                    echo_time = getattr(dataset, 'EchoTime', 0)
                    
                    if echo_time not in echo_groups:
                        echo_groups[echo_time] = []
                    echo_groups[echo_time].append(file_path)
                except Exception as e:
                    print(f"{self.style.RED}Error reading {file_path}: {e}{self.style.RESET}")
            
            # Sort echo times and find target echo
            sorted_echoes = sorted(echo_groups.keys())
            if len(sorted_echoes) < target_echo:
                print(f"{self.style.RED}Not enough echoes found for {seq_name}. Found {len(sorted_echoes)}, need {target_echo}{self.style.RESET}")
                continue
            
            # Get files for target echo (1-indexed)
            target_echo_time = sorted_echoes[target_echo - 1]
            target_files = echo_groups[target_echo_time]
            
            print(f"{self.style.GREEN}Found TE{target_echo} with {len(target_files)} files (TE={target_echo_time}ms){self.style.RESET}")
            
            # Load the data
            try:
                data, metadata, export_error = self.load_dicom_series(target_files)
                loaded_sequences[seq_name] = (data, metadata, export_error)
            except Exception as e:
                print(f"{self.style.RED}Error loading {seq_name}: {e}{self.style.RESET}")
        
        return loaded_sequences
    
    def sort_mixed_sequence_directory(self, source_directory: str, 
                                    create_subfolders: bool = True) -> Dict[str, List[str]]:
        """
        Sort mixed DICOM files by protocol, echo time, and image type.
        Based on dcmsort_v2.py logic.
        
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
        sorted_groups = {}
        
        for file_path in dicom_files:
            try:
                dataset = dcm.dcmread(file_path)
                
                # Extract sorting criteria
                protocol_name = getattr(dataset, 'ProtocolName', 'Unknown')
                series_number = getattr(dataset, 'SeriesNumber', 0)
                echo_time = getattr(dataset, 'EchoTime', 0)
                scan_date = getattr(dataset, 'PerformedProcedureStepStartDate', 'Unknown')
                
                # Get image type (phase/magnitude)
                image_type = 'Unknown'
                if hasattr(dataset, 'ImageType') and len(dataset.ImageType) > 2:
                    image_type = dataset.ImageType[2]
                
                # Create sorting key
                sort_key = f"{protocol_name}_{scan_date}_{series_number}_{echo_time}_{image_type}"
                
                if sort_key not in sorted_groups:
                    sorted_groups[sort_key] = []
                sorted_groups[sort_key].append(file_path)
                
            except Exception as e:
                print(f"{self.style.RED}Error reading {file_path}: {e}{self.style.RESET}")
        
        # Create subdirectories if requested
        if create_subfolders:
            for sort_key, file_list in sorted_groups.items():
                subfolder_path = os.path.join(source_directory, sort_key)
                
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                    print(f"{self.style.GREEN}Created: {subfolder_path}{self.style.RESET}")
                
                # Copy files to subfolder (keeping originals)
                for file_path in file_list:
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(subfolder_path, filename)
                    if not os.path.exists(dest_path):
                        import shutil
                        shutil.copy2(file_path, dest_path)
        
        print(f"{self.style.GREEN}Sorted into {len(sorted_groups)} groups{self.style.RESET}")
        return sorted_groups
    
    def load_multi_echo_sequence(self, mixed_directory: str, qa_test: str) -> Dict[str, Tuple[np.ndarray, dict, bool]]:
        """
        Load multi-echo sequences that are mixed in a single directory.
        Automatically sorts and extracts required image types.
        
        Args:
            mixed_directory: Directory containing mixed echo/image type files
            qa_test: Name of the QA test to determine required sequences
            
        Returns:
            Dictionary mapping sequence roles to loaded data
        """
        print(f"{self.style.CYAN}Processing mixed multi-echo directory: {mixed_directory}{self.style.RESET}")
        
        # Sort the mixed directory
        sorted_groups = self.sort_mixed_sequence_directory(mixed_directory, create_subfolders=False)
        
        if not self.config or qa_test not in self.config:
            print(f"{self.style.RED}No configuration found for QA test: {qa_test}{self.style.RESET}")
            return {}
        
        test_config = self.config[qa_test]['sequences']
        loaded_sequences = {}
        
        # Find and load each required sequence type
        for role, seq_config in test_config.items():
            target_echo = seq_config.get('echo_number', 1)
            target_image_type = seq_config.get('image_type', 'magnitude').upper()
            
            # Find matching sorted group
            matching_files = []
            for sort_key, file_list in sorted_groups.items():
                # Parse sort key: protocol_date_series_echo_imagetype
                parts = sort_key.split('_')
                if len(parts) >= 5:
                    echo_str = parts[3]
                    image_type = parts[4].upper()
                    
                    try:
                        echo_num = float(echo_str)
                        
                        # Match by echo number and image type
                        if (abs(echo_num - target_echo * 10) < 1 or  # TE in ms (e.g., 5.0 vs 50)
                            abs(echo_num - target_echo) < 0.1) and \
                           target_image_type in image_type:
                            matching_files = file_list
                            print(f"{self.style.GREEN}Found {role}: {len(file_list)} files (TE={echo_num}, type={image_type}){self.style.RESET}")
                            break
                    except ValueError:
                        continue
            
            # Load the matching files
            if matching_files:
                try:
                    data, metadata, export_error = self.load_dicom_series(matching_files)
                    loaded_sequences[role] = (data, metadata, export_error)
                except Exception as e:
                    print(f"{self.style.RED}Error loading {role}: {e}{self.style.RESET}")
            else:
                print(f"{self.style.YELLOW}Warning: Could not find files for {role} (TE{target_echo}, {target_image_type}){self.style.RESET}")
        
        return loaded_sequences


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