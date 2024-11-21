import h5py
import numpy as np
import os
import pickle
import traceback

def convert_to_hdf5_type(obj):
    if isinstance(obj, (int, float, str, bytes)):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, list):
        return [convert_to_hdf5_type(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_hdf5_type(value) for key, value in obj.items()}
    else:
        return str(obj)  # Convert other types to strings

def save_to_hdf5(data, filename):
    try:
        with h5py.File(filename, 'w') as f:
            if isinstance(data, dict):
                for key, value in data.items():
                    converted_value = convert_to_hdf5_type(value)
                    if isinstance(converted_value, np.ndarray):
                        f.create_dataset(key, data=converted_value, compression="gzip")
                    else:
                        f.create_dataset(key, data=np.array(converted_value), compression="gzip")
            elif isinstance(data, np.ndarray):
                f.create_dataset('data', data=data, compression="gzip")
            else:
                converted_data = convert_to_hdf5_type(data)
                f.create_dataset('data', data=np.array(converted_data), compression="gzip")
        print(f"Successfully saved HDF5 file: {filename}")
        return True
    except Exception as e:
        print(f"Error saving HDF5 file {filename}: {str(e)}")
        print(traceback.format_exc())
        return False

def batch_convert_to_hdf5(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    print(f"Input directory: {os.path.abspath(input_directory)}")
    print(f"Output directory: {os.path.abspath(output_directory)}")
    
    files = [f for f in os.listdir(input_directory) if f.endswith('.pkl')]
    print(f"Found {len(files)} .pkl files in the input directory")
    
    if len(files) == 0:
        print("No .pkl files found in the input directory.")
        return

    print("List of .pkl files found:")
    for file in files:
        print(f"  - {file}")

    successful_conversions = 0
    for filename in files:
        print(f"\nProcessing file: {filename}")
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.hdf5')
        
        if os.path.exists(output_path):
            print(f"HDF5 file already exists for {filename}, skipping")
            continue
        
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Successfully loaded pickle file: {input_path}")
            
            if save_to_hdf5(data, output_path):
                successful_conversions += 1
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            print(traceback.format_exc())
    
    print(f"\nConversion process completed. Successfully converted {successful_conversions} out of {len(files)} files.")

# Use the function
input_dir = '/home/hmoderinwale/Data/RCTN/data/synthetic/N=10_M=3_E=3'
output_dir = '/home/hmoderinwale/Data/RCTN/data/hdf5data'

print("Starting conversion process...")
batch_convert_to_hdf5(input_dir, output_dir)
