import pandas as pd
import glob

# extract the angular speed from each CSV file in the file path
# Note: this will read the WRONG file if you do not extract only the cleaned csv files.

def extract_angular_speed_z(file_paths):
    angular_speed_z_values = []

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            angular_speed_z_values.extend(df['angular_speed_z'].values)
        except KeyError:
            print(f"Column 'angular_speed_z' not found in {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to native Python floats to avoid np.float64 notation
    angular_speed_z_values = [float(val) for val in angular_speed_z_values]

    return angular_speed_z_values

def main():
    current_data_folder = 'newdata-2024-06-27-1442' # Change this folder path to the parent directory of your clean csv files
    folder_path = f'C:/Users/user/Desktop/rosbot/{current_data_folder}/' # Change this directory to match your computer
    file_paths = glob.glob(folder_path + '*.csv')

    angular_speed_z_array = extract_angular_speed_z(file_paths)

    # make a new txt file with the 1D array
    with open(f"{current_data_folder}.txt",'w') as f:
        f.write(str(angular_speed_z_array))

    print(f'Angular speeds written to {current_data_folder}.txt')

    #print(angular_speed_z_array) # Uncomment if you want to print in terminal instead of write to file

if __name__ == '__main__':
    main()

## After you run this script, use the MATLAB file make_histogram.m to make the histogram
## Paste the 1D Array into the top line