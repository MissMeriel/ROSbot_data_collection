import pandas as pd 
import os
import numpy as np
from pathlib import Path

parentdir = "/p/sdbb/ROSbot_data_collection/dataset/data-yili/"
dirs = os.listdir(parentdir)

# remove rows for which pictures aren't in directory
def remove_extra_rows():
    # Index(['timestamp', 'image_name', 'linear_speed_x', 'angular_speed_z',
    #        'is_turning', 'is_manually_off_course', 'lidar_ranges'], dtype='object')
    total_samples = 0
    for d in Path.iterdir(Path(parentdir)): #dirs:
        if os.path.isdir(d):
            try:
                csv_file = f"{d}/data.csv"
                df = pd.read_csv(csv_file) 
            except:
                csv_file = f"{d}/data_cleaned.csv"
                df = pd.read_csv(csv_file) 
            print(d, "\n", df.columns)
            # print(df.shape, df.size)
            # exit(0)
            drop_indices = df[(df.linear_speed_x == 0) & (df.angular_speed_z == 0)].index.values
            
            pics = os.listdir(f"{d}")
            pics = [p for p in pics if ".jpg" in p]
            for index, row in df.iterrows():
                try:
                    if row["image_name"] not in pics:
                        drop_indices = np.append(drop_indices, index)
                except:
                    if row["image name"] not in pics:
                        drop_indices = np.append(drop_indices, index)
            drop_indices = set(drop_indices.flatten())

            init_df_size = df.shape[0]
            try:
                drop_images = [df.iloc[i]["image_name"] for i in drop_indices]
            except:
                drop_images = [df.iloc[i]["image name"] for i in drop_indices]
            df = df.drop(drop_indices)

            for di in drop_images:
                if os.path.exists(f"{d}/{di}"):
                    os.remove(f"{d}/{di}")
            print(f"Finished cleaning {d}")
            print(f"Removed {init_df_size - df.shape[0]} samples\nRemaining samples: {df.shape[0]}")
            total_samples += df.shape[0]
            df.rename(columns={'image name': 'image_name'}, inplace=True)
            df.to_csv(csv_file, encoding='utf-8')
    print(f"Total remaining samples: {total_samples}")


# remove row if picture does not exist
def remove_extra_pics():
        # Index(['timestamp', 'image_name', 'linear_speed_x', 'angular_speed_z',
    #        'is_turning', 'is_manually_off_course', 'lidar_ranges'], dtype='object')
    total_samples = 0
    remaining_samples = 0
    for d in Path.iterdir(Path(parentdir)): #dirs:
        if os.path.isdir(d): 
            try:
                csv_file = f"{d}/data.csv"
                df = pd.read_csv(csv_file) 
            except:
                csv_file = f"{d}/data_cleaned.csv"
                df = pd.read_csv(csv_file) 
            # print(d, "\n", df.columns)
                   
            pics = os.listdir(f"{d}")
            pics = [p for p in pics if ".jpg" in p]
            init_pic_count = len(pics)
            total_samples += init_pic_count
            for p in pics:
                row = df.loc[df['image_name'] == p]
                if row.empty:
                    print("\tRemoving", d, p)
                    os.remove(f"{d}/{p}")

                # for di in drop_images:
                #     if os.path.exists(f"{d}/{di}"):
                #         os.remove(f"{d}/{di}")
            pics = os.listdir(f"{d}")
            remaining_samples += len([p for p in pics if ".jpg" in p])
            print(f"Finished cleaning {d}")
                # print(f"Removed {init_df_size - df.shape[0]} samples\nRemaining samples: {df.shape[0]}")
                # total_samples += df.shape[0]
                # df.rename(columns={'image name': 'image_name'}, inplace=True)
                # df.to_csv(csv_file, encoding='utf-8')
    print(f"Starting samples: {total_samples}")
    print(f"Total remaining samples: {remaining_samples}")

if __name__ == '__main__':
    remove_extra_pics()