import pandas as pd 
import os
import numpy as np

parentdir = "/p/rosbot/rosbotxl/data-meriel/cleaned/"
dirs = os.listdir(parentdir)


# Index(['timestamp', 'image_name', 'linear_speed_x', 'angular_speed_z',
#        'is_turning', 'is_manually_off_course', 'lidar_ranges'], dtype='object')
total_samples = 0
for d in dirs:
    csv_file = f"{parentdir}{d}/data.csv"
    df = pd.read_csv(csv_file) 
    # print(df.columns)
    # print(df.shape, df.size)
    # exit(0)
    drop_indices = df[(df.linear_speed_x == 0) & (df.angular_speed_z == 0)].index.values
    
    pics = os.listdir(f"{parentdir}{d}")
    pics = [p for p in pics if ".jpg" in p]
    for index, row in df.iterrows():
        if row.image_name not in pics:
            drop_indices = np.append(drop_indices, index)
    drop_indices = set(drop_indices.flatten())

    init_df_size = df.shape[0]
    drop_images = [df.iloc[i].image_name for i in drop_indices]
    df = df.drop(drop_indices)

    for di in drop_images:
        if os.path.exists(f"{parentdir}{d}/{di}"):
            os.remove(f"{parentdir}{d}/{di}")
    print(f"Finished cleaning {parentdir}{d}")
    print(f"Removed {init_df_size - df.shape[0]} samples\nRemaining samples: {df.shape[0]}")
    total_samples += df.shape[0]
    df.to_csv(csv_file, encoding='utf-8')
print(f"Total remaining samples: {total_samples}")