# ROSbot Data Cleaning
- Once you're done collecting the data, you will need to clean it to remove unwanted parts. It is better to do this on a laptop or on portal.
  - For example, the robot is not moving, and you don't want it to learn to stand still.
  - Or a random guy might walk in front, but you may or may not want to clean all examples of this as this could happen during deployment.
- To do this, you can create your own script or use the one provided (recommended).
  
## Retrieve data from the bot to your laptop or portal
  - You will need to scp (secure copy) the data from the rosbot into your system.
  - Identify the parent directory where your data is stored on the rosbot.
    - example: `/home/husarion/media/usb/rosbotxl_data`
  - On your machine (or portal), intiate copying the files:
    - `scp -r husarion@<your-bots-ip-address>:/home/husarion/<wherever-you-put-the-data-parent-directory> <your-local-machine-directory>`
    - Replace ip address with your rosbot's ip address and the directories with your directories.
    - Example: `scp -r husarion@172.27.179.144:/home/husarion/media/usb/rosbotxl_data C:/Users/user123/Desktop/data_to_clean`
    - you should now have the original raw data in your local directory, ready to clean. 
## Get and Run the script
- For starting, I recommend that you make a copy of your data somewhere else and attempt to clean that.
- Keep the data that got rejected, you might want it later. 

- In your laptop:
  - Download `clean_rosbot_data.py` from the github: ROSbot_data_collection/data_cleaning. Since it is just a script, you can just put it in your Downloads folder. You can git pull or simply touch a new file and copy the source code.
- In portal:
  - Git clone `ROSbot_data_collection` to your portal home directory: `git clone -b rosbotXL https://github.com/MissMeriel/ROSbot_data_collection.git`
  - change directory to find the python file: `cd ROSbot_data_collection/data_cleaning`
- Retrieve the directory that all your datacollection folders are in and copy it, you will need to enter that into the command. This directory should have the all the datacollection folders you want to clean.
   - windows example: `C:/Users/user123/Desktop/data_to_clean`. 
    - portal example: `/u/aqq4ax/ROSbot_data_collection/datasets`
- Move to wherever you put `clean_rosbot_data.py` and run the script in your terminal:
```
python3 clean_rosbot_data.py --parentdir <insert-your-directory-here>
```
- Replace `<insert-your-directory-here>` with your directory:
   - example: `python3 clean_rosbot_data.py --parentdir C:/Users/user123/Desktop/data_to_clean`
- You can comment out the specific cleaning tasks at the bottom (if name ==main) if you don't want to run them.


