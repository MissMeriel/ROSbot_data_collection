# ROSbot Data Cleaning
- Once you're done collecting the data, you will need to clean it to remove unwanted parts. 
  - For example, the robot is not moving, and you don't want it to learn to stand still.
  - Or a random guy might walk in front, but you may or may not want to clean all examples of this as this could happen during deployment.
- To do this, you can create your own script (or use the one provided). 

## Get and Run the script
- For starting, I recommend that you make a copy of your data somewhere else and attempt to clean that.
- Keep the data that got rejected, you might want it later. 


- Download `clean_rosbot_data.py` from the github: ROSbot_data_collection/data_cleaning. Since it is just a script, you can just put it in your Downloads folder. You can git pull or simply touch a new file and copy the source code.
- Retrieve the directory that all your datacollection folders are in and copy it, you will need to enter that into the command. This directory should have the all the datacollection folders you want to clean.
 - For example: `/home/husarion/Desktop/data_to_clean`. 
- Move to your downloads folder (or wherever you put `clean_rosbot_data.py`) and run the script:
```
cd Downloads
python3 clean_rosbot_data.py --parentdir /home/husarion/<insert-your-directory-here>
```
- Replace `<insert-your-directory-here>` with your directory:
   - example: `python3 clean_rosbot_data.py --parentdir /home/husarion/Desktop/data_to_clean`
- You can comment out the specific cleaning tasks at the bottom (if name ==main) if you don't want to run them.


