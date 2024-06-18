# ROSbot Data Cleaning
- Once you're done collecting the data, you will need to clean it to remove unwanted parts. 
- For example, the robot is not moving, or some random guy walks in front of it. 
- To do this, you can create your own script (or use the one provided). 

## Locate your data directory
- You will need to find the directory you have stored your data in. 
- For starting, I recommend that you make a copy of your data somewhere else and attempt to clean that. 
- Retrieve the directory that all your datacollection folders are in and copy it, you will need to enter that into the command later. 
 ## Run the script
- Download the python file, it will end up in your Downloads folder. 
- Move to your downloads folder and run the script:
```
cd Downloads
python3 clean_rosbot_data.py --parentdir "/home/husarion/<insert-your-directory-here>"
```
- Include the quotes, and add your directory. This directory should have the all the datacollection folders you want to clean.
- You can comment out the specific cleaning tasks at the bottom if you don't want to run them.
- Once its done, keep the data that got rejected, you might want it later. 

