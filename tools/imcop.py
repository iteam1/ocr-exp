'''
Copy images from remote to local machine

- Step1: Run script copy images (remote side): python3 imcop.py
- Step2: Compress destination: tar -zcvf dst.tar.gz dst
- Step3: remove destination folder sudo rm -rf dst
- Step3: Download compressed file (local side): scp greystone@172.16.21.0:/media/DATA_LAPTOP_BOOTH/1.LaptopBooth/4.History/6.History_Fedex_V2/dst.tar.gz /home/greystone/locchuong

Author: Loc Chuong
'''
import os
import shutil

dst = 'dst' # name of destination folder
if not os.path.exists(dst):
   os.mkdir(dst)

# list all files and folders
items = os.listdir(".")
print("Total (file or folder): ",len(items))

# list all dates
dates = []
for item in items:
   if len(item)==10 and item.count("_") == 2:
      dates.append(item)
      
# check valid date by month
valid_dates=[]
for date in dates:
   d,m,y = date.split('_')
   if y == "2023":
      valid_dates.append(date)
   elif y == "2022":
      if int(m) >=8:
         valid_dates.append(date)
   else:
      pass
      
print(f"Valid: {len(valid_dates)}/{len(dates)}")

# make floders in destiation
for date in valid_dates:
   # make date
   if not os.path.exists(os.path.join(dst,date)):
      os.mkdir(os.path.join(dst,date))
   # check trans
   trans = os.listdir(date)
   for tran in trans:
      if len(tran) == 14:
         # make tran
         if not os.path.exists(os.path.join(dst,date,tran)):
            os.mkdir(os.path.join(dst,date,tran))
         # copy img
         print(tran," copying")
         shutil.copy2(os.path.join(date,tran)+"/pictures/image_detect_profile.jpg",os.path.join(dst,date,tran))
         shutil.copy2(os.path.join(date,tran)+"/pictures/image_detect_postion.jpg",os.path.join(dst,date,tran))
         shutil.copy2(os.path.join(date,tran)+"/pictures/image_detect_profile_top.jpg",os.path.join(dst,date,tran))
  
