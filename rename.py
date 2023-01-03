import os

dir = input("Enter your directory: ")
prefix = input("Enter your prefix: ")
 
files = os.listdir(dir)

for i,f in enumerate(files):
    if ".jpg" in f:
        new_name = prefix+"_"+str(i)+".jpg"
        os.rename(os.path.join(dir,f),os.path.join(dir,new_name))
        print("Rename " + f + " to " + new_name)
    else:
    	pass
