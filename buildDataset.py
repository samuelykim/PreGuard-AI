import pandas as pd #imports pandas which is used to work with CSV/data tables
import matplotlib.pyplot as plt #matplotlibs for plotting tools for graphs
import glob #allaws us to search for files 
import os #helps me work with files easier

#this means the same folder the script is sitting in since i have cateforys folders (bend fall etc)
dataRoot = "." 
categories = ["fall", "walk", "sit", "stand", "bend"] #creates a loist containing all activity folder names program will look at

#load every recording instantly (store the data from every CSV recording )
allData = []

for category in categories: #goes through every category in the categories list one at a time (fall then walk etc)
    pattern = os.path.join(dataRoot, category, "*.csv") #mathces any csv file inside category folder (os.path.join combines folder location, name, and csv basically any file that ends in csv)
    matchingFiles = glob.glob(pattern) #search for every csv file that matches combination above (results in a list containg everything found)
    print(f"Found {len(matchingFiles)} recordings for '{category}'") #prints how many csv recording founf 
    
    for filePath in matchingFiles: #goes through every CSV file found for activity 
        dataFile = pd.read_csv(filePath) #reads csv file into pandas dataframe
        dataFile["label"] = category #add new column called  label (every row get this name such as fall walk)
        dataFile["sourceFile"] = filePath #anothyer column which records which csv file each row originally came from
        allData.append(dataFile) #adds datafram containg this recording to the alldata list

#check if alldatalist is empty
if not allData:
    print("\nNo recordings found. Check that your 'data' folder structure "
          "matches what this script expects (see the coment at the top).")
else:
    #combines all individual dataframes in all data into one large dataframe
    combinedDataFile = pd.concat(allData, ignore_index=True)
    print(f"\nTotal rows across ALL recordings: {len(combinedDataFile)}") #print total numbers
    print("\nHow many rows per activity: ") #print heading before displaying number of rows
    print(combinedDataFile["label"].value_counts()) #count how many times an activity appears telling us how much data we have
    
    #creat matplotlib figre and size
    plt.figure(figsize=(10, 5))
    for category in categories: #goes through each activity  one at a time
        subset = combinedDataFile[combinedDataFile["label"] == category] #create smaller dataframe containing only rows (shows fall only and keeps it in fall row)
        if subset.empty: 
            continue
        
        for i, filePath in enumerate(subset["sourceFile"].unique()): #gets every unique CSV file that belongs to this activity.  # enumerate() gives us both: # i = the number/index of the file
            example = subset[subset["sourceFile"] == filePath] #creates another smaller DataFrame containing only the data
            plt.plot( #plots the x-axis acceleration data from this recording.
                example["x"].values, #gets the values from the "x" column. # .values converts the Pandas column into an array of values that Matplotlib can plot.
                label=category if i == 0 else None, #gives the graph a label for the activity. The label is only added to the first file for each category. This prevents the legend from having the same activity listed over and over again.
                alpha=0.6, #makes the line slightly transparent.
                )
        
    plt.legend() #displays the legend on the graph.
    plt.title("One example recording per activity") #gives the graph a title.
    plt.xlabel("Time step") #gives the horizontal axis a label.
    plt.ylabel("Acceleration, x-axis (check units from app)") #gives the vertical axis a label.
    plt.show() #displays the graph in a window.