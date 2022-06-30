# Import needed Python libraries and functions
import numpy as np
import pandas as pd
import re

df = pd.read_csv("Data/laptops.csv", header=[0], sep=",", decimal=".", encoding = "ISO-8859-1",thousands=',')
cpu = pd.read_csv("Data/cpu_benchmarks.csv",  header=[0], sep=",", decimal=".",thousands=',')
gpu = pd.read_csv("Data/GPU_benchmarks_v5.csv", header=[0], sep=",", decimal=".",thousands=',')

# Pre-Processing

def setGB (value):
  """Function for converting memory feature from string to numeric value in GB

    Args:
        value (string): The string containing the memory value

    Returns: 
        numeric: Memory value in GB
  """
  if isinstance(value, str):
    if ('GB' in value):
      return int(float(re.search('(.*)GB',value).group(1)))
    elif ('TB' in value):
      return int(float(re.search('(.*)TB',value).group(1))) * 1000
    
    
    
def setHarddrive (value):
  """Function for setting the dummy variables determining if memory is HDD or SSD
        
       Args: 
           value (string): The string containing the memory value

    Returns:
        list: A list containing two values, the first value represents HDD the second SSD'
  """
  if isinstance(value, str):
    if ('HDD' in value):
      return [1,0]
    elif ('SSD' in value) | ('Flash' in value):
      return [0,1]
  else: return [None,None]
  

# Key for identifying laptop, price will be sorted ascending and if duplicates occur, cheaper one is considered
df['Key'] = df['Company'] + ' ' + df['Product'] + ' ' + df['Inches'].astype(str) + ' ' +  df['ScreenResolution'] + ' ' + df['Cpu'] + ' ' + df['Ram'] + ' ' + df['Memory'] + ' ' + df['Gpu'] + ' ' + df['OpSys']

# Converting Ram to GB
df['Ram'] = df.Ram.str.extract(r'(.*)GB').astype('Int64')

# Splitting memory column into two, for two hard rives
df[['Memory1','Memory2']] = df.Memory.str.split("+", expand=True)

# Setting dummy variables with SSD and HDD memory
df[['Memory1HDD', 'Memory1SSD']] = df.apply(lambda x:  setHarddrive(x['Memory1']), axis=1,result_type='expand')
df[['Memory2HDD','Memory2SSD']] = df.apply(lambda x:  setHarddrive(x['Memory2']), axis=1,result_type='expand')

# Converting memory size to numeric value in GB
df['Memory1'] = df['Memory1'].apply(lambda x:  setGB(x))
df['Memory2'] = df['Memory2'].apply(lambda x:  setGB(x))

# Converting weight to KG
df['Weight'] = df.Weight.str.extract(r'(.*)kg').astype(float)

# Converting display size to numeric value in Inches
df['Inches'] = df.Inches.astype(float)

# Extracting Cpu in such a form that it is matchable, deleting the clock frequency
df['Cpu'] = df.Cpu.str.extract(r'(.*)\s\d.\dGHz')

# Converting Screen Resolution to pixel density
df['Display'] = df["ScreenResolution"]
df['ScreenResolution'] = df.ScreenResolution.str.extract(r'(\d*)x(\d*)').apply(lambda x: int(x[0]) * int(x[1]), axis=1)

# Deleting whitespaces on Gpu for match
df.Gpu = df.Gpu.map(lambda x: x.replace(' Graphics','').strip())

df.set_index("Key", drop=True)


# Only considering intel cpus
cpu = cpu.rename(columns={"rating":"ProcessorRating"})

# Extracting processor name from url, which is identified by = and % as boundaries
cpu['Processor'] = cpu.url.str.extract(r'=(.*?)&')

# Formatting processor name
cpu['Processor'] = [str(x).replace('+',' ').strip().replace("-"," ") for x in cpu['Processor']]
cpu['Processor'] = cpu['Processor'].str.replace("(%).*","")
cpu["Processor"] = cpu["Processor"].str.strip() 

# Selecting relevant features from dataset cpu 
cpu = cpu[['ProcessorRating','Processor']].copy()

# Formatting CPU column for merge
df["Cpu"] = df["Cpu"].str.strip()


df = df.merge(cpu, how='left', left_on="Cpu" ,right_on="Processor")



# Filtering out desktop gpus since we only consider laptops
gpu = gpu.loc[gpu.category != 'Desktop'].copy()
# Reshaping gpuName for mapping
gpu["gpuName"] = gpu.gpuName.map(lambda x: x.replace(' Graphics','').replace('GeForce','Nvidia GeForce').replace('Radeon','AMD Radeon').replace(' (Mobile)','').strip())

gpu.rename({'G3Dmark' : 'GpuScore1','G2Dmark': 'GpuScore2' }, axis=1)

# Formatting GPU column for merge
df["Gpu"] = df["Gpu"].str.strip()
df = df.merge(gpu, how='left', left_on="Gpu" ,right_on="gpuName")


# Creating new dataframe with selected columns and no emtpty values. 
df = df.sort_values(by="Price_euros", ascending=True).copy()
df = df.drop_duplicates(subset=['Key'], keep='first').copy()

# Filling na for created columns indicating type of memory
df[['Memory2', 'Memory1HDD', 'Memory1SSD',
       'Memory2HDD', 'Memory2SSD']] = df[['Memory2', 'Memory1HDD', 'Memory1SSD',
       'Memory2HDD', 'Memory2SSD']].fillna(0)

# Dropping na values
df = df[['Company', 'Product', 'TypeName','Display',
       'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys','Key','Inches',
       'Ram','Weight', 'Price_euros', 'ProcessorRating','G3Dmark','G2Dmark', 'Memory1', 'Memory2',
       'Memory1HDD', 'Memory1SSD', 'Memory2HDD', 'Memory2SSD','testDate']].dropna()

# Resetting index to key column
df.set_index('Key', inplace=True)


