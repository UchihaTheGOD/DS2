 #Read data from CSV

import pandas as pd
df = pd.read_csv('Student_Marks.csv')
print("Our dataset ")
print(df)

# read data from json
import pandas as pd
data = pd.read_json('dataset.json')
print(data) 
