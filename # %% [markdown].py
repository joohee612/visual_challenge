# %% [markdown]
# # Pymaceuticals Inc.
# ---
# 
# ### Analysis
# 
# - Add your analysis here.
#  

# %%
# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

# Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)

# Combine the data into a single DataFrame

merged_data = pd.merge(study_results, mouse_metadata, how="left", on="Mouse ID")

# Display the data table for preview
merged_data.head()

# %%
# Checking the number of mice.
num_mice = merged_data['Mouse ID'].nunique()
print(num_mice)

# %%
# Our data should be uniquely identified by Mouse ID and Timepoint
# Get the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 
duplicated_mice = merged_data[merged_data.duplicated(['Mouse ID', 'Timepoint'])]
#['Mouse ID'].unique()

duplicated_mice = duplicated_mice['Mouse ID'].unique()
duplicated_mice


# %%
# Optional: Get all the data for the duplicate mouse ID. 
duplicated_data = merged_data.loc[merged_data['Mouse ID'].isin(duplicated_mice)]

duplicated_data


# %%
# Create a clean DataFrame by dropping the duplicate mouse by its ID.
#clean_data = merged_data.drop_duplicates(subset=['Mouse ID', 'Timepoint'], keep=False)
clean_data = merged_data[merged_data['Mouse ID'].isin(duplicated_mice)==False]
clean_data.head()


# %%
# Checking the number of mice in the clean DataFrame.
num_mice = clean_data['Mouse ID'].nunique()

num_mice

# %% [markdown]
# ## Summary Statistics

# %%
# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen

# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 
# Assemble the resulting series into a single summary DataFrame.

Mean_Tumor =clean_data.groupby("Drug Regimen")["Tumor Volume (mm3)"].mean()
Median_Tumor =clean_data.groupby("Drug Regimen")["Tumor Volume (mm3)"].median()
Variance_Tumor =clean_data.groupby("Drug Regimen")["Tumor Volume (mm3)"].var()
std_Tumor =clean_data.groupby("Drug Regimen")["Tumor Volume (mm3)"].std()
sem_Tumor =clean_data.groupby("Drug Regimen")["Tumor Volume (mm3)"].sem()

Summary_statistics=pd.DataFrame()
Summary_statistics["Mean Tumor Volume"]= Mean_Tumor
Summary_statistics["Median Tumor Volume"]=Median_Tumor
Summary_statistics["Tumor Volume Variance"]= Variance_Tumor
Summary_statistics["Tumor Volume Std.Dev"]=std_Tumor
Summary_statistics["Tumor Volume Std.Err"]=sem_Tumor



Summary_statistics

# %%
# A more advanced method to generate a summary statistics table of mean, median, variance, standard deviation,
# and SEM of the tumor volume for each regimen (only one method is required in the solution)

# Using the aggregation method, produce the same summary statistics in a single line
Summary_statistics = clean_data.groupby('Drug Regimen').agg({'Tumor Volume (mm3)': ['mean', 'median', 'var','std','sem']})

Mean_Tumor
Median_Tumor
Variance_Tumor
std_Tumor
sem_Tumor




Summary_statistics

# %% [markdown]
# ## Bar and Pie Charts

# %%
# Generate a bar plot showing the total number of rows (Mouse ID/Timepoints) for each drug regimen using Pandas.
import pandas as pd
import matplotlib.pyplot as plt

regimen_counts = clean_data["Drug Regimen"].value_counts()
regimen_counts.plot(kind="bar", align="center")

plt.xlabel('Drug Regimen')
plt.ylabel('# of Observed Mouse Timepoints')

plt.tight_layout()
plt.savefig("Mouse ID.png")
plt.show()

# %%
# Generate a bar plot showing the total number of rows (Mouse ID/Timepoints) for each drug regimen using pyplot.
import pandas as pd
import matplotlib.pyplot as plt

regimen_counts = clean_data["Drug Regimen"].value_counts()
plt.bar(regimen_counts.index, regimen_counts.values)

plt.xlabel("Drug Regimen")
plt.ylabel("# of Observed Mouse Timepoints")
plt.xticks(rotation=90)
plt.show()

# %%
# Generate a pie plot showing the distribution of female versus male mice using Pandas
import pandas as pd
import matplotlib.pyplot as plt

gender_counts = clean_data["Sex"].value_counts()
gender_counts.plot(kind="pie", y="Sex", autopct="%1.1f%%")

plt.show()

# %%
# Generate a pie plot showing the distribution of female versus male mice using pyplot
import pandas as pd
import matplotlib.pyplot as plt

gender_counts = clean_data["Sex"].value_counts()

plt.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%")
plt.ylabel("Sex")
plt.show()

# %% [markdown]
# ## Quartiles, Outliers and Boxplots

# %%
# Calculate the final tumor volume of each mouse across four of the treatment regimens:  
# Capomulin, Ramicane, Infubinol, and Ceftamin
drug_regimens = ["Capomulin", "Ramicane", "Infubinol", "Ceftamin"]
four_regimens_data = clean_data[clean_data["Drug Regimen"].isin(drug_regimens)]
# Start by getting the last (greatest) timepoint for each mouse
last_timepoints = four_regimens_data.groupby("Mouse ID")["Timepoint"].max()

# Merge this group df with the original DataFrame to get the tumor volume at the last timepoint
merged_data = pd.merge(clean_data, last_timepoints,how=("right"),on=["Mouse ID", "Timepoint"])


# %%
# Put treatments into a list for for loop (and later for plot labels)
treatments = ["Capomulin", "Ramicane", "Infubinol", "Ceftamin"]

# Create empty list to fill with tumor vol data (for plotting)

tumor_vol_data = []
# Calculate the IQR and quantitatively determine if there are any potential outliers. 
for drug in treatments: 


    
    # Locate the rows which contain mice on each drug and get the tumor volumes
    tumor_vol = merged_data.loc[merged_data["Drug Regimen"] == drug, 'Tumor Volume (mm3)']
    
    # add subset 
    tumor_vol_data.append(tumor_vol)
    

    
    # Determine outliers using upper and lower bounds
    quartiles = tumor_vol.quantile([.25,.5,.75])
    lowerq = quartiles[0.25]
    upperq = quartiles[0.75]
    iqr = upperq-lowerq
    lower_bound = lowerq - (1.5*iqr)
    upper_bound = upperq + (1.5*iqr)
    outliers = tumor_vol.loc[(tumor_vol < lower_bound) | (tumor_vol > upper_bound)]
    print(f"{drug}'s potential outliers: {outliers}")
    

# %%
# Generate a box plot that shows the distrubution of the tumor volume for each treatment group.
orange_out = dict(markerfacecolor='red',markersize=12)
plt.boxplot(tumor_vol_data, labels = treatments,flierprops=orange_out)
plt.ylabel('Final Tumor Volume (mm3)')
plt.show()

# %% [markdown]
# ## Line and Scatter Plots

# %%
# Generate a line plot of tumor volume vs. time point for a single mouse treated with Capomulin
capomulin_table = clean_data.loc[clean_data['Drug Regimen'] == "Capomulin"]
mousedata = capomulin_table.loc[capomulin_table['Mouse ID']== 'l509']
plt.plot(mousedata['Timepoint'],mousedata['Tumor Volume (mm3)'])
plt.xlabel('Timepoint (days)')
plt.ylabel('Tumor Volume (mm3)')
plt.title('Capomulin treatment of mouse l509')
plt.show()

# %%
# Generate a scatter plot of mouse weight vs. the average observed tumor volume for the entire Capomulin regimen
capomulin_table = clean_data.loc[clean_data['Drug Regimen'] == "Capomulin"]
capomulin_average = capomulin_table.groupby(['Mouse ID'])[['Weight (g)', 'Tumor Volume (mm3)']].mean()
plt.scatter(capomulin_average['Weight (g)'],capomulin_average['Tumor Volume (mm3)'])
plt.xlabel('Weight (g)')
plt.ylabel('Average Tumor Volume (mm3)')
plt.show()

# %% [markdown]
# ## Correlation and Regression

# %%
# Calculate the correlation coefficient and a linear regression model 
# for mouse weight and average observed tumor volume for the entire Capomulin regimen
corr=round(st.pearsonr(capomulin_average['Weight (g)'],capomulin_average['Tumor Volume (mm3)'])[0],2)
print(f"The correlation between mouse weight and the average tumor volume is {corr}")
model = st.linregress(capomulin_average['Weight (g)'],capomulin_average['Tumor Volume (mm3)'])

y_values = capomulin_average['Weight (g)']*model[0]+model[1]
plt.scatter(capomulin_average['Weight (g)'],capomulin_average['Tumor Volume (mm3)'])
plt.plot(capomulin_average['Weight (g)'],y_values,color="red")
plt.xlabel('Weight (g)')
plt.ylabel('Average Tumor Volume (mm3)')
plt.show()

# %%



