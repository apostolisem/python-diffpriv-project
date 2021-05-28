import pandas as pd
import matplotlib.pyplot as plt
from diffprivlib.mechanisms import ExponentialHierarchical, Geometric

# differential privacy mechanisms
def Geometric_Mechanism(x, eps):
    geometric = Geometric(epsilon=eps, sensitivity=1)
    x += geometric.randomise(0)
    # keep the privatized dataset withing 18-80 years as the original dataset
    if (x < 18):
        diff = 18 - x
        x += 2*diff
    elif (x > 80):
        diff = x - 80
        x -= 2*diff
    return x

def Exponential_Mechanism(x, eps, hierarchy):   
    exponential = ExponentialHierarchical(epsilon=eps, hierarchy=hierarchy)
    x = exponential.randomise(x)    
    return x

# function to get the MSE for a privatized dataset
def get_MSE(original_df, new_df, eps):
    original_df = pd.Series(original_votes_df['Age'])
    new_df = pd.Series(new_df['Age'].apply(Geometric_Mechanism, args=(eps,)))
    difference = original_df.add(-1*new_df, fill_value=0)
    MSE = difference.apply(lambda x: x**2).mean()
    return MSE

# importing datasets
original_votes_df = pd.read_csv('dataset.txt', sep=',').sample(frac=1).reset_index(drop=True)
privatized_votes_df = original_votes_df.copy()
cities = pd.read_csv('data/ZipCodesGR.csv', sep=',', dtype=str).values.tolist()

# anonymize the data - apply Geometric mechanism to age column
epsilon = 0.5
privatized_votes_df['Age'] = privatized_votes_df['Age'].apply(Geometric_Mechanism, args=(epsilon,))

# anonymize the data - apply Exponential Hierarchical Mechanism to Location data
privatized_votes_df['Location'] = privatized_votes_df['Location'].apply(Exponential_Mechanism, args=(epsilon,cities))

# anonymize the data - remove name and mobile columns
privatized_votes_df = privatized_votes_df.drop(columns=['Name', 'Mobile'])

# generate sample plots - votes by country
votes_by_country = original_votes_df['VoteCountry'].value_counts().head(10)
votes_by_country.plot.bar()

# generate sample plots - votes by age
votes_by_age = original_votes_df[['Age','VoteCountry']]
votes_by_age.plot.hist()

# generate sample plots - votes by age from anonymized data
votes_by_age_priv = privatized_votes_df[['Age','VoteCountry']]
votes_by_age_priv.plot.hist()

# count values and difference
original_values = original_votes_df["Age"].value_counts()
privatized_values = privatized_votes_df["Age"].value_counts()
difference = original_values.add(-1*privatized_values, fill_value=0)

# compute MSE for privatized dataset
MSE = get_MSE(original_votes_df, privatized_votes_df, epsilon)

# compute sensitivity between neighboring datasets
sensitivity = pd.DataFrame(abs(difference)).max()

# output our results
print(original_votes_df, privatized_votes_df)
print("The sensitivity is:", sensitivity.Age)
print("The MSE is:", MSE) # lower the better
plt.show()
