import pandas as pd
import matplotlib.pyplot as plt
from diffprivlib.mechanisms import ExponentialHierarchical, Geometric

# differential privacy mechanisms
def Geometric_Mechanism(x, eps):
    geometric = Geometric(epsilon=eps, sensitivity=1)
    x += geometric.randomise(0)
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

# importing datasets
original_votes_df = pd.read_csv('dataset.txt', sep=',').sample(frac=1).reset_index(drop=True)
privatized_votes_df = original_votes_df.copy()
cities = pd.read_csv('data/ZipCodesGR.csv', sep=',', dtype=str).values.tolist()

# anonymize the data - apply Geometric mechanism to age column
privatized_votes_df['Age'] = privatized_votes_df['Age'].apply(Geometric_Mechanism, args=(0.5,))

# anonymize the data - apply Exponential Hierarchical Mechanism to Location data
# privatized_votes_df['Location'] = privatized_votes_df['Location'].apply(Exponential_Mechanism, args=(2,cities))

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
plt.show()

# output the original and anonymized datasets
print(original_votes_df, privatized_votes_df)

# compute sensitivity between datasets
sensitivity = pd.DataFrame(abs(privatized_votes_df["Age"].value_counts() - original_votes_df["Age"].value_counts())).fillna(0)
print ("The sensitivity is:", sensitivity.Age.max())
