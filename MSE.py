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

# importing datasets
original_votes_df = pd.read_csv('dataset.txt', sep=',').sample(frac=1).reset_index(drop=True)
privatized_votes_df = original_votes_df.copy()

# pre-define epsilons
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10, 100]
MSE_data = []

# calculate MSE for each epsilon
for epsilon in epsilons:    
    original_df = pd.Series(original_votes_df['Age'])
    new_df = pd.Series(privatized_votes_df['Age'].apply(Geometric_Mechanism, args=(epsilon,)))
    difference = original_df.add(-1*new_df, fill_value=0)
    MSE = difference.apply(lambda x: x**2).mean()
    MSE_data.append((epsilon, MSE))

# convert to Pandas Series and Plot
MSE_data = pd.DataFrame(MSE_data, columns=['Epsilon','MSE']).set_index('Epsilon')
MSE_data.plot.bar()
print(MSE_data)
plt.show()
