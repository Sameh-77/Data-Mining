import re
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules


sequences = open('Apple_sequence_dataset.txt', 'r') # reading the .txt file 

#Reading all lines from our file
lines = sequences.readlines() # reading the lines 

#-----------------------------Preparing the data----------------------------------------------#

candidate_items = ""
transactions = []

# In this for loop, i check each line, and I remove all the extra characters and numbers from it 
# Then I put the characters (segments) for each person in a list after omitting duplicate segments, 
# so that I have a list of transcations for each person  
for line in lines: 
    line = line.strip()
    line = re.sub("[\[\],'0-9 ]", "", line) # removing extra characters and numbers 
    
    line = "".join(set(line)) #Omitting duplicate letters from the transaction
    candidate_items = candidate_items + line # items is used to find the first candidate set of items 
    
    transactions.append(list(line)) 

# omitting the duplicates so that I have the list of all candidate items 
candidate_items = "".join(set(candidate_items)) 

print("\n==============================\n")
print("The transaction are: \n")
print(transactions)

print("\n==============================")
print("First set of candidate items: \n") 
print(list(candidate_items))

#----------------------------------------Apriori Algorithm ----------------------------------------#

# I'm using apriori function from the mlxtend.frequent_patterns

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets=apriori(df, min_support=0.4, use_colnames=True) #calling the apriori function 

# preparing the format of displaying 
frequent_itemsets['length(k)'] = frequent_itemsets['itemsets'].apply(lambda x: len(x)) 

print("\n=====================================")
print("Frequent items sets:")
print("=====================================\n")
print(frequent_itemsets)

# generating the association rules 
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6) 
rules = rules[['antecedents', 'consequents', 'support','confidence']]
print("\n=====================================")
print("Association Rules: ")
print("=====================================\n")
print(rules)
print("\n=====================================")


