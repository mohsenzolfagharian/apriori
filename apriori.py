#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1

#read CSV file
data = pd.read_csv('bread basket.csv', sep=',')

# print(data.Item.unique())

#group data by Transaction nad Item and count Item
grouped_df = data.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
#create a private spread sheet with new columns
basket_df = grouped_df.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
#using applymap to map data on hot_encode
basket_encoded = basket_df.applymap(hot_encode)
basket_df = basket_encoded

print(basket_df)
#calc apriori with at least 1 persent apriori
frq_items = apriori(basket_df, min_support=0.01, use_colnames=True)
print(frq_items)


