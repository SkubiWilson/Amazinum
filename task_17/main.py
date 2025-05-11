import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('Groceries_dataset.csv')

transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list)


te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_trans, min_support=0.005, use_colnames=True)

print(f"Знайдено {len(frequent_itemsets)} частих наборів")

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

print(f"Знайдено {len(rules)} правил")
top_10_rules = rules.sort_values('lift', ascending=False).head(10)
print(top_10_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

