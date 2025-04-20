import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = {
    'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Region': ['North', 'North', 'North', 'South', 'South', 'South', 'East', 'East', 'East', 'West', 'West', 'West'],
    'Month': ['Jan', 'Jan', 'Jan', 'Feb', 'Feb', 'Feb', 'Mar', 'Mar', 'Mar', 'Apr', 'Apr', 'Apr'],
    'Sales': [120, 150, 100, 200, 180, 130, 250, 220, 140, 300, 260, 180],
}

df = pd.DataFrame(data)
print(df)


plt.figure(figsize=(8, 5))
sns.barplot(x=df.groupby('Product')['Sales'].sum().index,
y=df.groupby('Product')['Sales'].sum().values)
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.show()


monthly_sales = df.groupby('Month')['Sales'].sum().reindex(['Jan', 'Feb', 'Mar', 'Apr'])
plt.figure(figsize=(8, 5))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

heatmap_data = df.pivot_table(index='Region', columns='Product', values='Sales',
aggfunc='sum')
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt='.0f')
plt.title('Sales by Region and Product')
plt.show()
fig = px.bar(df, x='Region', y='Sales', color='Product', title='Sales by Region and Product')
fig.show()
