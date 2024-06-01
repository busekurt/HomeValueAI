import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Initialize LabelEncoders for categorical variables
label_encoder_brokered_by = LabelEncoder()
label_encoder_street = LabelEncoder()
label_encoder_city = LabelEncoder()
label_encoder_state = LabelEncoder()

# Transform categorical variables
data['brokered_by'] = label_encoder_brokered_by.fit_transform(data['brokered_by'])
data['street'] = label_encoder_street.fit_transform(data['street'])
data['city'] = label_encoder_city.fit_transform(data['city'])
data['state'] = label_encoder_state.fit_transform(data['state'])
data['status'] = data['status'].map({'for_sale': 0, 'ready_to_build': 1})

# Fill missing values
data = data.fillna(0)

# Distribution of property prices
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True, bins=30)
plt.title('Distribution of Property Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Count of properties by city
plt.figure(figsize=(12, 8))
city_counts = data['city'].value_counts().head(20)  # Top 20 cities for readability
sns.barplot(x=city_counts.index, y=city_counts.values, palette='viridis')
plt.xticks(rotation=90)
plt.title('Count of Properties by City (Top 20)')
plt.xlabel('City')
plt.ylabel('Count')
plt.show()

# Relationship between property features and price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['bed'], y=data['price'], alpha=0.5)
plt.title('Relationship Between Number of Beds and Price')
plt.xlabel('Number of Beds')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['bath'], y=data['price'], alpha=0.5)
plt.title('Relationship Between Number of Baths and Price')
plt.xlabel('Number of Baths')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['house_size'], y=data['price'], alpha=0.5)
plt.title('Relationship Between House Size and Price')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()
