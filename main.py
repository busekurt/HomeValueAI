import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('data.csv')

# Use LabelEncoder to convert categorical data to numerical values
label_encoder_brokered_by = LabelEncoder()
label_encoder_street = LabelEncoder()
label_encoder_city = LabelEncoder()
label_encoder_state = LabelEncoder()

data['brokered_by'] = label_encoder_brokered_by.fit_transform(data['brokered_by'])
data['street'] = label_encoder_street.fit_transform(data['street'])
data['city'] = label_encoder_city.fit_transform(data['city'])
data['state'] = label_encoder_state.fit_transform(data['state'])
data['status'] = data['status'].map({'for_sale': 0, 'ready_to_build': 1})

# Fill missing values
data = data.fillna(0)

# Select necessary columns
features = ['brokered_by', 'status', 'bed', 'bath', 'acre_lot', 'street', 'city', 'state', 'zip_code', 'house_size']
X = data[features]
y = data['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Features of a new house
new_house = {
    'brokered_by': 103378.0,
    'status': 'for_sale',
    'bed': 5,
    'bath': 4,
    'acre_lot': 0.12,
    'street': 1962661.0,
    'city': 'Adjuntas',
    'state': 'Puerto Rico',
    'zip_code': '00601',  # zip_code string olarak tutuluyor
    'house_size': 920.0
}

# Transform categorical variables
new_house['brokered_by'] = label_encoder_brokered_by.transform([new_house['brokered_by']])[0]
new_house['street'] = label_encoder_street.transform([new_house['street']])[0]
new_house['city'] = label_encoder_city.transform([new_house['city']])[0]
new_house['state'] = label_encoder_state.transform([new_house['state']])[0]
new_house['status'] = 0 if new_house['status'] == 'for_sale' else 1

# Convert features to DataFrame
new_house_df = pd.DataFrame([new_house])

# Predict the price
predicted_price = model.predict(new_house_df)[0]
print(f'The predicted price for the new house is: ${predicted_price:.2f}')
