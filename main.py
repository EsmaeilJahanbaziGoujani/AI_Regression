import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.linear_model import LinearRegression  # For linear regression modeling
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation

# 1. Create a sample dataset (since file is not provided)
data = {
    'Area': [1200, 1500, 800, 950, 1100, 1300, 1700, 1600, 900, 1000],
    'Rooms': [3, 4, 2, 2, 3, 3, 5, 4, 2, 3],
    'Distance_to_City': [10, 15, 7, 8, 12, 11, 20, 18, 9, 10],
    'Price': [300000, 400000, 200000, 220000, 280000, 320000, 450000, 420000, 210000, 250000]
}
df = pd.DataFrame(data)  # Create DataFrame

# 2. Split dataset into features (X) and target (y)
X = df[['Area', 'Rooms', 'Distance_to_City']]  # Features
y = df['Price']  # Target variable

# 3. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # 80% training, 20% testing

# 4. Create and train the LinearRegression model
model = LinearRegression()  # Initialize the model
model.fit(X_train, y_train)  # Train the model
print(model.coef_)
print(model.intercept_)

# 5. Predict house prices on the test set
y_pred = model.predict(X_test)  # Make predictions

# 6. Evaluate the model with R² and MSE
r2 = r2_score(y_test, y_pred)  # R-squared score
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error

print("R² score:", r2)  # Print R-squared
print("Mean Squared Error:", mse)  # Print MSE
print("Predicted prices:", y_pred)  # Print predicted prices
print("Actual prices:", y_test.values)  # Print actual prices

# 1. Convert predicted values into a DataFrame (keep same index as X_test)
predicted_df = pd.DataFrame({          # Create a new DataFrame for predicted values
    'Predicted Price': y_pred          # Name the column 'Predicted Price', values from model prediction
}, index=X_test.index)                 # Set the same index as X_test so they align row by row

# 2. Combine test features (X_test), actual prices (y_test), and predicted prices (y_pred)
result_df = pd.concat([                # Concatenate multiple DataFrames column-wise (axis=1)
    X_test,                            # First: original test features (Area, Rooms, Distance_to_City)
    y_test.rename("Actual Price"),     # Second: actual prices from the test set (renamed for clarity)
    predicted_df                       # Third: predicted prices calculated by the model
], axis=1)                             # Combine them side by side (columns)

# 3. Print the final table showing all details
print("\n🔍 House Price Prediction Results:\n")  # Print a title for better readability
print(result_df)