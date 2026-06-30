import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Square_Feet": [1500, 1800, 2400, 3000, 3500, 1200, 2000, 2700, 3200, 4000],
    "Bedrooms": [3, 3, 4, 4, 5, 2, 3, 4, 4, 5],
    "Bathrooms": [2, 2, 3, 3, 4, 1, 2, 3, 4, 4],
    "Price": [300000, 350000, 450000, 550000, 650000,
              200000, 380000, 480000, 590000, 700000]
}

df = pd.DataFrame(data)

print("\n Dataset Preview")
print("-" * 40)
print(df)


X = df[["Square_Feet", "Bedrooms", "Bathrooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n Model Performance Evaluation")
print("-" * 40)
print(f"Train R² Score : {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R² Score  : {r2_score(y_test, y_test_pred):.4f} (Ideal is close to Train R²)")
print(f"Test MAE       : ${mean_absolute_error(y_test, y_test_pred):,.2f}")
# RMSE is the square root of MSE, putting the error back in original currency units ($)
print(f"Test RMSE      : ${mean_squared_error(y_test, y_test_pred, squared=False):,.2f}")
print("\n Learned Regression Formula")
print("-" * 40)
intercept = model.intercept_
coefs = model.coef_
print(f"Price = {intercept:,.2f} \n"
      f"        + ({coefs[0]:,.2f} * Square_Feet) \n"
      f"        + ({coefs[1]:,.2f} * Bedrooms) \n"
      f"        + ({coefs[2]:,.2f} * Bathrooms)")


comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_test_pred.round(2),
    "Absolute Error": abs(y_test.values - y_test_pred).round(2)
})

print("\n Test Set Predictions & Error Margin")
print("-" * 40)
print(comparison.to_string(index=False))


new_house = pd.DataFrame({
    "Square_Feet": [2200],
    "Bedrooms": [4],
    "Bathrooms": [3]
})

predicted_price = model.predict(new_house)[0]

print("\n New House Prediction")
print("-" * 40)
print("House Features : 2,200 sqft, 4 Bedrooms, 3 Bathrooms")
print(f"Estimated Price: ${predicted_price:,.2f}")