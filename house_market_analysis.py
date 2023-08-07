import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error ,mean_squared_error

# Create sub-dataframe from csv file
df = pd.read_csv("house_market_data.csv")
sdf = df[['sqft_living', "price", 'sqft_lot', 'bedrooms', 'sqft_above', 'city', 'country', 'floors']].loc[df['price'] > 0]

# Removing outliers and restricting location to only one area, since the variance of all the data together is too high to model accurately.
sdf2 = sdf.loc[sdf['price'] <= 1000000].loc[sdf['city'] == 'Redmond']

# Split the dataframe up into dataframes x and y
X = sdf2[['sqft_living', 'sqft_lot', 'sqft_above']]
Y = sdf2['price']

# Perform a train test split on the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Set the model to linear regression and fit the model to the data of all Redmond houses, excluding outliers
model = LinearRegression()
model.fit(X_train, y_train)

# Make a test predictions dataframe and test the predictions by calculating mean absolute error
test_predictions = model.predict(X_test)
print("The absolute mean error is:")
print(mean_absolute_error(y_test, test_predictions))
print("The mean is:")
print(sdf2['price'].mean())

# Generate a plot of price to square foot living to indicate the linear relationship between the variables
# Take note that the data is trained on 3 x variables, not just one.
# Therefore, it is not possible to display the relationship of all three variables at once
x1 = sdf2['sqft_living']
x2 = sdf2['sqft_lot']
x3 = sdf2['bedrooms']
x4 = sdf2['sqft_above']
y = sdf2['price']
plt.scatter(x1,y)
plt.show()

# We could deduce that there is a linear relationship between the different square foot measurements of houses and the price of the houses.
