# Import libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load the data from the boston house-prices dataset
boston_data = fetch_california_housing()
x = boston_data['data']
y = boston_data['target']

# Make and fit the linear regression model
# TODO: Fit the model and Assign it to the model variable
model = LinearRegression()
model.fit(x, y)

# Make a prediction using the model
sample_house = [[   3.2031,       52.,            5.47761194 ,   1.07960199,  910.,    2.26368159,   37.85    ,   -122.26      ]]

prediction = model.predict(sample_house)
print(prediction)