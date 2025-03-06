import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the Auto dataset
auto = pd.read_csv("Auto.csv") 

# Horsepower is in object type
# print(auto.dtypes)

# Drop missing values if any
auto = auto.dropna()

# Convert 'horsepower' to numeric (if necessary)
auto["horsepower"] = pd.to_numeric(auto["horsepower"], errors="coerce")

# Drop NaN values after conversion
auto = auto.dropna()

# After conversion horsepower is in float64 type
print(auto.dtypes)

# Define predictor (X) and response (y)
X = auto["horsepower"]
y = auto["mpg"]

# Add a constant for the intercept, without it model assumes it passes through the origin
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())

# Part (a) - Answering the questions based on summary
# Predict mpg for horsepower of 98
hp_98 = np.array([[1, 98]])  # Constant term + horsepower value
predicted_mpg = model.predict(hp_98)[0]

# Compute confidence and prediction intervals
predictions = model.get_prediction(hp_98)
conf_int = predictions.conf_int(alpha=0.05)  # 95% confidence interval
pred_int = predictions.summary_frame(alpha=0.05)[["obs_ci_lower", "obs_ci_upper"]]

print(f"Predicted mpg for horsepower 98: {predicted_mpg}")
print(f"95% Confidence Interval: {conf_int}")
print(f"95% Prediction Interval: {pred_int}")

# Part (b) - Plotting the regression line
fig, ax = plt.subplots()
ax.scatter(auto["horsepower"], auto["mpg"], label="Data", alpha=0.5)
ax.axline((0, model.params["const"]), slope=model.params["horsepower"], color="red", label="Regression Line")
ax.set_xlabel("Horsepower")
ax.set_ylabel("MPG")
ax.legend()
ax.set_title("Linear Regression Model for Auto Dataset")
plt.savefig("img/regression.png")

# Part (c) - Diagnostic plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residual plot
axes[0].scatter(model.fittedvalues, model.resid, alpha=0.5)
axes[0].axhline(0, color="red", linestyle="dashed")
axes[0].set_xlabel("Fitted Values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residual Plot")

# Q-Q Plot for normality check
sm.qqplot(model.resid, line="s", ax=axes[1])
axes[1].set_title("Q-Q Plot")
plt.savefig("img/residual_qq.png")
