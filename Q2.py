import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf

# Load the dataset
auto = pd.read_csv("Auto.csv")

# Convert 'horsepower' to numeric
auto["horsepower"] = pd.to_numeric(auto["horsepower"], errors="coerce")

# Drop missing values
auto = auto.dropna()

# Drop the 'name' column as it is not a numeric predictor
auto = auto.drop(columns=["name"])

sns.pairplot(auto)
plt.savefig("img/scatterplot_matrix.png")

correlation_matrix = auto.corr()
print(correlation_matrix)

# Heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.savefig("img/correlation_matrix.png")

# Define predictors (all except 'mpg')
X = auto.drop(columns=["mpg"])
y = auto["mpg"]

# Add a constant for the intercept
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())


formula = "mpg ~ " + " + ".join(auto.drop(columns=["mpg"]).columns)
anova_model = smf.ols(formula=formula, data=auto).fit()
anova_results = anova_lm(anova_model)
print(anova_results)

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

plt.savefig("img/diagnostic_plots.png")

# Generate the leverage plot (influence plot)
fig, ax = plt.subplots(figsize=(10, 6))
sm.graphics.influence_plot(model, ax=ax, criterion="cooks")

# Save the leverage plot
plt.savefig("img/leverage_plot.png") 

# Add interaction term: horsepower * weight
auto["hp_weight"] = auto["horsepower"] * auto["weight"]

# Refit the model with interaction
X_interact = auto.drop(columns=["mpg"])
X_interact = sm.add_constant(X_interact)
model_interact = sm.OLS(y, X_interact).fit()

# Print summary
print(model_interact.summary())

# Apply transformations
auto["log_horsepower"] = np.log(auto["horsepower"])
auto["sqrt_horsepower"] = np.sqrt(auto["horsepower"])
auto["sq_horsepower"] = auto["horsepower"] ** 2

# Fit model with transformations
X_trans = auto.drop(columns=["mpg"])
X_trans = sm.add_constant(X_trans)
model_trans = sm.OLS(y, X_trans).fit()

# Print summary
print(model_trans.summary())

import numpy as np
import statsmodels.api as sm

# Apply transformations
auto["log_weight"] = np.log(auto["weight"])
auto["sqrt_weight"] = np.sqrt(auto["weight"])

auto["log_acceleration"] = np.log(auto["acceleration"])
auto["acceleration_squared"] = auto["acceleration"] ** 2
auto["inv_acceleration"] = 1 / auto["acceleration"]

auto["log_displacement"] = np.log(auto["displacement"])
auto["displacement_squared"] = auto["displacement"] ** 2


# Define predictors (including new transformed variables)
X_trans = auto[[
    "cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin",
    "log_weight", "sqrt_weight", 
    "log_acceleration", "acceleration_squared", "inv_acceleration",
    "log_displacement", "displacement_squared"
]]

# Add a constant for the intercept
X_trans = sm.add_constant(X_trans)

# Fit the model
model_trans = sm.OLS(auto["mpg"], X_trans).fit()

# Print summary
print(model_trans.summary())

# Extract key performance metrics
print(f"R-squared: {model_trans.rsquared:.3f}")
print(f"Adjusted R-squared: {model_trans.rsquared_adj:.3f}")
print(f"AIC: {model_trans.aic:.2f}")
print(f"BIC: {model_trans.bic:.2f}")

