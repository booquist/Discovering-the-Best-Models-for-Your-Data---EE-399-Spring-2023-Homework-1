# Discovering-the-Best-Models-for-Your-Data---EE-399-Spring-2023-Homework-1
**Author:** Brendan Oquist <br>
**Abstract:** This report explores data fitting techniques using least-squares error, including parameter sweeping and loss landscapes. We apply these techniques to a given dataset and investigate various models, such as sinusoids and polynomials. Our findings provide valuable insights into the process of data fitting for machine learning and numerical methods applications.

## I. Introduction and Overview
This project explores the use of data fitting techniques to create mathematical models that capture underlying trends or patterns in observed data. We begin by introducing the concept of least-squares error, a widely used method to measure the difference between predicted values and observed data points. We then discuss parameter sweeping techniques, which involve systematically varying the values of one or more model parameters to explore their effect on the error metric. To visualize the resulting error landscapes, we introduce the concept of loss landscapes, which provide a useful tool for identifying parameter combinations that provide better model performance.

## II. Theoretical Background
In this section, we provide the necessary mathematical background for data fitting, including least-squares error, parameter sweeping techniques, and loss landscapes. We also introduce the models we used, such as sinusoids and polynomials.

### 1. **Data Fitting** 
Data fitting is a method used to create a mathematical model that best approximates a set of observed data points. The goal is to find a function that captures the underlying trend or pattern in the data while minimizing the difference between the predicted values and the observed data points. This is often achieved using a parameterized function and adjusting its parameters to minimize a chosen error metric.

### 2. **Least-Squares Error** 
Least-squares error is a widely used method to measure the difference between predicted values and observed data points. The least-squares error is defined as:

$E = \frac{1}{n} \sum_{j=1}^{N} (f(x_j) - y_j)^2$,

where $n$ is the number of data points, $f(x_j)$ is the predicted value for the $j$-th data point, and $y_j$ is the observed value for the $j$-th data point.

The objective is to find the parameter values that minimize this error. For linear regression problems, there are closed-form solutions; for more complex models, optimization techniques like gradient descent can be used to minimize the error.

### 3. **Parameter Sweeping Techniques** 
Parameter sweeping techniques involve systematically varying the values of one or more parameters within a predetermined range to explore the effect of these changes on the error metric. This can be useful for understanding the sensitivity of a model's performance to changes in its parameters and identifying regions of the parameter space where the model performs well.

### 4. **Loss Landscapes** 
A loss landscape is a visual representation of the error metric for different values of the model's parameters. For a 2D loss landscape, two parameters are swept while the others are fixed, and the error metric is computed for each combination of the swept parameter values. The landscape can be visualized using techniques like pcolor, which generates a grid plot with colors indicating the error metric values. Local minima in the loss landscape correspond to parameter combinations that provide better model performance.

### 5. **Sinusoidal and Polynomial Models** 
In this work, we consider a sinusoidal model of the form:

$f(x) = A \cos(Bx) + Cx + D$,

where $A$, $B$, $C$, and $D$ are parameters to be determined.

Additionally, we explore polynomial models of varying degrees. A polynomial model is a function of the form:

$f(x) = a_0 + a_1 x + a_2 x^2 + \cdots + a_n x^n$,

where n is the degree of the polynomial and a_i are the polynomial coefficients.

For our purposes, we fit a linear model (degree 1), a parabolic model (degree 2), and a 19th-degree polynomial model to the data.

## III. Algorithm Implementation and Development
In this section, we detail the implementation of the data fitting models and techniques, including the code and the steps taken to apply them to the dataset. We also describe the process of parameter sweeping and the techniques used to generate loss landscapes.

**Implementing Data Fitting Model** <br>
We implemented a sinusoidal model using Python, along with NumPy and SciPy libraries. The model function f(x, A, B, C, D) is defined as A * np.cos(B * x) + C * x + D. To find the minimum error and determine the parameters A, B, C, and D, we used the minimize function from SciPy's optimize module to minimize the error function error(params, x, y), which calculates the sum of squared differences between the predicted and observed data points divided by 4. The initial guess for parameters A, B, C, and D was [1, 1, 0.7, 26].
<br>

```
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Initial guess for parameters A, B, C, and D
initial_guess = [1, 1, 0.7, 26]

# Minimizing the error function
result = minimize(error, initial_guess, args=(X, Y))

# Extracting the optimal parameters
A, B, C, D = result.x

print("Optimal parameters:")
print(f"A: {A:.4f}, B: {B:.4f}, C: {C:.4f}, D: {D:.4f}")

# Calculate the minimum error
min_error = error(result.x, X, Y)
print(f"Minimum error: {min_error:.4f}")
```
Where our error function is given by: 


```
# Error function
def error(params, x, y):
    A, B, C, D = params
    return np.sum((f(x, A, B, C, D) - y) ** 2) / 4
```

This yields the best fit model given by 
Optimal parameters:
A: 2.1717, B: 0.9093, C: 0.7325, D: 31.4528
Minimum error: 19.6600, 

As shown by the graph: <br> ![image](https://user-images.githubusercontent.com/103399658/231007034-cd692770-bb28-411b-bc4f-8b54b9e8de9b.png)


**Generating 2D Loss Landscape** <br>
We generated a 2D loss landscape by fixing two parameters at a time and sweeping through values of the other two parameters. We used the error_landscape function, which calculates the error grid for a given combination of fixed and swept parameters. For each combination of fixed and swept parameters, we used the pcolor function from the matplotlib.pyplot library to visualize the results in a grid.

```
def error_landscape(fixed_params, param_ranges, X, Y):
    grid_size = len(param_ranges[0])
    error_grid = np.zeros((grid_size, grid_size))

    for i, j in itertools.product(range(grid_size), range(grid_size)):
        params = [fixed_params[0], param_ranges[0][i], param_ranges[1][j], fixed_params[1]]
        error_grid[i, j] = error(params, X, Y)

    return error_grid

# Parameter sweep ranges
sweep_ranges = {
    'A': np.linspace(-10, 10, 100),
    'B': np.linspace(0, 2, 100),
    'C': np.linspace(-1, 2, 100),
    'D': np.linspace(20, 40, 100),
}

A, B, C, D = 2.1717, 0.9093, 0.7325, 31.4528  # Optimal parameters obtained from scipy minimize

# Fixed and swept parameter combinations
combinations = [
    ((A, B), ('A', 'B')),
    ((C, B), ('C', 'B')),
    ((A, C), ('A', 'C')),
    ((A, D), ('A', 'D')),
    ((B, C), ('B', 'C')),
    ((C, D), ('C', 'D')),
]

fig, axs = plt.subplots(3, 2, figsize=(10, 20))  # Change figsize to (10, 20) for more space between plots

for idx, (fixed_params, swept_params) in enumerate(combinations):
    error_grid = error_landscape(fixed_params, (sweep_ranges[swept_params[0]], sweep_ranges[swept_params[1]]), X, Y)
    ax = axs[idx // 2, idx % 2]
    c = ax.pcolor(sweep_ranges[swept_params[0]], sweep_ranges[swept_params[1]], error_grid, shading='auto')
    ax.set_xlabel(f'{swept_params[0]}')
    ax.set_ylabel(f'{swept_params[1]}')
    ax.set_title(f'Error landscape for {swept_params[0]}, {swept_params[1]} with {fixed_params[0]}, {fixed_params[1]} fixed')
    fig.colorbar(c, ax=ax)
```

This yields the following 2D loss landscape: <br>
![image](https://user-images.githubusercontent.com/103399658/231011906-e2f5fada-86fd-4b27-b31e-401c52ae7234.png)


**Fitting Linear, Parabolic, and 19th-Degree Polynomial Models** <br>
Using the first 20 data points as training data, we fitted linear, parabolic, and 19th-degree polynomial models to the data using the Polynomial.fit function from the NumPy's polynomial module. We then computed the least-squares errors for these models on both the training and test data (the remaining 10 data points). We observed a large degree of overfitting for the 19th-degree polynomial model, resulting in 0 error on the training data but massive error on the test data.

We took advantage of numpy's Polynomial library in order to fit a polynomial to our data: 
```
from numpy.polynomial import Polynomial

# Split data into training and test sets
X_train, Y_train = X[:20], Y[:20]
X_test, Y_test = X[20:], Y[20:]

# Fit models
linear_fit = Polynomial.fit(X_train, Y_train, 1)
parabola_fit = Polynomial.fit(X_train, Y_train, 2)
poly19_fit = Polynomial.fit(X_train, Y_train, 19)
```

While yields the following result: 
```
Least-squares errors on training data:
linear: 5.0299
parabola: 4.5179
19th-degree polynomial: 0.0000

Least-squares errors on test data:
linear: 11.3141
parabola: 75.9277
19th-degree polynomial: 38454308324079644442624.0000
```

The insane level of overfitting can be witnessed when looking at the graph: <br>
![image](https://user-images.githubusercontent.com/103399658/231007939-908a6e42-8c09-4b13-9325-446c290cddd2.png)

Removing the 19th degree polynomial, and we can see much less error: <br>
![image](https://user-images.githubusercontent.com/103399658/231007973-1b80b7b5-cbc1-4107-9b80-5a52850e0b9c.png)


**Modifying Training Data and Comparing Results** <br>
We repeated the process of fitting linear, parabolic, and 19th-degree polynomial models, but this time we used the first 10 and last 10 data points as training data. The test data consisted of the 10 middle data points. We computed the least-squares errors for these models on both the training and test data and observed extreme overfitting for the 19th-degree polynomial model, with 0 error on the outskirts of the data but a massive error in the middle.

This can be viewed in the following graph, having a similar level of error to the first polynomial fit: <br>
![image](https://user-images.githubusercontent.com/103399658/231012080-6093f196-d72e-495e-b801-3f1635249b15.png)

As we can see, the error on the outskirts of the data is 0, but the data in the middle is massive, once again signifying extreme overfitting by the 19th degree polynomial.

**By implementing these algorithms and techniques, we were able to explore various data fitting models, analyze their performance on the given dataset, and gain insights into the effects of parameter choices and training data selection on model performance.**

## IV. Computational Results
**Best-Fit Models** <br>
We determined the optimal parameters for our sinusoidal model using the least-squares error minimization approach. The best-fit model was plotted against the original data points, revealing a close fit between the model and the data. We also fitted linear, parabolic, and 19th-degree polynomial models to subsets of the data and plotted the resulting fits. <br>
![image](https://user-images.githubusercontent.com/103399658/231012363-7ac7681f-7504-4cdd-b696-162e510cd58c.png)


**Loss Landscapes** <br>
To gain insights into the behavior of the models, we generated 2D loss landscapes by fixing two parameters at a time and sweeping through the values of the other two parameters. Visualizing these landscapes provided an understanding of how varying the parameters affected the error and helped identify local minima in the error landscape. <br>
![image](https://user-images.githubusercontent.com/103399658/231012400-b7499d29-1c7d-41d6-b45c-fa2ebddef0cc.png)

**Model Comparisons** <br>
By comparing the linear, parabolic, and 19th-degree polynomial models, we observed the following:

The linear and parabolic models provided reasonable fits to the data but were somewhat limited in their flexibility.
The 19th-degree polynomial model exhibited severe overfitting, fitting the training data perfectly but performing poorly on the test data.
Strengths and Weaknesses
Based on our analysis, we identified the strengths and weaknesses of the models:

The sinusoidal, linear, and parabolic models have the advantage of simplicity, and they can provide adequate fits to the data without overfitting.
The 19th-degree polynomial model, while having great flexibility, suffers from overfitting and is prone to capturing noise in the data. This results in poor generalization performance on unseen data.
In conclusion, our computational results demonstrate the importance of choosing an appropriate model for data fitting. Simpler models, such as linear, parabolic, or sinusoidal, can provide adequate fits without overfitting, while more complex models like high-degree polynomials may overfit the data and perform poorly on unseen data.

## V. Summary and Conclusions
In this project, we explored the data fitting process using sinusoidal and polynomial models. We used the least-squares error metric to evaluate the performance of the models and employed parameter sweeping techniques to visualize the loss landscape. <br>

Our results showed that the sinusoidal model was able to capture the underlying trend in the data and had a lower error than the polynomial models. Additionally, the loss landscape provided insights into the sensitivity of the model's performance to changes in its parameters. <br>

In conclusion, data fitting is a powerful tool for approximating trends in data and can be achieved using parameterized functions and optimization techniques. Our findings suggest that the sinusoidal model may be a good choice for this particular dataset. For future work, it would be interesting to explore other types of models and evaluate their performance using different error metrics. <br>

The code used for this project is included in this repository.
