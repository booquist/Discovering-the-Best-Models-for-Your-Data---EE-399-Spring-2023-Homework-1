# Discovering-the-Best-Models-for-Your-Data---EE-399-Spring-2023-Homework-1
**Author:** Brendan Oquist <br>

## I. Introduction and Overview
**Abstract:** This report explores data fitting techniques using least-squares error, including parameter sweeping and loss landscapes. We apply these techniques to a given dataset and investigate various models, such as sinusoids and polynomials. Our findings provide valuable insights into the process of data fitting for machine learning and numerical methods applications.

## II. Theoretical Background
In this section, we provide the necessary mathematical background for data fitting, including least-squares error, parameter sweeping techniques, and loss landscapes. We also introduce the models we used, such as sinusoids and polynomials.

### 1. **Data Fitting** 
Data fitting is a method used to create a mathematical model that best approximates a set of observed data points. The goal is to find a function that captures the underlying trend or pattern in the data while minimizing the difference between the predicted values and the observed data points. This is often achieved using a parameterized function and adjusting its parameters to minimize a chosen error metric.

### 2. **Least-Squares Error** 
Least-squares error is a widely used method to measure the difference between predicted values and observed data points. The least-squares error is defined as:

$E = \frac{1}{n} \sum_{j=1}^{n} (f(x_j) - y_j)^2$,

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
Here, we detail the implementation of the data fitting models and techniques, including the code and the steps taken to apply them to the dataset. We also describe the process of parameter sweeping and the techniques used to generate loss landscapes.

## IV. Computational Results
In this section, we present the results of the data fitting process, including the best-fit models and the corresponding loss landscapes. We also compare the results of different models and discuss their strengths and weaknesses.

## V. Summary and Conclusions
Finally, we summarize our findings and provide conclusions on the data fitting process. We discuss the implications of our results and provide recommendations for future work in this area. Additionally, we include the code used for this project and provide instructions for running the code.
