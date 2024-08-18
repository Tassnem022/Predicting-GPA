# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:48:28 2024

@author: Tassnem Abdelrahman
"""
# Predict student GPA based on Age, StudyHours, and Attendance

# Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt



# Function to compute the cost (mean squared error)
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# Function to perform gradient descent
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

# Read Data
path = 'E:\\study\\Neevas Internship\\Task 1\\student_performance_data.csv'
data = pd.read_csv(path, header=None, names=['ID', 'Gender', 'Age', 'StudyHours', 'AttendanceRate', 'GPA', 'Major', 'PartTimeJob', 'ExtraCurricularActivities'])

# Show data
print('Data = ')
print(data.head(10))
print('Data Description = ')
print(data.describe())
print('#####################################################################################')

# Data Processing
cleanData = data.drop(index=0)

print('Clean Data = ')
print(cleanData.head(10))
print('Clean Data Description = ')
print(cleanData.describe())
print('#####################################################################################')

# Add ones column for the intercept term
cleanData.insert(2, 'Ones', 1)

print('Clean Data with Ones = ')
print(cleanData.head(10))
print('Clean Data Description = ')
print(cleanData.describe())
print('#####################################################################################')

# Ensure relevant columns are numeric
cols_to_convert = ['Age', 'StudyHours', 'AttendanceRate', 'GPA']
cleanData[cols_to_convert] = cleanData[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values after conversion
cleanData.dropna(subset=cols_to_convert, inplace=True)

# Separate X (training data) from y (target variable)
columns = cleanData.shape[1] # Total columns
X = cleanData.iloc[:, columns-8:columns-4] # Columns 2, 3, 4 ,5 (Age, StudyHours, AttendanceRate)
y = cleanData.iloc[:, columns-4:columns-3] # Column 6 (GPA)

print('**************************************')
print('X data = \n', X.head(10))
print('y data = \n', y.head(10))
print('**************************************')

# Convert to matrices and initialize theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0, 0, 0]))

print('X.shape = ', X.shape)
print('**************************************')
print('theta \n', theta)
print('theta.shape = ', theta.shape)
print('**************************************')
print('y.shape = ', y.shape)
print('**************************************')

# Initialize variables for learning rate and iterations
alpha = 0.0001 # 0.1 0.01  0.001
iters = 2000   # 100 200 1000
#they are the best values

# Perform linear regression on the data set
g, cost = gradientDescent(X, y, theta, alpha, iters)

# Get the cost (error) of the model
finalCost = computeCost(X, y, g)

print('g = ', g)
print('cost = ', cost[:50])
print('Final Cost = ', finalCost)
print('**************************************')

# Clean all relevant columns in the original data
cols_to_clean = ['Age', 'StudyHours', 'AttendanceRate', 'GPA']
data[cols_to_clean] = data[cols_to_clean].apply(pd.to_numeric, errors='coerce')
data.dropna(subset=cols_to_clean, inplace=True)

#
X = np.linspace(data.Age.min(), data.Age.max(), 100)
X2 = np.linspace(data.StudyHours.min(), data.StudyHours.max(), 100)
X3 = np.linspace(data.AttendanceRate.min(), data.AttendanceRate.max(), 100)


# get best fit line for Age vs. GPA

print('x \n',X)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * X)#+ (g[0, 2] * X2)+ (g[0, 3] * X3)
print('f \n',f)

# draw the line for  Age vs. GPA

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(X, f, 'r', label='Prediction')
ax.scatter(data.Age , data.GPA , label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Age')
ax.set_ylabel('GPA')
ax.set_title('Age vs. GPA')

# get best fit line for StudyHours vs. GPA


print('x \n',X)
print('g \n',g)

f = g[0, 0] + (g[0, 2] * X2)#+ (g[0, 1] * X)+ (g[0, 3] * X3)
print('f \n',f)

# draw the line for StudyHours vs. GPA

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(X2, f, 'r', label='Prediction')
ax.scatter(data.StudyHours , data.GPA , label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('StudyHours')
ax.set_ylabel('GPA')
ax.set_title('StudyHours vs. GPA')

# get best fit line for AttendanceRate vs. GPA


print('x \n',X)
print('g \n',g)

f = g[0, 0]+ (g[0, 3] * X3) #+ (g[0, 1] * X)+ (g[0, 2] * X2)
print('f \n',f)

# draw the line for AttendanceRate vs. GPA

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(X3, f, 'r', label='Prediction')
ax.scatter(data.AttendanceRate , data.GPA , label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('AttendanceRate')
ax.set_ylabel('GPA')
ax.set_title('AttendanceRate vs. GPA')

# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


 # 3D Plotting
fig = plt.figure(figsize=(18, 6))

# Define grid for age, study hours, and attendance rate
age = np.linspace(data.Age.min(), data.Age.max(), 100)
study_hours = np.linspace(data.StudyHours.min(), data.StudyHours.max(), 100)
attendance_rate = np.linspace(data.AttendanceRate.min(), data.AttendanceRate.max(), 100)

# Plot Age vs. StudyHours with constant AttendanceRate
ax1 = fig.add_subplot(131, projection='3d')
age_grid, study_hours_grid = np.meshgrid(age, study_hours)
attendance_rate_const = np.mean(data.AttendanceRate)
gpa_grid = g[0, 0] + g[0, 1] * age_grid + g[0, 2] * study_hours_grid + g[0, 3] * attendance_rate_const
ax1.plot_surface(age_grid, study_hours_grid, gpa_grid, color='b', alpha=0.3)
ax1.scatter(data.Age, data.StudyHours, data.GPA, c='r', marker='o')
ax1.set_xlabel('Age')
ax1.set_ylabel('Study Hours')
ax1.set_zlabel('GPA')
ax1.set_title('Age vs. Study Hours with constant Attendance Rate')

# Plot Age vs. AttendanceRate with constant StudyHours
ax2 = fig.add_subplot(132, projection='3d')
age_grid, attendance_rate_grid = np.meshgrid(age, attendance_rate)
study_hours_const = np.mean(data.StudyHours)
gpa_grid = g[0, 0] + g[0, 1] * age_grid + g[0, 2] * study_hours_const + g[0, 3] * attendance_rate_grid
ax2.plot_surface(age_grid, attendance_rate_grid, gpa_grid, color='g', alpha=0.3)
ax2.scatter(data.Age, data.AttendanceRate, data.GPA, c='r', marker='o')
ax2.set_xlabel('Age')
ax2.set_ylabel('Attendance Rate')
ax2.set_zlabel('GPA')
ax2.set_title('Age vs. Attendance Rate with constant Study Hours')

# Plot StudyHours vs. AttendanceRate with constant Age
ax3 = fig.add_subplot(133, projection='3d')
study_hours_grid, attendance_rate_grid = np.meshgrid(study_hours, attendance_rate)
age_const = np.mean(data.Age)
gpa_grid = g[0, 0] + g[0, 1] * age_const + g[0, 2] * study_hours_grid + g[0, 3] * attendance_rate_grid
ax3.plot_surface(study_hours_grid, attendance_rate_grid, gpa_grid, color='r', alpha=0.3)
ax3.scatter(data.StudyHours, data.AttendanceRate, data.GPA, c='r', marker='o')
ax3.set_xlabel('Study Hours')
ax3.set_ylabel('Attendance Rate')
ax3.set_zlabel('GPA')
ax3.set_title('Study Hours vs. Attendance Rate with constant Age')

plt.show()

#Test Algorithm
Age =  input("Age = ") 
StudyHours =  input("StudyHours = ")
AttendanceRate =  input("AttendanceRate = ")
GPA = g[0, 0] + (g[0, 1] * np.float64(Age)) + (g[0, 2] * np.float64(StudyHours)) + (g[0, 3] * np.float64(AttendanceRate))
print('Expected GPA = ', GPA)