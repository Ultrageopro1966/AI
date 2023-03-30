from random import random, choice
import numpy as np
from math import e, ceil
from matplotlib import pyplot as plt

#READING DATA SAMPLE
f = open('dataset.txt')

def readSample():
    sample = f.readline().strip()
    input_data = list(map(int, sample[:3]))
    correct_output = int(sample[4])
    return [input_data, correct_output]

# WEIGHTS
weights = np.array([random() - random() for _ in range(3)])

# NECESSARY FUNCTIONS AND ARGS
learning_rate = 0.5
x_data = []
y_data = []
def activation_func(num:float):
    return 1/(e**(-num)+1)

def AI_prediction(ai_input:list):
    total = sum(weights * np.array(ai_input))
    ai_output = activation_func(total)
    return ai_output

def MSE(predicted: float, fact:float):
    return (fact - predicted) ** 2

def partial_derivative(values, weights, fact):
    change = 10**-5
    result = []
    for ind in range(len(weights)):
        small_value = weights.copy()
        weights[ind]+=change
        derivative = (MSE(fact, activation_func(sum(weights * values))) - MSE(fact, activation_func(sum(small_value * values))))/change
        weights[ind]-=change
        result.append(derivative)
    return result
        
# LEARNING
learning_iterations = 1000
correctAnswers = 0
for iteration in range(learning_iterations):
    input_data, correct_output = readSample()
    predicted_value = AI_prediction(input_data)
    error = MSE(predicted_value, correct_output)
    derivatives = partial_derivative(np.array(input_data), np.array(weights), correct_output)
    for ind in range(len(weights)):
        weights[ind] -= derivatives[ind] * learning_rate
    
    AI_answer = AI_prediction(input_data) - 0.5
    if ceil(AI_answer) == correct_output:
        correctAnswers+=1
    
    rate = round(correctAnswers/(iteration + 1), 3) * 100
    x_data.append(iteration+1)
    y_data.append(rate)


# CHECKING THE CORRECTNESS OF THE SCALES
correctAnswers = 0
totalAnswers = 10000
for _ in range(totalAnswers):
    sample = [choice([0, 1]) for _ in range(3)]
    correct_answer = 1 if sample[1] == 1 or sample[2] == 1 else 0
    AI_answer = ceil(AI_prediction(sample) - 0.5)
    if AI_answer == correct_answer:
        correctAnswers+=1

# RESULT
print("After learning:")
print(f"AI accuracy: {round(correctAnswers/totalAnswers, 5) * 100}%")
print(f'Weights: {list(map(lambda x: round(x, 2), weights))}')
plt.plot(x_data, y_data)
plt.xlabel("Iteration of training")
plt.ylabel("System learning rate (%)")
plt.show()
