import math

def function_a(x):
    return 5 * x - math.cos(x ** 2 + 1)

def function_b(x):
    return math.cos(x ** 2 + 1) / 5

def fixed_point_iteration(initial_guess, epsilon):
    iteration_count = 0
    x_current = initial_guess

    while True:
        x_next = function_b(x_current)

        if abs(x_next - x_current) < epsilon:
            return x_next, iteration_count + 1
        
        x_current = x_next
        iteration_count += 1

result, iterations = fixed_point_iteration(0, 0.00001)
print("Root:", result)
print("Iterations:", iterations)
