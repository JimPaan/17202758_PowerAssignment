import autograd.numpy as aunp
from autograd import jacobian
from autograd.numpy.linalg import solve
import math
import numpy as np


def equations(vars):
    theta2, theta3, theta4, theta5, V3, V5 = vars

    f1 = (-1.32685 - 7.2869e-4 * aunp.cos(theta2) + 4.857762 * aunp.sin(theta2) - 1.821e-4 * V3 * aunp.cos(theta2 - theta3) +
          1.821 * V3 * aunp.sin(theta2 - theta3) - 4.102e-4 * aunp.cos(theta2 - theta4) + 1.823 * aunp.sin(theta2 - theta4) -
          1.748e-4 * V5 * aunp.cos(theta2 - theta5) + 4.033 * V5 * aunp.sin(theta2 - theta5))

    f2 = (1.5 + 2.6464e-3 * V3 ** 2 - 1.809e-4 * V3 * aunp.cos(theta3) + 2.0679e-4 * V3 * aunp.sin(theta3) -
          1.821e-4 * V3 * aunp.cos(theta3 - theta2) + 1.821 * V3 * aunp.sin(theta3 - theta2) -
          2.345e-4 * V3 * aunp.cos(theta3 - theta4) + 6.171 * V3 * aunp.sin(theta3 - theta4))

    f3 = (0.133 - 4.102e-4 * aunp.cos(theta4 - theta2) + 1.823 * aunp.sin(theta4 - theta2) -
          2.345e-4 * V3 * aunp.cos(theta4 - theta3) + 6.171 * V3 * aunp.sin(theta4 - theta3) -
          1.799e-4 * V5 * aunp.cos(theta4 - theta5) + 1.574 * V5 * aunp.sin(theta4 - theta5))

    f4 = (1.3 + 1.857e-3 * V5 ** 2 - 1.748e-3 * V5 * aunp.cos(theta5 - theta2) + 4.033 * V5 * aunp.sin(theta5 - theta2) -
          1.799e-4 * V5 * aunp.sin(theta5 - theta4) + 1.574 * V5 * aunp.sin(theta5 - theta4))

    f5 = (0.5 + 9.695 * V3 ** 2 - 1.819e-4 * V3 * aunp.sin(theta3) - 2.068 * V3 * aunp.cos(theta3) -
          1.821e-4 * V3 * aunp.sin(theta3 - theta2) - 1.821 * V3 * aunp.cos(theta3 - theta2) -
          2.345e-3 * aunp.sin(theta3 - theta4) - 6.171 * V3 * aunp.cos(theta3 - theta4))

    f6 = (0.4 + 5.362 * V5 ** 2 - 1.748e-3 * V5 * aunp.sin(theta5 - theta2) -
          4.033 * V5 * aunp.cos(theta5 - theta2) - 1.799e-4 * V5 * aunp.cos(theta5 - theta4) -
          1.574 * V5 * aunp.cos(theta5 - theta4))

    return aunp.array([f1, f2, f3, f4, f5, f6])


def calculate_power(V, theta):
    # Values for voltages and angles for each bus
    V1, V2, V4, V3, V5 = V
    theta1, theta2, theta3, theta4, theta5 = theta

    P1 = (V1 * V1 * (0.0008396019865774078 * math.cos(theta1 - theta1) + -6.377849088261236 * math.sin(theta1 - theta1))
          + V1 * V2 * (-0.0006672793967509193 * math.cos(theta1 - theta2) + 4.448529311672796 * math.sin(theta1 - theta2))
          + V1 * V3 * (-0.0001723225898264885 * math.cos(theta1 - theta3) + 1.9694010265884399 * math.sin(theta1 - theta3)))

    Q1 = (V1 * V1 * (0.0008396019865774078 * math.sin(theta1 - theta1) + 6.377849088261236 * math.cos(theta1 - theta1))
          + V1 * V2 * (-0.0006672793967509193 * math.sin(theta1 - theta2) - 4.448529311672796 * math.cos(theta1 - theta2))
          + V1 * V3 * (-0.0001723225898264885 * math.sin(theta1 - theta3) - 1.9694010265884399 * math.cos(theta1 - theta3)))

    P2 = (V2 * V2 * (0.0029096112357777024 * math.cos(theta2 - theta2) - 11.682813873323523 * math.sin(theta2 - theta2))
          + V2 * V1 * (-0.0006672793967509193 * math.cos(theta2 - theta1) + 4.448529311672796 * math.sin(
                theta2 - theta1))
          + V2 * V3 * (-0.00017505786861979168 * math.cos(theta2 - theta3) + 1.750578686197917 * math.sin(
                theta2 - theta3))
          + V2 * V4 * (-0.0003867187304223643 * math.cos(theta2 - theta4) + 1.7187499129882857 * math.sin(
                theta2 - theta4))
          + V2 * V5 * (-0.001680555239984627 * math.cos(theta2 - theta5) + 3.8782043999645244 * math.sin(
                theta2 - theta5)))

    Q2 = (V2 * V2 * (0.0029096112357777024 * math.sin(theta2 - theta2) + 11.682813873323523 * math.cos(theta2 - theta2))
          + V2 * V1 * (-0.0006672793967509193 * math.sin(theta2 - theta1) - 4.448529311672796 * math.cos(
                theta2 - theta1))
          + V2 * V3 * (-0.00017505786861979168 * math.sin(theta2 - theta3) - 1.750578686197917 * math.cos(
                theta2 - theta3))
          + V2 * V4 * (-0.0003867187304223643 * math.sin(theta2 - theta4) - 1.7187499129882857 * math.cos(
                theta2 - theta4))
          + V2 * V5 * (-0.001680555239984627 * math.sin(theta2 - theta5) - 3.8782043999645244 * math.cos(
                theta2 - theta5)))

    P3 = (V3 * V1 * (-0.0001723225898264885 * math.cos(theta3 - theta1) + 1.9694010265884399 * math.sin(theta3 - theta1))
          + V3 * V2 * (-0.00017505786861979168 * math.cos(theta3 - theta2) + 1.750578686197917 * math.sin(theta3 - theta2))
          + V3 * V3 * (0.002646380126470728 * math.cos(theta3 - theta3) - 9.694542901666482 * math.sin(theta3 - theta3))
          + V3 * V4 * (-0.0022989996680244477 * math.cos(theta3 - theta4) + 6.049999126380126 * math.sin(theta3 - theta4)))

    Q3 = (V3 * V1 * (-0.0001723225898264885 * math.sin(theta3 - theta1) - 1.9694010265884399 * math.cos(theta3 - theta1))
          + V3 * V2 * (-0.00017505786861979168 * math.sin(theta3 - theta2) - 1.750578686197917 * math.cos(theta3 - theta2))
          + V3 * V3 * (0.002646380126470728 * math.sin(theta3 - theta3) + 9.694542901666482 * math.cos(theta3 - theta3))
          + V3 * V4 * (-0.0022989996680244477 * math.sin(theta3 - theta4) - 6.049999126380126 * math.cos(theta3 - theta4)))

    P4 = (V4 * V1 * (0.0 * math.cos(theta4 - theta1) + 0.0 * math.sin(theta4 - theta1))
          + V4 * V2 * (-0.0003867187304223643 * math.cos(theta4 - theta2) + 1.7187499129882857 * math.sin(theta4 - theta2))
          + V4 * V3 * (-0.0022989996680244477 * math.cos(theta4 - theta3) + 6.049999126380126 * math.sin(theta4 - theta3))
          + V4 * V4 * (0.002862103235793156 * math.cos(theta4 - theta4) - 9.25104917864892 * math.sin(theta4 - theta4))
          + V4 * V5 * (-0.00017638483734634384 * math.cos(theta4 - theta5) + 1.5433673267805086 * math.sin(theta4 - theta5)))

    Q4 = (V4 * V1 * (0.0 * math.sin(theta4 - theta1) - 0.0 * math.cos(theta4 - theta1))
          + V4 * V2 * (-0.0003867187304223643 * math.sin(theta4 - theta2) - 1.7187499129882857 * math.cos(theta4 - theta2))
          + V4 * V3 * (-0.0022989996680244477 * math.sin(theta4 - theta3) - 6.049999126380126 * math.cos(theta4 - theta3))
          + V4 * V4 * (0.002862103235793156 * math.sin(theta4 - theta4) + 9.25104917864892 * math.cos(theta4 - theta4))
          + V4 * V5 * (-0.00017638483734634384 * math.sin(theta4 - theta5) - 1.5433673267805086 * math.cos(theta4 - theta5)))

    P5 = (V5 * V1 * (0.0 * math.cos(theta5 - theta1) + 0.0 * math.sin(theta5 - theta1))
          + V5 * V2 * (-0.001680555239984627 * math.cos(theta5 - theta2) + 3.8782043999645244 * math.sin(theta5 - theta2))
          + V5 * V3 * (0.0 * math.cos(theta5 - theta3) + 0.0 * math.sin(theta5 - theta3))
          + V5 * V4 * (-0.00017638483734634384 * math.cos(theta5 - theta4) + 1.5433673267805086 * math.sin(theta5 - theta4))
          + V5 * V5 * (0.001856940077330971 * math.cos(theta5 - theta5) - 5.362395164245033 * math.sin(theta5 - theta5)))

    Q5 = (V5 * V1 * (0.0 * math.sin(theta5 - theta1) - 0.0 * math.cos(theta5 - theta1))
          + V5 * V2 * (-0.001680555239984627 * math.sin(theta5 - theta2) - 3.8782043999645244 * math.cos(theta5 - theta2))
          + V5 * V3 * (0.0 * math.sin(theta5 - theta3) - 0.0 * math.cos(theta5 - theta3))
          + V5 * V4 * (-0.00017638483734634384 * math.sin(theta5 - theta4) - 1.5433673267805086 * math.cos(theta5 - theta4))
          + V5 * V5 * (0.001856940077330971 * math.sin(theta5 - theta5) + 5.362395164245033 * math.cos(theta5 - theta5)))

    return aunp.array([P1, P2, P3, P4, P5, Q1, Q2, Q3, Q4, Q5])


def equation_fd(vars):
    # Values for voltages and angles for each bus
    V3, V5, theta2, theta3, theta4, theta5 = vars

    V1, V2, V4, theta1 = [1.05, 1.04, 1.02, 0]

    P2 = (V2 * V1 * (4.448529311672796 * math.sin(theta2 - theta1))
          + V2 * V3 * (1.750578686197917 * math.sin(theta2 - theta3))
          + V2 * V4 * (1.7187499129882857 * math.sin(theta2 - theta4))
          + V2 * V5 * (3.8782043999645244 * math.sin(theta2 - theta5)))

    P3 = (V3 * V1 * (1.9694010265884399 * math.sin(theta3 - theta1))
          + V3 * V2 * (1.750578686197917 * math.sin(theta3 - theta2))
          + V3 * V4 * (-6.049999126380126 * math.sin(theta3 - theta4)))

    P4 = (V4 * V1 * (0.0 * math.cos(theta4 - theta1) + 0.0 * math.sin(theta4 - theta1))
          + V4 * V2 * (-0.0003867187304223643 * math.cos(theta4 - theta2) + 1.7187499129882857 * math.sin(
                theta4 - theta2))
          + V4 * V3 * (-0.0022989996680244477 * math.cos(theta4 - theta3) + 6.049999126380126 * math.sin(
                theta4 - theta3))
          + V4 * V4 * (0.002862103235793156 * math.cos(theta4 - theta4) - 9.18998199114892 * math.sin(theta4 - theta4))
          + V4 * V5 * (-0.00017638483734634384 * math.cos(theta4 - theta5) + 1.5433673267805086 * math.sin(
                theta4 - theta5)))

    P5 = (V5 * V1 * (0.0 * math.cos(theta5 - theta1) + 0.0 * math.sin(theta5 - theta1))
          + V5 * V2 * (-0.001680555239984627 * math.cos(theta5 - theta2) + 3.8782043999645244 * math.sin(
                theta5 - theta2))
          + V5 * V3 * (0.0 * math.cos(theta5 - theta3) + 0.0 * math.sin(theta5 - theta3))
          + V5 * V4 * (-0.00017638483734634384 * math.cos(theta5 - theta4) + 1.5433673267805086 * math.sin(
                theta5 - theta4))
          + V5 * V5 * (0.001856940077330971 * math.cos(theta5 - theta5) + -5.303218601745033 * math.sin(
                theta5 - theta5)))

    Q3 = (V3 * V1 * (0.0001723225898264885 * math.sin(theta3 - theta1) - 1.9694010265884399 * math.cos(theta3 - theta1))
          + V3 * V2 * (0.00017505786861979168 * math.sin(theta3 - theta2) - 1.750578686197917 * math.cos(
                theta3 - theta2))
          + V3 * V3 * (0.002646380126470728 * math.sin(theta3 - theta3) + 9.619106964166482 * math.cos(theta3 - theta3))
          + V3 * V4 * (
                      0.0022989996680244477 * math.sin(theta3 - theta4) + 6.049999126380126 * math.cos(theta3 - theta4)))

    Q5 = (V5 * V1 * (0.0 * math.sin(theta5 - theta1) - 0.0 * math.cos(theta5 - theta1))
          + V5 * V2 * (-0.001680555239984627 * math.sin(theta5 - theta2) - 3.8782043999645244 * math.cos(
                theta5 - theta2))
          + V5 * V3 * (0.0 * math.sin(theta5 - theta3) - 0.0 * math.cos(theta5 - theta3))
          + V5 * V4 * (-0.00017638483734634384 * math.sin(theta5 - theta4) - 1.5433673267805086 * math.cos(
                theta5 - theta4))
          + V5 * V5 * (0.001856940077330971 * math.sin(theta5 - theta5) - -5.303218601745033 * math.cos(
                theta5 - theta5)))

    return aunp.array([])


def jacobian_matrix(vars):
    return jacobian(equations)(vars)


def multivariable_newton_raphson(initial_vars, tolerance=1e-8, max_iter=200):
    vars_current = initial_vars
    iteration = 0
    while iteration < max_iter:
        f_current = equations(vars_current)
        jac_matrix = jacobian_matrix(vars_current)
        vars_next = vars_current - solve(jac_matrix, f_current)

        # Check for convergence
        if aunp.max(aunp.abs(vars_next - vars_current)) < tolerance:
            print(f"Converged in {iteration + 1} iterations")
            return vars_next

        vars_current = vars_next
        iteration += 1

    print("Did not converge")
    return vars_current


def ybusmatrix_gen(bus_data, line_data, base_impedance):
    num_buses = len(bus_data)

    # Initialize Y matrix
    Y_matrix = aunp.zeros((num_buses, num_buses), dtype=aunp.complex_)

    # Convert line data to admittance and update Y matrix
    for line in line_data:
        from_bus, to_bus, length, R, X, B = line
        from_bus -= 1  # Adjust for 0-based indexing
        to_bus -= 1

        R = (R * length) / base_impedance
        X = (X * length) / base_impedance
        Z = complex(R, X)
        Y = 1 / Z
        B = complex(0, (B * length) * base_impedance)

        # Off-diagonal elements
        Y_matrix[from_bus][to_bus] -= Y
        Y_matrix[to_bus][from_bus] -= Y

        # Diagonal elements
        Y_matrix[from_bus][from_bus] += Y + B
        Y_matrix[to_bus][to_bus] += Y + B

    return Y_matrix


bus_data = [
    [1, 1.05, 0, None, None, -350, 500, None, None],
    [2, 1.04, None, 183, None, -200, 300, 50, 30],
    [3, None, None, None, None, None, None, 150, 50],
    [4, 1.02, None, 72, None, -100, 200, 85, 35],
    [5, None, None, None, None, None, None, 130, 40]
]

# Line data
line_data = [
    [1, 2, 17, 0.0015, 10, 2e-6],
    [1, 3, 24, 0.0014, 16, 3e-6],
    [2, 3, 36, 0.0012, 12, 2.5e-6],
    [2, 4, 55, 0.0018, 8, 1.3e-6],
    [2, 5, 65, 0.0013, 3, 1.6e-6],
    [3, 4, 25, 0.0019, 5, 1.5e-6],
    [4, 5, 35, 0.0016, 14, 1.5e-6]
]

# Base voltage
base_voltage = 275e3
base_power = 100e6
base_impedance = base_voltage**2 / base_power

Y_bus = ybusmatrix_gen(bus_data, line_data, base_impedance)

print(f'Y matrix: {Y_bus}')

# Initial guess for variables
initial_vars = aunp.array([0, 0, 0, 0, 1.0, 1.0])

v1 = 1.05
v2 = 1.04
v4 = 1.02
thet1 = 0.0

V = [v1, v2, v4]
theta = [thet1]

print(jacobian_matrix(initial_vars))

# Compute using Newton-Raphson method
result = multivariable_newton_raphson(initial_vars)
print("Result:", result)

result_V = []
result_theta = []

for i in range(4, 6):
    cur = result[i]
    result_V.append(cur)

for i in range(4):
    cur = result[i]
    result_theta.append(cur)

V.extend(result_V)
theta.extend(result_theta)

power = calculate_power(V, theta)

print("Power(P,Q):", power)
