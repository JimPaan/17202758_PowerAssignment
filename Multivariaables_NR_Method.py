import autograd.numpy as np
import sympy as sp
from autograd.numpy.linalg import solve


def equations(vars):
    V3, V5, theta2, theta3, theta4, theta5 = vars
    f1 = (-1.327 - 7.286e-4 * np.cos(theta2) + 4.858 * np.sin(theta2) - 1.821e-4 * V3 * np.cos(theta2 - theta3)
          + 1.821 * V3 * np.sin(theta2 - theta3) - 4.102e-4 * np.cos(theta2 - theta4) + 1.824 * np.sin(theta2 - theta4)
          - 1.748e-4 * V5 * np.cos(theta2 - theta5) + 4.033 * V5 * np.sin(theta2 - theta5))
    f2 = (-1.5 + 2.947e-3 * V3**2 - 4.963e-4 * V3 * np.cos(theta3) + 3.309 * V3 * np.sin(theta3)
          - 1.821e-4 * V3 * np.cos(theta3 - theta2) + 1.821 * V3 * np.sin(theta3 - theta2)
          - 2.345e-4 * V3 * np.cos(theta3 - theta4) + 6.171 * V3 * np.sin(theta3 - theta4))
    f3 = (0.133 - 4.102e-4 * np.cos(theta4 - theta2) + 1.824 * np.sin(theta4 - theta2)
          - 2.345e-4 * V3 * np.cos(theta4 - theta3) + 6.171 * V3 * np.sin(theta4 - theta3)
          - 1.799e-4 * V5 * np.cos(theta4 - theta5) + 1.574 * V5 * np.sin(theta4 - theta5))
    f4 = (-1.3 + 1.857e-3 * V5**2 - 1.748e-3 * V5 * np.cos(theta5 - theta2) + 4.033 * V5 * np.sin(theta5 - theta2)
          - 1.799e-4 * V5 * np.sin(theta5 - theta4) + 1.574 * V5 * np.sin(theta5 - theta4))
    f5 = (-0.5 - 10.952 * V3**2 - 4.963e-4 * V3 * np.sin(theta3) - 3.309 * V3 * np.cos(theta3)
          - 1.821e-4 * V3 * np.sin(theta3 - theta2) - 1.821 * V3 * np.cos(theta3 - theta2)
          - 2.345e-4 * np.sin(theta3 - theta4) - 6.171 * V3 * np.cos(theta3 - theta4))
    f6 = (-0.4 - 5.421 * V5**2 - 1.748e-3 * V5 * np.sin(theta5 - theta2) - 4.033 * V5 * np.cos(theta5 - theta2)
          - 1.799e-4 * V5 * np.cos(theta5 - theta4) - 1.574 * V5 * np.cos(theta5 - theta4))

    return np.array([f1, f2, f3, f4, f5, f6])


# Define a function that returns the Jacobian matrix using autograd
def jacobian_matrix(vars):
    V3, V5, theta2, theta3, theta4, theta5 = vars

    # Define the symbolic variables
    V3_sym, V5_sym, theta2_sym, theta3_sym, theta4_sym, theta5_sym = sp.symbols('V3 V5 theta2 theta3 theta4 theta5')

    # Define the equations
    f1 = -1.327 - 7.286e-4 * sp.cos(theta2_sym) + 4.858 * sp.sin(theta2_sym) - 1.821e-4 * V3_sym * sp.cos(
        theta2_sym - theta3_sym) + 1.821 * V3_sym * sp.sin(theta2_sym - theta3_sym) - 4.102e-4 * sp.cos(theta2_sym - theta4_sym) + 1.824 * sp.sin(
        theta2_sym - theta4_sym) - 1.748e-4 * V5_sym * sp.cos(theta2_sym - theta5_sym) + 4.033 * V5_sym * sp.sin(theta2_sym - theta5_sym)
    f2 = -1.5 + 2.947e-3 * V3_sym ** 2 - 4.963e-4 * V3_sym * sp.cos(theta3_sym) + 3.309 * V3_sym * sp.sin(
        theta3_sym) - 1.821e-4 * V3_sym * sp.cos(theta3_sym - theta2_sym) + 1.821 * V3_sym * sp.sin(
        theta3_sym - theta2_sym) - 2.345e-4 * V3_sym * sp.cos(theta3_sym - theta4_sym) + 6.171 * V3_sym * sp.sin(theta3_sym - theta4_sym)
    f3 = 0.133 - 4.102e-4 * sp.cos(theta4_sym - theta2_sym) + 1.824 * sp.sin(theta4_sym - theta2_sym) - 2.345e-4 * V3_sym * sp.cos(
        theta4_sym - theta3_sym) + 6.171 * V3_sym * sp.sin(theta4_sym - theta3_sym) - 1.799e-4 * V5_sym * sp.cos(
        theta4_sym - theta5_sym) + 1.574 * V5_sym * sp.sin(theta4_sym - theta5_sym)
    f4 = -1.3 + 1.857e-3 * V5_sym ** 2 - 1.748e-3 * V5_sym * sp.cos(theta5_sym - theta2_sym) + 4.033 * V5_sym * sp.sin(
        theta5_sym - theta2_sym) - 1.799e-4 * V5_sym * sp.sin(theta5_sym - theta4_sym) + 1.574 * V5_sym * sp.sin(theta5_sym - theta4_sym)
    f5 = -0.5 - 10.952 * V3_sym ** 2 - 4.963e-4 * V3_sym * sp.sin(theta3_sym) - 3.309 * V3_sym * sp.cos(
        theta3_sym) - 1.821e-4 * V3_sym * sp.sin(theta3_sym - theta2_sym) - 1.821 * V3_sym * sp.cos(theta3_sym - theta2_sym) - 2.345e-4 * sp.sin(
        theta3_sym - theta4_sym) - 6.171 * V3_sym * sp.cos(theta3_sym - theta4_sym)
    f6 = -0.4 - 5.421 * V5_sym ** 2 - 1.748e-3 * V5_sym * sp.sin(theta5_sym - theta2_sym) - 4.033 * V5_sym * sp.cos(
        theta5_sym - theta2_sym) - 1.799e-4 * V5_sym * sp.cos(theta5_sym - theta4_sym) - 1.574 * V5_sym * sp.cos(theta5_sym - theta4_sym)

    # Calculate the Jacobian matrix
    functions = [f1, f2, f3, f4, f5, f6]
    Jacobian_matrix = sp.Matrix(
        [[sp.diff(f, var) for var in (V3_sym, V5_sym, theta2_sym, theta3_sym, theta4_sym, theta5_sym)] for f in
         functions])

    # Substitute numerical values into the Jacobian matrix
    subs = {V3_sym: V3, V5_sym: V5, theta2_sym: theta2, theta3_sym: theta3, theta4_sym: theta4, theta5_sym: theta5}
    Jacobian_matrix = Jacobian_matrix.subs(subs)

    return Jacobian_matrix


def multivariable_newton_raphson(initial_vars, tolerance=1e-8, max_iter=500):
    vars_current = initial_vars.astype(np.float64)
    iteration = 0
    while iteration < max_iter:
        f_current = equations(vars_current)
        jac_matrix = jacobian_matrix(vars_current)
        jac_matrix_np = np.array(jac_matrix).astype(np.float64)
        f_current_np = np.array(f_current).astype(np.float64)

        vars_next = vars_current - solve(jac_matrix_np, f_current_np)

        # Check for convergence
        if np.max(np.abs(vars_next - vars_current)) < tolerance:
            print(f"Converged in {iteration + 1} iterations")
            return vars_next

        vars_current = vars_next
        iteration += 1

    print("Did not converge")
    return vars_current


# Initial guess for variables
initial_vars = np.array([1.0, 1.0, 0, 0, 0, 0])

# Compute using Newton-Raphson method
result = multivariable_newton_raphson(initial_vars)
print("Result:", result)

