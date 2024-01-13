import numpy as np

# Bus data
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
print(f'Base Impedance: {base_impedance}')

# Number of buses
num_buses = len(bus_data)

# Initialize Y matrix
Y_matrix = np.zeros((num_buses, num_buses), dtype=np.complex_)

# Convert line data to admittance and update Y matrix
for line in line_data:
    from_bus, to_bus, length, R, X, B = line
    from_bus -= 1  # Adjust for 0-based indexing
    to_bus -= 1

    R = (R * length) / base_impedance
    X = (X * length) / base_impedance
    Z = complex(R, X)
    Y = 1 / Z
    B = complex(0, (B * length * base_impedance))

    # Off-diagonal elements
    Y_matrix[from_bus][to_bus] -= Y
    Y_matrix[to_bus][from_bus] -= Y

    # Diagonal elements
    Y_matrix[from_bus][from_bus] += Y + B
    Y_matrix[to_bus][to_bus] += Y + B

# Display the Y matrix
print("Y matrix:")
print(Y_matrix)

V_mag = [bus[1] if bus[1] is not None else 1.0 for bus in bus_data]
V_angle = [bus[2] if bus[2] is not None else 0.0 for bus in bus_data]

# Display equations for power flow calculation
for i in range(len(bus_data)):
    equations_P = []
    equations_Q = []

    for j in range(len(bus_data)):
        Gij = Y_matrix[i][j].real
        Bij = Y_matrix[i][j].imag
        theta_ij = f"θ{i + 1} - θ{j + 1}"

        if j == 0:
            equation_Pi = f"P{i + 1} = V{i + 1} * V{j + 1} * ({Gij}*cos({theta_ij}) + {Bij}*sin({theta_ij}))"
            equation_Qi = f"Q{i + 1} = V{i + 1} * V{j + 1} * ({Gij}*sin({theta_ij}) - {Bij}*cos({theta_ij}))"
        else:
            equation_Pi = f"+ V{i + 1} * V{j + 1} * ({Gij}*cos({theta_ij}) + {Bij}*sin({theta_ij}))"
            equation_Qi = f"+ V{i + 1} * V{j + 1} * ({Gij}*sin({theta_ij}) - {Bij}*cos({theta_ij}))"

        equations_P.append([equation_Pi])
        equations_Q.append([equation_Qi])

    print(f"Bus {i + 1} equations:")
    for equation in equations_P:
        print(equation)
    for equation in equations_Q:
        print(equation)
