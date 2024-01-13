import pandas as pd
import pypsa

# Base values
base_voltage = 275e3
base_power = 100e6
base_impedance = base_voltage**2 / base_power
slack_bus = 'bus 1'

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
    ['bus 1', 'bus 2', 17, 0.0015, 10, 2e-6],
    ['bus 1', 'bus 3', 24, 0.0014, 16, 3e-6],
    ['bus 2', 'bus 3', 36, 0.0012, 12, 2.5e-6],
    ['bus 2', 'bus 4', 55, 0.0018, 8, 1.3e-6],
    ['bus 2', 'bus 5', 65, 0.0013, 3, 1.6e-6],
    ['bus 3', 'bus 4', 25, 0.0019, 5, 1.5e-6],
    ['bus 4', 'bus 5', 35, 0.0016, 14, 1.5e-6]
]

# Convert data to DataFrames
bus_df = pd.DataFrame(bus_data, columns=['bus', 'v_nom', 'angle', 'p_set', 'q_set', 'q_min', 'q_max', 'p_load', 'q_load'])
line_df = pd.DataFrame(line_data, columns=['bus0', 'bus1', 'length', 'r', 'x', 'b'])

# Create a PyPSA network
network = pypsa.Network()

# Add buses and lines to the network
network.add("Bus", bus_df)
network.add("Line", line_df)

# Set base values for voltage, power, and impedance
network.buses['v_nom'] = base_voltage
network.buses['p_nom'] = base_power
network.buses['impedance'] = base_impedance

# Set slack bus
network.slack_bus = slack_bus

# Print intermediate results for debugging
print("Before power flow calculation:")
print(network.buses)

# Run power flow calculation
network.pf()

# Retrieve results
voltages = network.buses_t.v_m_pu
line_flows = network.lines_t.p0

# Print results
print("\nBus Voltages:")
print(voltages)

print("\nLine Flows:")
print(line_flows)
