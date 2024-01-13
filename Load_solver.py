from pypower.casefile import casefile
from pypower.runpf import runpf
from pypower.ppoption import ppoption
from pypower.rundcpf import rundcpf

# Base value
base_voltage = 275e3
base_power = 100e6
base_impedance = base_voltage**2 / base_power
slack_bus = 'bus 1'

# Bus data = [bus no., Voltage in pu, bus angle, P generated, Q generated, Min MVar, Max MVar, P load, Q load]
bus_data = [
    [1, 1.05, 0, None, None, -350, 500, None, None],
    [2, 1.04, None, 183, None, -200, 300, 50, 30],
    [3, None, None, None, None, None, None, 150, 50],
    [4, 1.02, None, 72, None, -100, 200, 85, 35],
    [5, None, None, None, None, None, None, 130, 40]
]

# Line data = [from, to, length, resistance per km, reactance per km, shunt susceptance per km]
line_data = [
    [1, 2, 17, 0.0015, 10, 2e-6],
    [1, 3, 24, 0.0014, 16, 3e-6],
    [2, 3, 36, 0.0012, 12, 2.5e-6],
    [2, 4, 55, 0.0018, 8, 1.3e-6],
    [2, 5, 65, 0.0013, 3, 1.6e-6],
    [3, 4, 25, 0.0019, 5, 1.5e-6],
    [4, 5, 35, 0.0016, 14, 1.5e-6]
]

# Create a PyPower case
ppc = casefile()

# Add base values
ppc["baseMVA"] = 100
ppc["bus"] = bus_data
ppc["branch"] = line_data

# Specify the slack bus
ppc["bus"][0][1] = -1  # Set voltage angle to -1 for slack bus

# Set generator active power values
for i in range(1, len(bus_data)):
    if bus_data[i][3] is not None:
        ppc["gen"].append([i, bus_data[i][3] / ppc["baseMVA"], 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])

# Set load active and reactive power values
for i in range(1, len(bus_data)):
    if bus_data[i][7] is not None:
        ppc["bus"][i - 1][2] = bus_data[i][7] / ppc["baseMVA"]

    if bus_data[i][8] is not None:
        ppc["bus"][i - 1][3] = bus_data[i][8] / ppc["baseMVA"]

# Run power flow analysis
results, success = runpf(ppc)

# Run DC power flow for better voltage angle accuracy
ppopt = ppoption(PF_DC=1)
results_dc, success_dc = rundcpf(ppc, ppopt)

# Print results
if success_dc:
    print("DC Power Flow Results:")
    print(results_dc["bus"][:, [0, 7, 8]])
else:
    print("DC Power Flow did not converge.")
