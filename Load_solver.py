import pandapower as pp

# Define the power system
net = pp.create_empty_network()

# Create buses
bus_data = {
    1: dict(vm_pu=1.05),
    2: dict(vm_pu=1.04),
    3: dict(vm_pu=0.8757),
    4: dict(vm_pu=1.02),
    5: dict(vm_pu=0.9221),
}

for bus_number, data in bus_data.items():
    pp.create_bus(net, vn_kv=275e3, name=f"Bus {bus_number}", index=bus_number, **data)

# Create lines
line_data = [
    [1, 2, 17, 0.0015, 10, 2000],
    [1, 3, 24, 0.0014, 16, 3000],
    [2, 3, 36, 0.0012, 12, 2500],
    [2, 4, 55, 0.0018, 8, 1300],
    [2, 5, 65, 0.0013, 3, 1600],
    [3, 4, 25, 0.0019, 5, 1500],
    [4, 5, 35, 0.0016, 14, 1500]
]

for line in line_data:
    from_bus, to_bus, length, r, x, c = line
    pp.create_line_from_parameters(net, from_bus, to_bus, length_km=length, r_ohm_per_km=r, x_ohm_per_km=x, c_nf_per_km=c, max_i_ka=4)

# Set up slack bus
pp.create_ext_grid(net, bus=1)

# Define loads
P = [3.00318839, 1.32889925, -2.899204, -0.13196745, -1.30026121]
Q = [1.2508916, 1.43985876, -0.49958435, 1.08445408, -0.40015728]

for i, (p, q) in enumerate(zip(P, Q), start=1):
    pp.create_load(net, bus=i, p_mw=p, q_mvar=q)

# Perform load flow analysis
pp.runpp(net, algorithm='fdbx', max_iteration=10000, tolerance_mva=0.1)

# Power Loss
power_losses = sum(net.res_line.pl_mw)

# Display results
print(f"Total power losses in the network: {power_losses} MW")
