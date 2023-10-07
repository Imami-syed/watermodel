import random

# Define constants for TIP4P water molecule
oxygen_mass = 15.9994  # g/mol
hydrogen_mass = 1.00794  # g/mol
box_size = 2.6  # nm()

# Calculate the number of water molecules
num_water_molecules = 592

# Initialize an empty list to store coordinates
coordinates = []

# Generate random coordinates for water molecules within the box
for i in range(num_water_molecules):
    x = random.uniform(0, box_size)
    y = random.uniform(0, box_size)
    z = random.uniform(0, box_size)
    coordinates.append((x, y, z))

# Create and write the GRO file
gro_filename = "tip4pwater.gro"
with open(gro_filename, "w") as gro_file:
    gro_file.write("TIP4P Water Molecules\n")
    gro_file.write(f"{num_water_molecules}\n")
    
    atom_index = 1
    atom_index2 = 1
    for x, y, z in coordinates:
        gro_file.write(f"  {atom_index:5}SOL       OW    {atom_index2:5}   {x:.3f}   {y:.3f}   {z:.3f}\n")
        gro_file.write(f"  {atom_index:5}SOL      HW1    {atom_index2+1:5}   {x+0.790:.3f}   {y:.3f}   {z:.3f}\n")
        gro_file.write(f"  {atom_index:5}SOL      HW2    {atom_index2+2:5}   {x+0.395:.3f}   {y+0.685:.3f}   {z:.3f}\n")
        gro_file.write(f"  {atom_index:5}SOL       MW    {atom_index2+3:5}   {x+0.395:.3f}   {y-0.685:.3f}   {z:.3f}\n")
        atom_index += 1
        atom_index2 = atom_index2 + 4
    
    gro_file.write(f"{box_size:.5f}  {box_size:.5f}  {box_size:.5f}\n")

print(f"GRO file '{gro_filename}' generated with {num_water_molecules} TIP4P water molecules.")
