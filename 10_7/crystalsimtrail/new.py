#write code for 592 points coordinates in box of size 2.6nm and write to gro file as a crystal structure
#use random.uniform() to generate random coordinates
#use f-strings to write the coordinates to the gro file


import random

oxygen_mass = 15.9994
hydrogen_mass = 1.00794
box_size = 1

num_water_molecules = 3

coordinates = []

for i in range(num_water_molecules):
    x = random.uniform(0, box_size)
    y = random.uniform(0, box_size)
    z = random.uniform(0, box_size)
    coordinates.append((x, y, z))

gro_filename = "test.gro"
with open(gro_filename, "w") as gro_file:
    gro_file.write("TIP4P Water Molecules\n")
    gro_file.write(f"{num_water_molecules*4}\n")

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

