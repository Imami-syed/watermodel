#!/bin/bash
# energy minimization 
gmx grompp -f em.mdp -c box.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em -ntomp 39
# npt simulations
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -v -deffnm npt -ntomp 39