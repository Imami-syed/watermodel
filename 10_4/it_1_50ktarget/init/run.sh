#!/bin/bash
#rm -r it1
#mkdir it1
#cp -r init_files/* it1/
#source /usr/local/gromacs/bin/GMXRC
gmx grompp -f min.mdp -o min.tpr -pp min.top -po min.mdp
gmx mdrun -s min.tpr -o min.trr -x min.xtc -c min.gro -e min.edr -g min.log
gmx grompp -f min2.mdp -o min2.tpr -pp min2.top -po min2.mdp -c min.gro
gmx mdrun -s min2.tpr -o min2.trr -x min2.xtc -c min2.gro -e min2.edr -g min2.log
#gmx energy -f min.edr -o min-energy.xvg
#gmx energy -f min2.edr -o min2-energy.xvg

gmx grompp -f eql.mdp -o eql.tpr -pp eql.top -po eql.mdp -c min2.gro
gmx mdrun -s eql.tpr -o eql.trr -x eql.xtc -c eql.gro -e eql.edr -g eql.log
#gmx energy -f eql.edr -o eql-temp.xvg
gmx grompp -f eql2.mdp -o eql2.tpr -pp eql2.top -po eql2.mdp -c eql.gro
gmx mdrun -s eql2.tpr -o eql2.trr -x eql2.xtc  -c eql2.gro -e eql2.edr -g eql2.log
#gmx energy -f eql2.edr -o eql-press.xvg

# gmx grompp -f prd.mdp -o prd.tpr -pp prd.top -po prd.mdp -c eql2.gro
# gmx mdrun -s prd.tpr -o prd.trr -x prd.xtc -c prd.gro -e prd.edr -g prd.log 

