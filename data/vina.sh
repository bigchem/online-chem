#!/bin/bash

WDIR="work/temp-docking/$$"
pwd=`pwd`

rm -rf $WDIR
mkdir $WDIR

obabel -i sdf $1 -o pdbqt -O $WDIR/ligand.pdbqt -m 2>/dev/null
cd $WDIR

score=`vina --config ../../../data/mdmx.conf --ligand ligand1.pdbqt  | grep "  1 " | awk '{print $2;}'`
echo $score

cd $pwd
rm -rf $WDIR 

