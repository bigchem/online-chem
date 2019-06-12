#!/bin/bash

WDIR="work/$$"
pwd=`pwd`

rm -rf $WDIR
mkdir $WDIR
cd $WDIR

#use openbabel to convert to PDBQT format
obabel -i sdf ../$1 -o pdbqt -O ligand.pdbqt -m 2>/dev/null

score=`vina --config ../../data/mdmx.conf --ligand ligand1.pdbqt  | grep "  1 " | awk '{print $2;}'`
echo $score

cd $pwd
rm -rf $WDIR
rm -f work/$1
