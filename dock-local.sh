#!/bin/bash

mol=`basename $1 .sdf`;

export PATH=/usr/local/bin/:$PATH
./vina.sh $mol.sdf

