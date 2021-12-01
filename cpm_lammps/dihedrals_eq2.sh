#!/bin/sh

if [ -z "$1" ]
  then
    echo "Please provide filename of LAMMPS data file as 1st input parameter."
    exit 1
fi

testing="${1##*_}"
testing2="${1%_*}"
endingonly="${testing##*.}"

eq2name="$testing2"_eq2in."$endingonly"
echo $eq2name

# find positions in LAMMPS file where dihedral section starts and ends
dstart=$(grep -n Dihedrals $1 | cut -d : -f 1)
dend=$(grep -n Impropers $1 | cut -d : -f 1)

# split LAMMPS file into dihedral section and sections before and after
head -n+$((dstart+1)) $1  > header.txt
tail -n+$((dend-1)) $1  > footer.txt

# now recompile lammps file by attaching together the split up parts
# header + new_dihedral_list + footer
cat header.txt recomb_dh_eq2.txt footer.txt > "$eq2name"  ## or is dh.txt enough? ?????
#cat header.txt dh.txt footer.txt > "$eq2name"