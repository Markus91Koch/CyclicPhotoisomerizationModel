#!/bin/sh

if [ -z "$1" ]
  then
    echo "Please provide filename of LAMMPS data file as 1st input parameter."
    exit 1
fi


extension="${1##*.}"
filename="${1%.*}"
testing="${1##*_}"
testing2="${1%_*}"
endingonly="${testing##*.}"
numberonly="${testing%.*}"
startname="${testing2%_*}"

re='^[0-9]+$'
if ! [[ $numberonly =~ $re ]] ; then
   #echo "error: Not a number" >&2; exit 1
   tcname="$filename"_1_tcin."$endingonly"
else
   #newnr=$((numberonly+1))
   tcname="$filename"_tcin."$endingonly"
fi

echo $tcname


# i think this is not needed anymore ..?
#cat azodih.txt | awk '{ print $2 }' > da.txt

# find positions in LAMMPS file where dihedral section starts and ends
dstart=$(grep -n Dihedrals $1 | cut -d : -f 1)
dend=$(grep -n Impropers $1 | cut -d : -f 1)

# split LAMMPS file into dihedral section and sections before and after
head -n+$((dstart+1)) $1  > header.txt
tail -n+$((dend-1)) $1  > footer.txt
tail -n+$((dstart+2)) $1 | head -$((dend-dstart-3)) > dh.txt

# first sort the dihedrals by 3rd column
# the C-N=N-C dihedrals will then appear in the same order as I got them in the azodih.txt file (LAMMPS dump)
# create two files: 1st file with only dihedrals of type 19, 25, 26
# 2nd file with all dihedrals that are not of type 19, 25, 26

# SO: 1st a file withonly dihedrals of type 19, 23, 24
awk '$2 == "19" || $2 == "23" || $2 == "24" { print $0 }' dh.txt | sort -nk3 > azdh.txt

# AND: a file with dihedrals that are not of type 19, 23, 24
awk '$2 != "19" && $2 != "23" && $2 != "24" { print $0 }' dh.txt | sort -nk3 > no_azdh.txt 

python3 change_dihedrals.py  # create from azdh.txt new files azdh_ct.txt and azdh_tc.txt where switches will happen

# recombine azdh.txt and no_azdh.txt and sort by 1st column again
cat no_azdh.txt azdh_tc.txt | sort -nk1 > recomb_dh_tc.txt
cat no_azdh.txt azdh_ct.txt | sort -nk1 > recomb_dh_ct.txt
cat no_azdh.txt azdh.txt | sort -nk1 > recomb_dh_eq1.txt
cat no_azdh.txt azdh.txt | sort -nk1 > recomb_dh_eq2.txt

# now recompile lammps file by attaching together the split up parts
# header + new_dihedral_list + footer
cat header.txt recomb_dh_tc.txt footer.txt > "$tcname" #"$newname"
