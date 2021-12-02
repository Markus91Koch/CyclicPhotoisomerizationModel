#!/bin/sh
if [ -z "$1" ]
  then
    echo "Please provide filename of LAMMPS data file as 1st input parameter."
    exit 1
fi

cp in.temp_loop in.loop

sed -i s/FILENAME/"$1"/g in.loop

extension="${1##*.}"
filename="${1%.*}"
testing="${1##*_}"
testing2="${1%_*}"
endingonly="${testing##*.}"
numberonly="${testing%.*}"

newnr=$((numberonly+1))
prevnr=$((numberonly-1))
intc="$filename"_tcin."$endingonly"
inct="$filename"_ctin."$endingonly"
ineq1="$filename"_eq1in."$endingonly"
ineq2="$filename"_eq2in."$endingonly"

outtc="$filename"_tcout."$endingonly"
outct="$filename"_ctout."$endingonly"
outeq1="$filename"_eq1out."$endingonly"
outeq2="$testing2"_"$newnr"."$endingonly"

dcdtc=unwrap_"$numberonly"_Atc.dcd
dcdct=unwrap_"$numberonly"_Cct.dcd
dcdeq1=unwrap_"$numberonly"_Beq1.dcd
dcdeq2=unwrap_"$numberonly"_Deq2.dcd

oldintc="$testing2"_"$prevnr"_tcin."$endingonly"
oldinct="$testing2"_"$prevnr"_ctin."$endingonly"
oldineq1="$testing2"_"$prevnr"_eq1in."$endingonly"
oldineq2="$testing2"_"$prevnr"_eq2in."$endingonly"
oldouttc="$testing2"_"$prevnr"_tcout."$endingonly"
oldoutct="$testing2"_"$prevnr"_ctout."$endingonly"
oldouteq1="$testing2"_"$prevnr"_eq1out."$endingonly"

sed -i s/INNAMETC/"$intc"/g in.loop
sed -i s/OUTNAMETC/"$outtc"/g in.loop
sed -i s/DCDOUTTC/"$dcdtc"/g in.loop

sed -i s/INNAMECT/"$inct"/g in.loop
sed -i s/OUTNAMECT/"$outct"/g in.loop
sed -i s/DCDOUTCT/"$dcdct"/g in.loop

sed -i s/INNAMEEQ1/"$ineq1"/g in.loop
sed -i s/OUTNAMEEQ1/"$outeq1"/g in.loop
sed -i s/DCDOUTEQ1/"$dcdeq1"/g in.loop

sed -i s/INNAMEEQ2/"$ineq2"/g in.loop
sed -i s/OUTNAMEEQ2/"$outeq2"/g in.loop
sed -i s/DCDOUTEQ2/"$dcdeq2"/g in.loop

### DELETE THE OLD FILES

# not deleting everything but some
rm header.txt footer.txt azdh.txt no_azdh.txt
rm recomb_dh_tc.txt recomb_dh_ct.txt recomb_dh_eq1.txt recomb_dh_eq2.txt
mv azdh_ct.txt azdh_"$numberonly"_ct.txt
mv azdh_tc.txt azdh_"$numberonly"_tc.txt

cp tc_ids.txt tc_ids_"$numberonly".txt
cp ct_ids.txt ct_ids_"$numberonly".txt 
cp azid_trans.txt azid_trans_"$numberonly".txt
cp azid_cis.txt azid_cis_"$numberonly".txt

