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


re='^[0-9]+$'
if ! [[ $numberonly =~ $re ]] ; then
   intc="$filename"_1_tcin."$endingonly"
   inct="$filename"_1_ctin."$endingonly"
   ineq1="$filename"_1_eq1in."$endingonly"
   ineq2="$filename"_1_eq2in."$endingonly"

   outtc="$filename"_1_tcout."$endingonly"
   outct="$filename"_1_ctout."$endingonly"
   outeq1="$filename"_1_eq1out."$endingonly"
   outeq2="$filename"_2."$endingonly"

   dcdtc=unwrap_1_Atc.dcd
   dcdct=unwrap_1_Cct.dcd
   dcdeq1=unwrap_1_Beq1.dcd
   dcdeq2=unwrap_1_Deq2.dcd

else
   #newnr=$((numberonly+1))
   #newname="$testing2"_."$endingonly"
   #echo "Supply a file that does not end on a number!"
   newnr=$((numberonly+1))
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
fi

echo $intc
echo $inct
echo $ineq1
echo $ineq2
echo $outtc
echo $outct
echo $outeq1
echo $outeq2

sed -i s/INNAMETC/"$intc"/g in.loop
sed -i s/OUTNAMETC/"$outtc"/g in.loop
sed -i s/DCDOUTTC/"$dcdtc"/g in.loop
#sed -i s/DCDOUTTC/unwrap_1a.dcd/g in.loop

sed -i s/INNAMECT/"$inct"/g in.loop
sed -i s/OUTNAMECT/"$outct"/g in.loop
sed -i s/DCDOUTCT/"$dcdct"/g in.loop
#sed -i s/DCDOUTCT/unwrap_1c.dcd/g in.loop

sed -i s/INNAMEEQ1/"$ineq1"/g in.loop
sed -i s/OUTNAMEEQ1/"$outeq1"/g in.loop
sed -i s/DCDOUTEQ1/"$dcdeq1"/g in.loop
#sed -i s/DCDOUTEQ1/unwrap_1b.dcd/g in.loop

sed -i s/INNAMEEQ2/"$ineq2"/g in.loop
sed -i s/OUTNAMEEQ2/"$outeq2"/g in.loop
sed -i s/DCDOUTEQ2/"$dcdeq2"/g in.loop
#sed -i s/DCDOUTEQ2/unwrap_1d.dcd/g in.loop


#sed -i s/OUTNAMEFINAL/"$newname"/g in.loop


#sbatch slurm.loop
