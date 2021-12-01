#!/bin/bash

#cp trimer1.dcd new_trimer1_.dcd
#/home/koch_local/catdcd/LINUX/bin/catdcd4.0/catdcd -o new_trimer1.dcd -first 2 trimer1.dcd
#catdcd="/scratch/s4688360/catdcd/LINUX/bin/catdcd4.0/catdcd"
#catdcd="/scratch/ws/0/s4688360-misc/catdcd/LINUXAMD64/bin/catdcd4.0/catdcd"

tc_names=$(find . -type f -name 'tc_ids_*txt' | sort -n -t _ -k 3)
ct_names=$(find . -type f -name 'ct_ids_*.txt' | sort -n -t _ -k 3)
trans_names=$(find . -type f -name 'azid_trans_*.txt' | sort -n -t _ -k 3)
cis_names=$(find . -type f -name 'azid_cis_*.txt' | sort -n -t _ -k 3)


foldername="switching_freq"

#echo $tc_names | tail -1
#echo $ct_names | tail -1
#echo $trans_names | tail -1
#echo $cis_names | tail -1


mkdir -p "$foldername"


echo "TRANS -to- CIS"
i=0
var=""
for item in $tc_names; do

        #echo $item
	mysum=$(awk '{s+=$2}END{print s}' "$item")
        #echo "$mysum"
        
        cd "$foldername"
        if [ $i -eq 0 ]; then
        	echo "$mysum" > tc_count.txt
        else
		echo "$mysum" >> tc_count.txt
	fi

        i=$((i+1))
	cd ..
done

echo "CIS -to- TRANS"
i=0
var=""
for item in $ct_names; do

        #echo $item
        mysum=$(awk '{s+=$2}END{print s}' "$item")
        #echo "$mysum"

        cd "$foldername"
        if [ $i -eq 0 ]; then
                echo "$mysum" > ct_count.txt
        else
                echo "$mysum" >> ct_count.txt
        fi

        i=$((i+1))
        cd ..

done


echo "TRANS"
i=0
var=""
for item in $trans_names; do

        #echo $item
        mysum=$(awk '{s+=$2}END{print s}' "$item")
        #echo "$mysum"

        cd "$foldername"
        if [ $i -eq 0 ]; then
                echo "$mysum" > trans_count.txt
        else
                echo "$mysum" >> trans_count.txt
        fi

        i=$((i+1))
        cd ..
done


echo "CIS"
i=0
var=""
for item in $cis_names; do

        #echo $item
        mysum=$(awk '{s+=$2}END{print s}' "$item")
        #echo "$mysum"

        cd "$foldername"
        if [ $i -eq 0 ]; then
                echo "$mysum" > cis_count.txt
        else
                echo "$mysum" >> cis_count.txt
        fi

        i=$((i+1))
        cd ..
done




exit


	echo $item
	extension="${item##*.}"
	filename="${item%.*}"
	end="${filename##*_}"
	thestart="${filename%_*}"
	nr="${thestart##*_}"
	echo $extension
	echo $filename
	echo $end
	echo $nr
	if [ $i -eq 0 ]; then
		cp $item new_"$nr"_"$end"."$extension"
		mv new_"$nr"_"$end"."$extension" "$foldername"
	else
		$catdcd -o new_"$nr"_"$end"."$extension" -first 2 $item 
		mv new_"$nr"_"$end"."$extension" "$foldername"
	fi
	i=$((i+1))
	echo $i
	var="${var} new_${nr}_${end}.${extension}"
	#echo $var
done

cd "$foldername"

echo "$var"

$catdcd -o tot.dcd $var
rm new*.dcd

cd ..
