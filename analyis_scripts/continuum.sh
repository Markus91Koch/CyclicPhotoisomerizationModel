#!/bin/bash

#cp trimer1.dcd new_trimer1_.dcd
#/home/koch_local/catdcd/LINUX/bin/catdcd4.0/catdcd -o new_trimer1.dcd -first 2 trimer1.dcd
#catdcd="/scratch/s4688360/catdcd/LINUX/bin/catdcd4.0/catdcd"
#catdcd="/scratch/ws/0/s4688360-misc/catdcd/LINUXAMD64/bin/catdcd4.0/catdcd"

tc_names=$(find . -type f -name 'tc_ids_*txt' | sort -n -t _ -k 3)
ct_names=$(find . -type f -name 'ct_ids_*.txt' | sort -n -t _ -k 3)
trans_names=$(find . -type f -name 'azid_trans_*.txt' | sort -n -t _ -k 3)
cis_names=$(find . -type f -name 'azid_cis_*.txt' | sort -n -t _ -k 3)


foldername="TRANS_CIS_CONTINUUM"

#echo $tc_names | tail -1
#echo $ct_names | tail -1
#echo $trans_names | tail -1
#echo $cis_names | tail -1


mkdir -p "$foldername"


##echo "TRANS -to- CIS"
#i=0
#var=""
#for item in $tc_names; do
#
#        #echo $item
#	mysum=$(awk '{s+=$2}END{print s}' "$item")
#        #echo "$mysum"
#        
#        cd "$foldername"
#        if [ $i -eq 0 ]; then
#        	echo "$mysum" > tc_count.txt
#        else
#		echo "$mysum" >> tc_count.txt
#	fi
#
#        i=$((i+1))
#	cd ..
#done
#
#echo "CIS -to- TRANS"
#i=0
#var=""
#for item in $ct_names; do
#
#        #echo $item
#        mysum=$(awk '{s+=$2}END{print s}' "$item")
#        #echo "$mysum"
#
#        cd "$foldername"
#        if [ $i -eq 0 ]; then
#                echo "$mysum" > ct_count.txt
#        else
#                echo "$mysum" >> ct_count.txt
#        fi
#
#        i=$((i+1))
#        cd ..
#
#done


echo "TRANS"
i=0
var=""
for item in $trans_names; do

	mycol=$(awk '{print $2}' "$item")
	
        cd "$foldername"
        if [ $i -eq 0 ]; then
                echo "$mycol" > trans_CONT.txt
        else
		echo "$mycol" > new_buffer.txt
		cp trans_CONT.txt trans_buffer.txt
		paste -d' ' trans_buffer.txt new_buffer.txt > trans_CONT.txt
		rm trans_buffer.txt new_buffer.txt
        fi

        i=$((i+1))

        cd ..
done

#exit


echo "CIS"
i=0
var=""
for item in $cis_names; do

        mycol=$(awk '{print $2}' "$item")

        cd "$foldername"
        if [ $i -eq 0 ]; then
                echo "$mycol" > cis_CONT.txt
        else
                echo "$mycol" > new_buffer.txt
                cp cis_CONT.txt cis_buffer.txt
                paste -d' ' cis_buffer.txt new_buffer.txt > cis_CONT.txt
                rm cis_buffer.txt new_buffer.txt
        fi

        i=$((i+1))

        cd ..
done


echo "TRANS-to-CIS"
i=0
var=""
for item in $tc_names; do

        mycol=$(awk '{print $2}' "$item")

        cd "$foldername"
        if [ $i -eq 0 ]; then
                echo "$mycol" > tc_CONT.txt
        else
                echo "$mycol" > new_buffer.txt
                cp tc_CONT.txt tc_buffer.txt
                paste -d' ' tc_buffer.txt new_buffer.txt > tc_CONT.txt
                rm tc_buffer.txt new_buffer.txt
        fi

        i=$((i+1))

        cd ..
done

echo "CIS-to-TRANS"
i=0
var=""
for item in $ct_names; do

        mycol=$(awk '{print $2}' "$item")

        cd "$foldername"
        if [ $i -eq 0 ]; then
                echo "$mycol" > ct_CONT.txt
        else
                echo "$mycol" > new_buffer.txt
                cp ct_CONT.txt ct_buffer.txt
                paste -d' ' ct_buffer.txt new_buffer.txt > ct_CONT.txt
                rm ct_buffer.txt new_buffer.txt
        fi

        i=$((i+1))

        cd ..
done

