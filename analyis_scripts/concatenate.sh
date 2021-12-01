#!/bin/bash

#cp trimer1.dcd new_trimer1_.dcd
#/home/koch_local/catdcd/LINUX/bin/catdcd4.0/catdcd -o new_trimer1.dcd -first 2 trimer1.dcd
#catdcd="/scratch/s4688360/catdcd/LINUX/bin/catdcd4.0/catdcd"
catdcd="/scratch/ws/0/s4688360-misc/catdcd/LINUXAMD64/bin/catdcd4.0/catdcd"

names=$(find . -type f -name 'unwrap*dcd' | sort -n -t _ -k 2)

foldername="extract_dcd"

echo $names | tail -1

#exit

mkdir -p "$foldername"

i=0
var=""
for item in $names; do
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
