#!/bin/bash

# Tool to concatenate the separate but consecutive MD trajectories (DCD format),
# which were created during the simulations. This allows the analysis of one 
# large file instead of several. The tool requires the free
# software catdcd, see: https://www.ks.uiuc.edu/Development/MDTools/catdcd/

#########################################
# Place this script in the parent folder of
# the directories which contain the dcd files.
# Execute as "sh continuum.sh"


# path to catdcd executable
catdcd="/scratch/ws/0/s4688360-misc/catdcd/LINUXAMD64/bin/catdcd4.0/catdcd"

# general name of the dcd files
names=$(find . -type f -name 'unwrap*dcd' | sort -n -t _ -k 2)

# name where to save the concatenated trajectory
foldername="extract_dcd"

# overview of dcd files
echo $names | tail -1

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
done

cd "$foldername"

echo "$var"

# connect separate files to one DCD file
$catdcd -o tot.dcd $var
rm new*.dcd

cd ..
