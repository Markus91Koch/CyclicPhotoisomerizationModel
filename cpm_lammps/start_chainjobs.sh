#!/bin/sh
# This script starts a chainjob of simulations.
# To start a chain of N jobs, call the script as follows:
# sh start_chainjobs.sh N LAMMPSFILE.data 

if [ -z "$1" ]
  then
    echo "Please provide the number of chained jobs as the 1st input parameter.."
    exit 1
fi


if [ -z "$2" ]
  then
    echo "Please provide the number of internal LOOPS as the 2nd input parameter.."
    exit 1
fi

if [ -z "$3" ]
  then
    echo "Please provide the name of the Lammps (parent) file as the 3rd input parameter."
    exit 1
fi



# adjust to the number of MD steps during each stage of the CPM loop
tcsteps=1500
ctsteps=1500
eq1steps=15000
eq2steps=30000


extension="${3##*.}"
filename="${3%.*}"
testing="${3##*_}"
testing2="${3%_*}"
endingonly="${testing##*.}"
numberonly="${testing%.*}"
loopnumber=$2


DEPENDENCY=""
JOB_FILE="slurm.loop"

for i in `seq 1 "$1"`;
        do
		j=$((i-1))
		k=$((i+1))
	
		echo "ITERATION $i"
		mkdir "$i"

		### COPY AND EDIT SCRIPTS AND FILES
		cp in.temp_loop slurm.loop *.py *.sh "$i"
		if [ "$i" -eq 1 ]; then
			cp "$3" "$i"
		fi
		
		# delete "NEXTFOLDER" LINE IN LAST FOLDER
		if [ "$i" -eq "$1" ]; then
			sed -i '/NEXTFOLDER/d' "$i"/in.temp_loop
                fi
		
		sed -i s/LOOPNUMBER/"$2"/g "$i"/in.temp_loop
		sed -i s/NEXTFOLDER/"$k"/g "$i"/in.temp_loop
		sed -i s/NAME/"$i"run/g "$i"/slurm.loop
		
		sed -i s/TCSTEPS/"$tcsteps"/g "$i"/in.temp_loop
		sed -i s/CTSTEPS/"$ctsteps"/g "$i"/in.temp_loop
		sed -i s/EQ1STEPS/"$eq1steps"/g "$i"/in.temp_loop
		sed -i s/EQ2STEPS/"$eq2steps"/g "$i"/in.temp_loop
		

		### SUBMIT JOBS WITH DEPENDENCY
		cd "$i"

		
		if [ "$i" -eq 1 ]; then
			sh start_loop.sh "$3"
		else
			nextnumber=$(((i-1)*loopnumber+1))
			nextfilename="$filename"_"$nextnumber"."$endingonly"
			sh start_loop.sh $nextfilename
		fi

		
		JOB_CMD="sbatch"
        	if [ -n "$DEPENDENCY" ] ; then
                	JOB_CMD="$JOB_CMD --dependency afterok:$DEPENDENCY"
        	fi
        	JOB_CMD="$JOB_CMD $JOB_FILE"
        	echo -n "$i - Running command: $JOB_CMD  "

        	OUT=`$JOB_CMD`
        	echo "Result: $OUT"
        	DEPENDENCY=`echo $OUT | awk '{print $4}'`

        	cd ..
done
