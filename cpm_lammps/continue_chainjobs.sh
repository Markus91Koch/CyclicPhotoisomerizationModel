#!/bin/sh
# This script continues a chain of slurm jobs.
# To continue a chain of N jobs, call the script as follows:
# sh chainjobs.sh


#tcsteps=750  #150 #1500
#ctsteps=750  #150 #1500
#eq1steps=5000  #300 #15000
#eq2steps=10000  #300 #30000

#extension="${3##*.}"
#filename="${3%.*}"
#testing="${3##*_}"
#testing2="${3%_*}"
#endingonly="${testing##*.}"
#numberonly="${testing%.*}"
#loopnumber=$2

DEPENDENCY=""
JOB_FILE="slurm.loop"
DELETE_OLD=0	#should old folders be deleted first?

for i in `seq 1 125`;
        do
		j=$((i-1))
		k=$((i+1))
		if [ -d "$i" ]
		then
            nrfiles=`find "$i" -mindepth 1 -maxdepth 1 -type f -name "*.data" -printf x | wc -c`
            if [ "$nrfiles" -lt "2" ]; then
                folder="$i"
                cd $i
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
				#exit 1
			fi
		else
			exit 1	
		fi
	done

