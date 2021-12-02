#!/bin/sh
# This script continues a chain of slurm jobs.
# All the settings remain the same as in the old job.
# Only the jobs are reinserted into the queue.
# To continue a chain of N jobs, call the script as follows:
# sh chainjobs.sh

DEPENDENCY=""
JOB_FILE="slurm.loop"

# adjust the maximum number of jobs (125 here) to your needs
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

