#/bin/bash

function run_sbatch {
    sbcommand=`sbatch $1 $2`
    if [[ "$sbcommand" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        jobid=${BASH_REMATCH[1]}
        echo "$jobid"
    else
        echo "SBATCH_FAILED"
    fi
}

if [ "$#" -eq 0 ]; then
    echo "no arguments given, running $runscript"
    exit 1
else
    runscript=$1
    arg1=$2
    echo "running $runscript $arg1"
fi

jobid=`run_sbatch submit.sh $runscript $arg1`
slurmfile=slurm-$jobid.out

while true; do
    if [ -f $slurmfile ]; then
        break
    else
        echo "... waiting for" $slurmfile
        sleep 1;
    fi
done

echo "... keeping track of" $slurmfile
tail -f -n 2000 $slurmfile
