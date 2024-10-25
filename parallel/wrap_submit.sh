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
    runscript=autoencoder_original.py
    echo "no arguments given, running $runscript"
else
    runscript=$1
    echo "running $runscript"
fi

jobid=`run_sbatch submit.sh $runscript`
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
