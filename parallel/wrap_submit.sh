#/bin/bash

# list of seeds, generate this with np.random.randint and check with
# np.unique for duplicates
ensemble_mode=true

if [ "$ensemble_mode" = true ]; then
    seeds=(978362 601950 620839 317686 813163 608432 \
                  396754 274530 639995 643774)
else
    seeds=(978362)
fi


function run_sbatch {
    sbcommand=`sbatch $@`
    if [[ "$sbcommand" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        jobid=${BASH_REMATCH[1]}
        echo "$jobid"
    else
        echo "SBATCH_FAILED"
    fi
}

if [ "$#" -eq 0 ]; then
    echo "no arguments given, not running $runscript"
    exit 1
else
    runscript=$1
    exp_id=$2
    echo "running $runscript $exp_id"
fi


if [ "$ensemble_mode" = true ]; then
    echo "ensemble mode --------------- "

    ctr=0
    for i in "${seeds[@]}"; do
        echo "ensemble member" $ctr", seed" $i
        exp_id_ctr=${exp_id}"/member_"$ctr
        jobid=`run_sbatch submit.sh $runscript $exp_id_ctr $i`
        ctr=$((ctr + 1))
    done
else
    echo "normal mode --------------- "
    jobid=`run_sbatch submit.sh $runscript $exp_id ${seeds[0]}`
fi

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
