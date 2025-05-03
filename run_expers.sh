#!/bin/bash

## Usage
## bash run_expers.sh > commands
## cat commands | xargs -n1 -P8 -I{} /bin/sh -c "{}" 

# some buffer for GPU scheduling at the start
sleep 10

for i in 2 3 4 5
do
    for probType in cbf # toy toyfull nonconvex cbf 
    do
        python baseline_dc3.py --probType $probType  --seed $i&
        # python baseline_opt.py --probType $probType &
        # python baseline_nn.py --probType $probType  --seed $i&
        # python baseline_nn.py --probType $probType --suffix _noSoft --softWeight 0.0 --seed $i&
        # python hardnet_aff.py --probType $probType --seed $i&
        # python hardnet_cvx.py --probType $probType --seed $i&
        # wait
        sleep 2
    done
done
wait

# for i in 1 2 3 4 5
# do
#     for probType in toyfull # toy toyfull nonconvex cbf 
#     do
#         python hardnet_cvx.py --probType $probType --seed $i&
#     done
#     wait
# done

# python test_nets.py --probType toyfull --expDir results/ToyFullProblem-1-1-0-50
# python test_nets.py --probType toy --expDir results/ToyProblem-1-1-0-50
# python test_nets.py --probType nonconvex --expDir results/NonconvexOpt-100-50-50-10000
# python test_nets.py --probType cbf --expDir results/SafeControl-2-2-0-1000