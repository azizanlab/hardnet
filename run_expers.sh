#!/bin/bash

## Usage
## bash run_expers.sh > commands
## cat commands | xargs -n1 -P8 -I{} /bin/sh -c "{}" 

# some buffer for GPU scheduling at the start
sleep 10

for probType in cbf # pwc pwcfull pwcbox opt cbf
do
    # for i in 1 2 3 4 5
    # do
    #     python baseline_nn.py --probType $probType --seed $i&
    # done
    # wait
    # for i in 1 2 3 4 5
    # do
    #     python baseline_nn.py --probType $probType --suffix _noSoft --softWeight 0.0 --seed $i&
    # done
    # wait
    # for i in 1 2 3 4 5
    # do
    #     python hardnet_aff.py --probType $probType --suffix _soft --softEpochs 100 --seed $i&
    # done
    # wait
    # for i in 1 2 3 4 5
    # do
    #     python baseline_dc3.py --probType $probType --seed $i&
    # done
    # wait
    # for i in 1 2 3 4 5
    # do
    #     python hardnet_cvx.py --probType $probType --seed $i&
    #     wait
    # done
    # for i in 1 2 3 4 5
    # do
    #     python baseline_opt.py --probType $probType --seed $i&
    # done
    # wait
    # for i in 1 2 3 4 5
    # do
    #     python baseline_cbfqp.py --probType $probType --seed $i&
    # done
    # wait
done

# for i in 1 2 3 4 5
# do
#     for probType in pwc # pwc pwcbox opt cbf 
#     do
#         python baseline_dc3.py --probType $probType  --seed $i&
#         # python baseline_opt.py --probType $probType --seed $i&
#         python baseline_nn.py --probType $probType  --seed $i&
#         python baseline_nn.py --probType $probType --suffix _noSoft --softWeight 0.0 --seed $i&
#         python hardnet_aff.py --probType $probType --seed $i&
#         # python hardnet_cvx.py --probType $probType --seed $i&
#         wait
#         # sleep 10
#     done
# done

# python test_nets.py --probType pwc --expDir results/PWCProblem-50
# python test_nets.py --probType pwc --expDir results/PWCFullProblem-50
# python test_nets.py --probType pwcbox --expDir results/PWCBoxProblem-10
# python test_nets.py --probType opt --expDir results/NonconvexOpt-10000
# python test_nets.py --probType cbf --expDir results/SafeControl-1000