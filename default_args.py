def baselineOpt_default_args(prob_type):
    defaults = {}
    defaults['tol'] = 1e-4

    if prob_type in ['toy', 'toyfull']:
        defaults['nVar'] = 1
        defaults['nIneq'] = 1
        defaults['nEq'] = 0
        defaults['nEx'] = 50
    elif prob_type in ['qp', 'nonconvex']:
        defaults['nVar'] = 100
        defaults['nIneq'] = 50
        defaults['nEq'] = 50
        defaults['nEx'] = 10000
    elif 'cbf' in prob_type:
        defaults['nVar'] = 2
        defaults['nIneq'] = 2
        defaults['nEq'] = 0
        defaults['nEx'] = 1000
    else:
        raise NotImplementedError

    return defaults

def baselineNN_default_args(prob_type):
    defaults = {}
    defaults['useTestProj'] = False
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 100
    defaults['epochs'] = 1000
    defaults['batchSize'] = 500
    defaults['lr'] = 1e-4
    defaults['hiddenSize'] = 200
    defaults['softWeight'] = 10
    defaults['softWeightEqFrac'] = 0.5

    if prob_type in ['toy', 'toyfull']:
        defaults['nVar'] = 1
        defaults['nIneq'] = 1
        defaults['nEq'] = 0
        defaults['nEx'] = 50
        defaults['softWeightEqFrac'] = 0.0
    elif prob_type in ['qp', 'nonconvex']:
        defaults['nVar'] = 100
        defaults['nIneq'] = 50
        defaults['nEq'] = 50
        defaults['nEx'] = 10000
    elif 'cbf' in prob_type:
        defaults['nVar'] = 2
        defaults['nIneq'] = 2
        defaults['nEq'] = 0
        defaults['nEx'] = 1000
        defaults['batchSize'] = 100
        defaults['softWeight'] = 0.01
        defaults['softWeightEqFrac'] = 0.0
    else:
        raise NotImplementedError

    return defaults

def baselineDC3_default_args(prob_type):
    defaults = {}
    defaults['useTestProj'] = False
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 100
    defaults['epochs'] = 1000
    defaults['batchSize'] = 200
    defaults['lr'] = 1e-4
    defaults['hiddenSize'] = 200
    defaults['softWeight'] = 10
    defaults['softWeightEqFrac'] = 0.5
    defaults['useCompl'] = True
    defaults['useTrainCorr'] = True
    defaults['useTestCorr'] = False
    defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
    defaults['corrTrainSteps'] = 10
    defaults['corrTestMaxSteps'] = 0
    defaults['corrEps'] = 1e-4
    defaults['corrLr'] = 1e-7
    defaults['corrMomentum'] = 0.5

    if prob_type in ['toy', 'toyfull']:
        defaults['nVar'] = 1
        defaults['nIneq'] = 1
        defaults['nEq'] = 0
        defaults['nEx'] = 50
        defaults['softWeightEqFrac'] = 0.0
        defaults['useCompl'] = False
        defaults['corrMode'] = 'full'
        defaults['corrLr'] = 1e-2
    elif prob_type in ['qp', 'nonconvex']:
        defaults['nVar'] = 100
        defaults['nIneq'] = 50
        defaults['nEq'] = 50
        defaults['nEx'] = 10000
    elif 'cbf' in prob_type:
        defaults['nVar'] = 2
        defaults['nIneq'] = 2
        defaults['nEq'] = 0
        defaults['nEx'] = 1000
        defaults['batchSize'] = 100
        defaults['softWeight'] = 0.01
        defaults['softWeightEqFrac'] = 0.0
        defaults['useCompl'] = False
        defaults['corrMode'] = 'full'
        defaults['useTestCorr'] = True
        defaults['corrTrainSteps'] = 5 # reduced correction steps due to too long training time
        defaults['corrTestMaxSteps'] = 5 # instead, added more correction steps at test time
    else:
        raise NotImplementedError

    return defaults

def hardnetAff_default_args(prob_type):
    defaults = {}
    defaults['nEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 100
    defaults['epochs'] = 1000
    defaults['batchSize'] = 500
    defaults['lr'] = 1e-4
    defaults['hiddenSize'] = 200
    defaults['softEpochs'] = 0
    defaults['softWeight'] = 10
    defaults['softWeightEqFrac'] = 0.5

    if prob_type in ['toy', 'toyfull']:
        defaults['nVar'] = 1
        defaults['nIneq'] = 1
        defaults['nEq'] = 0
        defaults['nEx'] = 50
        defaults['softWeightEqFrac'] = 0.0
    elif prob_type in ['qp', 'nonconvex']:
        defaults['nVar'] = 100
        defaults['nIneq'] = 50
        defaults['nEq'] = 50
        defaults['lr'] = 1e-3
    elif 'cbf' in prob_type:
        defaults['nVar'] = 2
        defaults['nIneq'] = 2
        defaults['nEq'] = 0
        defaults['nEx'] = 1000
        defaults['batchSize'] = 100
        defaults['softWeight'] = 0.01
        defaults['softWeightEqFrac'] = 0.0
    else:
        raise NotImplementedError

    return defaults

def hardnetCvx_default_args(prob_type):
    defaults = {}
    defaults['nEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 100
    defaults['epochs'] = 1000
    defaults['batchSize'] = 500
    defaults['lr'] = 1e-4
    defaults['hiddenSize'] = 200
    defaults['softEpochs'] = 0
    defaults['softWeight'] = 10
    defaults['softWeightEqFrac'] = 0.5

    if prob_type in ['toy', 'toyfull']:
        defaults['nVar'] = 1
        defaults['nIneq'] = 1
        defaults['nEq'] = 0
        defaults['nEx'] = 50
        defaults['softWeightEqFrac'] = 0.0
    elif prob_type in ['qp', 'nonconvex']:
        defaults['nVar'] = 100
        defaults['nIneq'] = 50
        defaults['nEq'] = 50
    elif 'cbf' in prob_type:
        defaults['nVar'] = 2
        defaults['nIneq'] = 2
        defaults['nEq'] = 0
        defaults['nEx'] = 1000
        defaults['batchSize'] = 100
        defaults['softWeight'] = 0.01
        defaults['softWeightEqFrac'] = 0.0
    else:
        raise NotImplementedError

    return defaults