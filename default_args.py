def set_common_args_pwc(defaults):
    defaults['nEx'] = 50

def set_common_args_pwcbox(defaults):
    defaults['nEx'] = 10

def set_common_args_opt(defaults):
    defaults['nEx'] = 10000
    defaults['batchSize'] = 200

def set_common_args_cbf(defaults):
    defaults['nEx'] = 1000
    defaults['evalFreq'] = 100
    defaults['batchSize'] = 100
    defaults['softWeight'] = 0.01

def baselineOpt_default_args(prob_type):
    defaults = {}
    defaults['tol'] = 1e-4

    if prob_type == 'opt':
        set_common_args_opt(defaults)
    else:
        raise NotImplementedError

    return defaults

def baselineCBFQP_default_args(prob_type):
    defaults = {}

    if prob_type == 'cbf':
        set_common_args_cbf(defaults)
    else:
        raise NotImplementedError

    return defaults

def baselineNN_default_args(prob_type):
    defaults = {}
    defaults['testProj'] = 'none'
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 100
    defaults['epochs'] = 1000
    defaults['batchSize'] = 500
    defaults['lr'] = 1e-4
    defaults['hiddenSize'] = 200
    defaults['softWeight'] = 10

    if prob_type == 'pwc' or prob_type == 'pwcfull':
        set_common_args_pwc(defaults)
    elif prob_type == 'pwcbox':
        set_common_args_pwcbox(defaults)
    elif prob_type == 'opt':
        set_common_args_opt(defaults)
    elif prob_type == 'cbf':
        set_common_args_cbf(defaults)
    else:
        raise NotImplementedError

    return defaults

def baselineDC3_default_args(prob_type):
    defaults = {}
    defaults['testProj'] = 'none'
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 100
    defaults['epochs'] = 1000
    defaults['batchSize'] = 500
    defaults['lr'] = 1e-4
    defaults['hiddenSize'] = 200
    defaults['softWeight'] = 10
    defaults['useCompl'] = True
    defaults['useTrainCorr'] = True
    defaults['useTestCorr'] = False
    defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
    defaults['corrTrainSteps'] = 10
    defaults['corrTestMaxSteps'] = 0
    defaults['corrEps'] = 1e-4
    defaults['corrLr'] = 1e-7
    defaults['corrMomentum'] = 0.5

    if prob_type == 'pwc' or prob_type == 'pwcfull':
        set_common_args_pwc(defaults)
        defaults['useCompl'] = False
        defaults['corrMode'] = 'full'
        defaults['corrLr'] = 1e-2
    elif prob_type == 'pwcbox':
        set_common_args_pwcbox(defaults)
        defaults['useCompl'] = False
        defaults['corrMode'] = 'full'
        defaults['corrLr'] = 1e-2
    elif prob_type == 'opt':
        set_common_args_opt(defaults)
    elif prob_type == 'cbf':
        set_common_args_cbf(defaults)
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
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 100
    defaults['epochs'] = 1000
    defaults['batchSize'] = 500
    defaults['lr'] = 1e-4
    defaults['hiddenSize'] = 200
    defaults['softEpochs'] = 0
    defaults['softWeight'] = 10

    if prob_type == 'pwc' or prob_type == 'pwcfull':
        set_common_args_pwc(defaults)
    elif prob_type == 'pwcbox':
        set_common_args_pwcbox(defaults)
        defaults['softEpochs'] = 100
    elif prob_type == 'opt':
        set_common_args_opt(defaults)
        defaults['softEpochs'] = 100
    elif prob_type == 'cbf':
        set_common_args_cbf(defaults)
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

    if prob_type == 'pwc' or prob_type == 'pwcfull':
        set_common_args_pwc(defaults)
    elif prob_type == 'pwcbox':
        set_common_args_pwcbox(defaults)
    elif prob_type == 'opt':
        set_common_args_opt(defaults)
    elif prob_type == 'cbf':
        set_common_args_cbf(defaults)
    else:
        raise NotImplementedError

    return defaults