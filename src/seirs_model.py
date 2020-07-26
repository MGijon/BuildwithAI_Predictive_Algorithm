from seirsplus.models import SEIRSModel

params = {
    'beta': 0.155,  # Rate of transmission
    'sigma': 1 / 5.2,  # Rate of progression
    'gamma': 1 / 12.39,  # Rate of recovery
    'xi': 0.001  # Rate of re-susceptibility
}


def seirs_prediction(initI, initN, **params):
    model = SEIRSModel(initN=initN, initI=initI, **params)
    model.run(T=15)
    return model.total_num_infections()[10::10]


def seirs_prediction_with_a_lot_of_stuff(initI, initN, **params):
    model = SEIRSModel(initN=initN, initI=initI, **params)
    model.run(T=15)
    # Return infected, recovered
    return model.numS[10::10], model.numE[10::10], model.numI[10::10], model.numR[10::10], model.numF[10::10]
