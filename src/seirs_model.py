from seirsplus.models import SEIRSModel
from src.GeneticOptimizer import block_print, enable_print

params = {
    'beta': 0.155,  # Rate of transmission
    'sigma': 1 / 5.2,  # Rate of progression
    'gamma': 1 / 12.39,  # Rate of recovery
    'xi': 0.001  # Rate of re-susceptibility
}


def seirs_prediction(initI, initN, initR, predict_num_days=15, **params):
    block_print()
    model = SEIRSModel(initN=initN, initI=initI, initR=initR, **params)
    model.run(T=predict_num_days)
    enable_print()
    return model.total_num_infections()[10::10]


def seirs_prediction_with_a_lot_of_stuff(initI, initN, initR, predict_num_days=15, **params):
    block_print()
    model = SEIRSModel(initN=initN, initI=initI, initR=initR, **params)
    model.run(T=predict_num_days)
    enable_print()
    # Return infected, recovered
    return model.numS[10::10], model.numE[10::10], model.numI[10::10], model.numR[10::10], model.numF[10::10]
