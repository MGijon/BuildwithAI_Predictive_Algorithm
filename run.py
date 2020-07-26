from src.pipeline import Predictor
from src.plot_to_gif import generate_gif_from_iterations_for_the_seir_parameters

param_ranges = {
    'beta': (0.0001, 2),  # Rate of transmission
    'sigma': (1 / 14, 1),  # Rate of progression
    'gamma': (1 / 10, 1),  # Rate of recovery
    'xi': (0.0001, 0.0001)  # Rate of re-susceptibility
}

genetic_params = {
    'max_gen': 3000,
    'stop_cond': 10000,
    'mut_range': 0.1,
    'p_regen': 0.2,
    'p_mut': 0.4
}

predictor = Predictor(loss_days=15, init_date='2020-07-01', param_ranges=param_ranges, genetic_params=genetic_params)
iterations = predictor.run(verbose=1)

predictor.report()

# This takes a while to save the gif, use frames_to_save to avoid it to take too long
generate_gif_from_iterations_for_the_seir_parameters(iterations, frames_to_save=30)
