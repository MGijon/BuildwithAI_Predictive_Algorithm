from src.pipeline import Predictor


predictor = Predictor(loss_days=15)
predictor.run()
predictor.report()
