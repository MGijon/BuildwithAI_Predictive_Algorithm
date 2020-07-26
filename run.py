from src.pipeline import Predictor


predictor = Predictor(loss_days=15, init_date='2020-07-01')
predictor.run()
predictor.report()
