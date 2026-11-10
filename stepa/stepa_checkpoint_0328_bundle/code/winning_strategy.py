"""Итерация 10: заранее фиксированная смесь 50/50 двух MAE-поправок."""
import residual_ewma as shallow
import residual_deep as deep

NAME='ensemble_shallow450_deep900_equal_ewma7'


def fit(frame,root):
    return {'shallow':shallow.fit(frame,root),'deep':deep.fit(frame,root)}


def predict(model,frame,root):
    return .5*shallow.predict(model['shallow'],frame,root)+.5*deep.predict(model['deep'],frame,root)

