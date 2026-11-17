"""Action-conditioned predictors."""

from mosaic_predictors.dinogru import DinoGRU
from mosaic_predictors.predictor import MosaicsPredictor

__all__ = ['DinoGRU', 'MosaicsPredictor']


def main() -> None:
    print("Hello from mosaic-predictors!")
