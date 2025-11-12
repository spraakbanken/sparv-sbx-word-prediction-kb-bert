"""Sparv plugin to annotate words with predicted words in the position in the sentence."""

from sparv import api as sparv_api  # type: ignore [import-untyped]

from sbx_word_prediction_kb_bert.annotations import predict_words__kb_bert
from sbx_word_prediction_kb_bert.constants import PROJECT_NAME

__all__ = ["predict_words__kb_bert"]

__description__ = "Calculating word predictions by mask a word in a BERT model."


__config__ = [
    sparv_api.Config(
        f"{PROJECT_NAME}.num_predictions",
        description="The number of predictions to list",
        default=5,
    ),
    sparv_api.Config(
        f"{PROJECT_NAME}.num_decimals",
        description="The number of decimals to round the score to",
        default=3,
    ),
]

__version__ = "0.6.1"
