# sparv-sbx-word-prediction-kb-bert

[![PyPI version](https://badge.fury.io/py/sparv-sbx-word-prediction-kb-bert.svg)](https://pypi.org/project/sparv-sbx-word-prediction-kb-bert)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sparv-sbx-word-prediction-kb-bert)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/sparv-sbx-word-prediction-kb-bert)](https://pypi.org/project/sparv-sbx-word-prediction-kb-bert/)

[![Maturity badge - level 2](https://img.shields.io/badge/Maturity-Level%202%20--%20First%20Release-yellowgreen.svg)](https://github.com/spraakbanken/getting-started/blob/main/scorecard.md)
[![Stage](https://img.shields.io/pypi/status/sparv-sbx-word-prediction-kb-bert)](https://pypi.org/project/sparv-sbx-word-prediction-kb-bert/)

[![Codecov](https://codecov.io/gh/spraakbanken/sparv-sbx-word-prediction/coverage.svg)](https://codecov.io/gh/spraakbanken/sparv-sbx-word-prediction)

[![CI(check)](https://github.com/spraakbanken/sparv-sbx-word-prediction/actions/workflows/check.yml/badge.svg)](https://github.com/spraakbanken/sparv-sbx-word-prediction/actions/workflows/check.yml)
[![CI(release)](https://github.com/spraakbanken/sparv-sbx-word-prediction/actions/workflows/release-kb-bert.yml/badge.svg)](https://github.com/spraakbanken/sparv-sbx-word-prediction/actions/workflows/release-kb-bert.yml)
[![CI(scheduled)](https://github.com/spraakbanken/sparv-sbx-word-prediction/actions/workflows/scheduled.yml/badge.svg)](https://github.com/spraakbanken/sparv-sbx-word-prediction/actions/workflows/scheduled.yml)
[![CI(test)](https://github.com/spraakbanken/sparv-sbx-word-prediction/actions/workflows/test.yml/badge.svg)](https://github.com/spraakbanken/sparv-sbx-word-prediction/actions/workflows/test.yml)

Plugin for applying bert masking as a [Sparv](https://github.com/spraakbanken/sparv-pipeline) annotation.

## Install

First, install Sparv, as suggested:

```bash
pipx install sparv-pipeline
```

Then install install `sparv-sbx-word-prediction-kb-bert` with

```bash
pipx inject sparv-pipeline sparv-sbx-word-prediction-kb-bert
```

## Usage

Depending on how many explicit exports of annotations you have you can decide to use this
annotation exclusively by adding it as the only annotation to export under `xml_export`:

```yaml
xml_export:
  annotations:
    - <token>:sbx_word_prediction_kb_bert.word-prediction--kb-bert
```

To use it together with other annotations you might add it under `export`:

```yaml
export:
    annotations:
        - <token>:sbx_word_prediction_kb_bert.word-prediction--kb-bert
        ...
```

### Configuration

You can configure this plugin by the number of predictions to generate.

#### Number of Predictions

The number of predictions defaults to `5` but can be configured in `config.yaml`:

```yaml
sbx_word_prediction_kb_bert:
  num_predictions: 5
```

#### Number of Decimals

The number of decimals defaults to `3` but can be configured in `config.yaml`:

```yaml
sbx_word_prediction_kb_bert:
  num_decimals: 3
```

> [!NOTE] This also controls the cut-off, so all values where the score round to 0.000 (or the number of decimals) is discarded.

### Metadata

#### Model

| Type      | HuggingFace Model                                                                       | Revision                                 |
| --------- | --------------------------------------------------------------------------------------- | ---------------------------------------- |
| Model     | [`KBLab/bert-base-swedish-cased`](https://huggingface.co/KBLab/bert-base-swedish-cased) | c710fb8dff81abb11d704cd46a8a1e010b2b022c |
| Tokenizer | same as Model                                                                           | same as Model                            |

## Supported Python versions

This library thrives to support a Python version to End-Of-Life, and will at
least bump the minor version when support for a Python version is dropped.

The following versions of this library supports these Python versions:

- v0.7: Python 3.11
- v0.6: Python 3.8

## Changelog

This project keeps a [changelog](./CHANGELOG.md).
