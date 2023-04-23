# CSE/OAN 6250 Reproducibility Challenge Team G4

## Description
This effort is an attempt to reproduce the results found in the paper “Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention.”[^1] This effort will compare the peformance of a 5-label convolutional neural network to assess the risk of suicide sourced from Reddit posts. It will be compared to to three, four, and five-label models utilizing support vector machine, random forest, and feed-forward neural network. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)

This effort is in partial completion of CSE/OAN 6250 reproducibility challenge. [Original dataset](500_Reddit_users_posts_labels.csv) included is sourced from the original project team repository.[^2] Either source is sufficient to reproduce the results of this effort.

## Installation
1. Download or copy this repository to a local directory
2. Reference the associated [environment.yml](environment.yml) for a list of dependencies. Alternatively, if using Anaconda, apply the environment.yml by `conda env create --name envname --file=environment.yml`

## Usage
1. `0_enrichment.py` is the first stage in the analysis pipeline. This has the file `500_Reddit_users_posts_labels.csv` as input and the file `reddit_data_with_cf.csv` as output.
2. `1_main.py` is the most major step in the pipeline. This performs that model fitting and testing. Output files from this script are saved in the output folder in a folder named after the input parameters that you name the script. The input parameters are `script.py DATA_SIZE NUM_EPOCHS [SVM-RBF|SVM-L|RF|FFNN|CNN] [5|4|3+1] [USE_CF|NO_CF]`.
3. `2_extract_table.py` this script turns data in the `figures` folder into tables 7, 8, and 9 in the paper. [^1]

## Credits

[^1]: Manas Gaur et al., “Knowledge-Aware Assessment of Severity of Suicide Risk for Early Intervention,” Zenodo (CERN European Organization for Nuclear Research), May 13, 2019, https://doi.org/10.1145/3308558.3313698
[^2]: https://github.com/manasgaur/Knowledge-aware-Assessment-of-Severity-of-Suicide-Risk-for-Early-Intervention
