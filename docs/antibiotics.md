# Generating Novel Antibiotics with SyntheMol

This file contains instructions for using SyntheMol to generate novel antibiotic candidates for the Gram-negative bacterium _Acinetobacter baumannii_. These instructions reproduce the work in the paper [TODO](TODO).

The data and models referred to in this file can be downloaded from the Google Drive folder [here](https://drive.google.com/drive/folders/1VLPPUbY_FTKMjlXgRm09bPSSms206Dce?usp=share_link). Note that the instructions below assume that the relevant data is downloaded to the `data` directory.

* [Process Enamine REAL Space data](#process-enamine-real-space-data)
* [Process antibiotics training data](#process-antibiotics-training-data)
* [Process ChEMBL antibacterials](#process-chembl-antibacterials)
* [Build bioactivity prediction models](#build-bioactivity-prediction-models)
  + [Train models](#train-models)
  + [Map building blocks to model scores](#map-building-blocks-to-model-scores)
* [Generate molecules with SyntheMol](#generate-molecules-with-synthemol)
* [Filter generated molecules](#filter-generated-molecules)
  + [Novelty](#novelty)
  + [Bioactivity](#bioactivity)
  + [Diversity](#diversity)
* [Map molecules to REAL IDs](#map-molecules-to-real-ids)
* [Predict toxicity](#predict-toxicity)


## Process Enamine REAL Space data

See the [REAL Space data processing instructions](real.md) for details on how to process the Enamine REAL Space data.


## Process antibiotics training data

The antibiotics training data consists of three libraries of molecules tested against the Gram-negative bacterium _Acinetobacter baumannii_. Here, we merge the three libraries into a single file and determine which molecules are hits (i.e., active against _A. baumannii_).

Merge the three libraries into a single file and determine which molecules are hits.
```bash
python -m SyntheMol.data.process_data \
    --data_paths data/1_training_data/library_1.csv data/library_2.csv data/library_3.csv \
    --save_path data/1_training_data/antibiotics.csv \
    --save_hits_path data/1_training_data/antibiotics_hits.csv
```

Output:
```
library_1
Data size = 2,371
Mean activity = 0.9759337199493885
Std activity = 0.2673771821337539
Activity threshold of mean - 2 std = 0.4411793556818807
Number of hits = 130
Number of non-hits = 2,241

library_2
Data size = 6,680
Mean activity = 0.9701813713618264
Std activity = 0.17623388953330232
Activity threshold of mean - 2 std = 0.6177135922952217
Number of hits = 294
Number of non-hits = 6,386

library_3
Data size = 5,376
Mean activity = 0.9980530249533108
Std activity = 0.13303517604898704
Activity threshold of mean - 2 std = 0.7319826728553367
Number of hits = 112
Number of non-hits = 5,264

Full data size = 14,427
Data size after dropping non-conflicting duplicates = 13,594
Data size after dropping conflicting duplicates = 13,524

Final data size = 13,524
Number of hits = 470
Number of non-hits = 13,054
```


## Process ChEMBL antibacterials

Download lists of known antibiotic-related compounds from ChEMBL using the following search terms. For each, click the CSV download button, unzip the downloaded file, and rename the CSV file appropriately.

- [antibacterial](https://www.ebi.ac.uk/chembl/g/#search_results/compounds/query=antibacterial)
- [antibiotic](https://www.ebi.ac.uk/chembl/g/#search_results/compounds/query=antibiotic)

The results of the queries on August 8, 2022, are available in the Google Drive folder in the `Data/2. ChEMBL` subfolder.

The files are:

- `chembl_antibacterial.csv` with 604 molecules (589 with SMILES)
- `chembl_antibiotic.csv` with 636 molecules (587 with SMILES)

Merge these two files to form a single collection of antibiotic-related compounds.
```bash
python -m SyntheMol.data.merge_chembl_downloads \
    --data_paths data/2_chembl/chembl_antibacterial.csv data/2_chembl/chembl_antibiotic.csv \
    --labels antibacterial antibiotic \
    --save_path data/2_chembl/chembl.csv
```

The file `chembl.csv` contains 1,005 molecules.


## Build bioactivity prediction models

Here, we build three binary classification bioactivity prediction models to predict antibiotic activity against _A. baumannii_. The three models are:

1. Chemprop: a graph neural network model
2. Chemprop-RDKit: a graph neural network model augmented with 200 RDKit features
3. Random forest: a random forest model using 200 RDKit features


### Train models

For each model type, we trained 10 models using 10-fold cross-validation on the training data. Each ensemble of 10 models took less than 90 minutes to train on a 16-CPU machine. Trained models are available in the `Models` subfolder of the Google Drive folder.

TODO: figure out appropriate train and predict package imports

Chemprop
```bash
python -m SyntheMol.models.train \
    --data_path data/1_training_data/antibiotics.csv \
    --save_dir models/antibiotic_chemprop \
    --dataset_type classification \
    --model_type chemprop \
    --property_column activity \
    --num_models 10
```

Chemprop-RDKit
```bash
python -m SyntheMol.models.train \
    --data_path data/1_training_data/antibiotics.csv \
    --save_dir models/antibiotic_chemprop_rdkit \
    --model_type chemprop \
    --dataset_type classification \
    --fingerprint_type rdkit \
    --property_column activity \
    --num_models 10
```

Random forest
```bash
python -m SyntheMol.models.train \
    --data_path data/1_training_data/antibiotics.csv \
    --save_dir models/antibiotic_random_forest \
    --model_type random_forest \
    --dataset_type classification \
    --fingerprint_type rdkit \
    --property_column activity \
    --num_models 10
```

| Model          | ROC-AUC         | PRC-AUC         |
|----------------|-----------------|-----------------|
| Chemprop       | 0.803 +/- 0.036 | 0.354 +/- 0.086 |
| Chemprop-RDKit | 0.827 +/- 0.028 | 0.388 +/- 0.078 |
| Random forest  | 0.835 +/- 0.035 | 0.401 +/- 0.099 |


### Map building blocks to model scores

In order to speed up the generative model, we pre-compute the scores of all building blocks for each model.

Chemprop
```bash
python -m SyntheMol.models.predict \
    --data_path data/4_real_space/building_blocks.csv \
    --model_path models/antibiotic_chemprop \
    --model_type chemprop \
    --average_preds
```

Chemprop-RDKit
```bash
python -m SyntheMol.models.predict \
    --data_path data/4_real_space/building_blocks.csv \
    --model_path models/antibiotic_chemprop_rdkit \
    --model_type chemprop \
    --fingerprint_type rdkit \
    --average_preds
```

Random forest
```bash
python -m SyntheMol.models.predict \
    --data_path data/4_real_space/building_blocks.csv \
    --model_path models/antibiotic_random_forest \
    --model_type random_forest \
    --fingerprint_type rdkit \
    --average_preds
```


## Generate molecules with SyntheMol

Here, we apply SyntheMol to generate molecules using a Monte Carlo tree search (MCTS). We limit the search space to molecules that can be formed using a single reaction (as opposed to multi-step reactions) to ensure easy, quick, and cheap chemical synthesis. We run SyntheMol three times, each time using a different property prediction model to score molecules and guide the search.

Chemprop
```bash
python -m SyntheMol.generate \
    --model_path models/antibiotic_chemprop \
    --model_type chemprop \
    --building_blocks_path data/4_real_space/building_blocks.csv \
    --save_dir data/5_generations_chemprop \
    --reaction_to_building_blocks_path data/4_real_space/reaction_to_building_blocks.pkl \
    --max_reactions 1 \
    --n_rollout 20000
```

Chemprop-RDKit
```bash
python -m SyntheMol.generate \
    --model_path models/antibiotic_chemprop_rdkit \
    --model_type chemprop \
    --building_blocks_path data/4_real_space/building_blocks.csv \
    --save_dir data/6_generations_chemprop_rdkit \
    --fingerprint_type rdkit \
    --reaction_to_building_blocks_path data/4_real_space/reaction_to_building_blocks.pkl \
    --max_reactions 1 \
    --n_rollout 20000
```

Random forest
```bash
python -m SyntheMol.generate \
    --model_path models/antibiotic_random_forest \
    --model_type random_forest \
    --building_blocks_path data/4_real_space/building_blocks.csv \
    --save_dir data/7_generations_random_forest \
    --fingerprint_type rdkit \
    --reaction_to_building_blocks_path data/4_real_space/reaction_to_building_blocks.pkl \
    --max_reactions 1 \
    --n_rollout 20000
```


## Filter generated molecules

Run three filtering steps to select molecules for experimental testing based on novelty, predicted bioactivity, and diversity.


### Novelty

Filter for molecules that are structurally novel, i.e., dissimilar from the training hits and the ChEMBL antibacterials.

First, Compute the nearest neighbor Tversky similarity between the generated molecules and the training hits or ChEMBL antibacterials to determine the novelty of each generated molecule. Tversky similarity is used to determine what proportion of the functional groups of known antibacterials are contained within the generated molecules.

Compute Tversky similarity to train hits.
```bash
#!/bin/bash

for NAME in 5_generations_chemprop 6_generations_chemprop_rdkit 7_generations_random_forest
do
chemfunc nearest_neighbor \
    --data_path data/${NAME}/molecules.csv \
    --reference_data_path data/1_training_data/antibiotics_hits.csv \
    --reference_name antibiotics_hits \
    --metrics tversky
done
```

Compute Tversky similarity to ChEMBL antibacterials.
```bash
#!/bin/bash

for NAME in 5_generations_chemprop 6_generations_chemprop_rdkit 7_generations_random_forest
do
chemfunc nearest_neighbor \
    --data_path data/${NAME}/molecules.csv \
    --reference_data_path data/2_chembl/chembl.csv \
    --reference_name chembl \
    --metrics tversky
done
```

Now, filter to only keep molecules with nearest neighbor Tverksy similarity to the train hits or ChEMBL antibacterials <= 0.5.

Filter by Tversky similarity to train hits.
```bash
#!/bin/bash

for NAME in 5_generations_chemprop 6_generations_chemprop_rdkit 7_generations_random_forest
do
chemfunc filter_molecules \
    --data_path data/${NAME}/molecules.csv \
    --save_path data/${NAME}/molecules_train_sim_below_0.5.csv \
    --filter_column antibiotics_hits_tversky_nearest_neighbor_similarity \
    --max_value 0.5
done
```

Filter by Tversky similarity to ChEMBL antibacterials.
```bash
#!/bin/bash

for NAME in 5_generations_chemprop 6_generations_chemprop_rdkit 7_generations_random_forest
do
chemfunc filter_molecules \
    --data_path data/${NAME}/molecules_train_sim_below_0.5.csv \
    --save_path data/${NAME}/molecules_train_sim_below_0.5_chembl_sim_below_0.5.csv \
    --filter_column chembl_tversky_nearest_neighbor_similarity \
    --max_value 0.5
done
```


### Bioactivity

Filter for high-scoring molecules (i.e., predicted bioactivity) by only keeping molecules with the top 20% of model scores.
```bash
#!/bin/bash

for NAME in 5_generations_chemprop 6_generations_chemprop_rdkit 7_generations_random_forest
do
chemfunc filter_molecules \
    --data_path data/${NAME}/molecules_train_sim_below_0.5_chembl_sim_below_0.5.csv \
    --save_path data/${NAME}/molecules_train_sim_below_0.5_chembl_sim_below_0.5_top_20_percent.csv \
    --filter_column score \
    --top_proportion 0.2
done
```


### Diversity

Filter for diverse molecules by clustering molecules based on their Morgan fingerprint and only keeping the top scoring molecule from each cluster.

Cluster molecules based on their Morgan fingerprint.
```bash
#!/bin/bash

for NAME in 5_generations_chemprop 6_generations_chemprop_rdkit 7_generations_random_forest
do
chemfunc cluster_molecules \
    --data_path data/${NAME}/molecules_train_sim_below_0.5_chembl_sim_below_0.5_top_20_percent.csv \
    --num_clusters 50
done
```

Select the top scoring molecule from each cluster.
```bash
#!/bin/bash

for NAME in 5_generations_chemprop 6_generations_chemprop_rdkit 7_generations_random_forest
do
chemfunc select_from_clusters \
    --data_path data/${NAME}/molecules_train_sim_below_0.5_chembl_sim_below_0.5_top_20_percent.csv \
    --save_path data/${NAME}/molecules_train_sim_below_0.5_chembl_sim_below_0.5_top_20_percent_selected_50.csv \
    --value_column score
done
```

We now have 50 molecules selected from each model's generations that meet our desired novelty, score, and diversity criteria.


## Map molecules to REAL IDs

TODO: change to use all possible IDs for each SMILES

Map generated molecules to REAL IDs in the format expected by Enamine to enable a lookup in the Enamine REAL Space database.
```bash
#!/bin/bash

for NAME in 5_generations_chemprop 6_generations_chemprop_rdkit 7_generations_random_forest
do
python -m SyntheMol.data.map_generated_molecules_to_real_ids \
    --data_path data/${NAME}/molecules_train_sim_below_0.5_chembl_sim_below_0.5_top_20_percent_selected_50.csv \
    --save_dir data/${NAME}/molecules_train_sim_below_0.5_chembl_sim_below_0.5_top_20_percent_selected_50_real_ids
done
```


## Predict toxicity

Predict the toxicity of the synthesized molecules by training a toxicity prediction model on the ClinTox dataset from [MoleculeNet](https://moleculenet.org/). ClinTox is a clinical trial toxicity binary classification dataset with 1,478 molecules with 112 (7.58%) toxic.

Train toxicity model using the `CT_TOX` property of the ClinTox dataset. (Note that the ClinTox properties `CT_TOX` and `FDA_APPROVED` are nearly always identical, so we only use one.)
```bash
python -m SyntheMol.models.train \
    --data_path data/1_training_data/clintox.csv \
    --save_dir models/clintox_chemprop_rdkit \
    --model_type chemprop \
    --dataset_type classification \
    --fingerprint_type rdkit \
    --property_column CT_TOX \
    --num_models 10
```

The model has an ROC-AUC of 0.881 +/- 0.045 and a PRC-AUC of 0.514 +/- 0.141 across 10-fold cross-validation.

Make toxicity predictions on the synthesized molecules. The list of successfully synthesized generated molecules is in the `Data/8. Synthesized` subfolder of the Google Drive folder.
```bash
python -m SyntheMol.models.predict \
    --data_path data/8_synthesized/synthesized.csv \
    --model_path models/clintox_chemprop_rdkit \
    --model_type chemprop \
    --fingerprint_type rdkit \
    --average_preds
```