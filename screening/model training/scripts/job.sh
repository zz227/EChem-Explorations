#!/bin/bash

chemprop_dir=~/package/chemprop  # location of chemprop directory, CHANGE ME

# Reactivity
for split in random scaffold_balanced
do
    # Random Forest
    results_dir=reactivity/$split/random_forest
    python $chemprop_dir/sklearn_train.py \
    --data_path ../data/data.csv \
    --smiles_columns smiles \
    --target_columns Reactivity \
    --split_type $split \
    --dataset_type classification \
    --adding_h \
    --num_folds 5 \
    --save_smiles_splits \
    --save_dir $results_dir \
    --model_type random_forest

    # D-MPNN
    results_dir=reactivity/$split/d_mpnn
    python $chemprop_dir/train.py \
    --data_path ../data/data.csv \
    --smiles_columns smiles \
    --target_columns Reactivity \
    --split_type $split \
    --dataset_type classification \
    --epochs 200 \
    --aggregation norm \
    --adding_h \
    --ensemble_size 5 \
    --num_folds 5 \
    --save_smiles_splits \
    --save_preds \
    --save_dir $results_dir

    # D-MPNN + QM
    results_dir=reactivity/$split/d_mpnn_qm_des
    python $chemprop_dir/train.py \
    --data_path ../data/data.csv \
    --atom_descriptors_path ../data/atom_qm_des.pkl \
    --atom_descriptors feature \
    --bond_descriptors_path ../data/bond_qm_des.pkl \
    --bond_descriptors feature \
    --features_path ../data/molecule_qm_des.csv \
    --smiles_columns smiles \
    --target_columns Reactivity \
    --split_type $split \
    --dataset_type classification \
    --epochs 200 \
    --aggregation norm \
    --adding_h \
    --ensemble_size 5 \
    --num_folds 5 \
    --save_smiles_splits \
    --save_preds \
    --save_dir $results_dir
done

# Site-selectivity
for split in random scaffold_balanced
do
    # D-MPNN
    results_dir=site_selectivity/$split/default
    python $chemprop_dir/train.py \
    --data_path ../data/data.csv \
    --smiles_columns smiles \
    --target_columns "Oxidation Site (one hot)" \
    --split_type $split \
    --dataset_type classification \
    --epochs 200 \
    --aggregation norm \
    --ensemble_size 5 \
    --num_folds 5 \
    --save_smiles_splits \
    --save_preds \
    --save_dir $results_dir \
    --is_atom_bond_targets

    # D-MPNN + QM
    results_dir=site_selectivity/$split/default_qm_des
    python $chemprop_dir/train.py \
    --data_path ../data/data.csv \
    --atom_descriptors feature \
    --atom_descriptors_path ../data/atom_qm_des_no_hs.pkl \
    --bond_descriptors feature \
    --bond_descriptors_path ../data/bond_qm_des_no_hs.pkl \
    --smiles_columns smiles \
    --target_columns "Oxidation Site (one hot)" \
    --split_type $split \
    --dataset_type classification \
    --epochs 200 \
    --aggregation norm \
    --ensemble_size 5 \
    --num_folds 5 \
    --save_smiles_splits \
    --save_preds \
    --save_dir $results_dir \
    --is_atom_bond_targets
done