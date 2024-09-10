import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os
import pickle

def load_models():
    model_names = ["RidgeClassifierCV", "SVC_linear", "SVC_rbf", "KNN_dist", "KNN_uniform",
                   "RandomForest", "GaussianProcess", "AdaBoost", "MLP", "GradientBoosting"]
    models = {}
    for name in model_names:
        model_path = f'{name}_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                models[name] = pickle.load(file)
        else:
            print(f"Model file {model_path} not found. Please check the directory.")
    return models

def predict_new_compounds(models, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        X_new = np.array([fp])
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X_new)[0]  # Using predict instead of predict_proba
        predictions['Average'] = np.mean([float(val) for val in predictions.values()])
        return canonical_smiles, predictions
    else:
        return None, "Invalid SMILES string"

def find_compound_in_excel(canonical_input_smiles, filepath):
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        # Normalize the SMILES in the DataFrame
        df['Canonical SMILES'] = df['SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True) if pd.notna(x) else None)

        # Find the row with the matching canonical SMILES
        matches = df[df['Canonical SMILES'] == canonical_input_smiles]
        if not matches.empty:
            return matches[['Item Description', 'CAS']].iloc[0]
        else:
            return "No matching compound found in the Excel file."
    except FileNotFoundError:
        return "Excel file not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def display_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        img.show()
    else:
        print("Invalid SMILES string.")

def main():
    print("Welcome to the SMILES compound analyzer!")
    smiles_input = input("Please enter a SMILES string: ")

    models = load_models()
    canonical_smiles, predictions = predict_new_compounds(models, smiles_input)
    if isinstance(predictions, dict):
        print(f"Canonical SMILES: {canonical_smiles}")
        print("Prediction scores:")
        for model, score in predictions.items():
            print(f"{model}: {score}")
        #display_molecule(canonical_smiles)
    else:
        print(predictions)

    compound_info = find_compound_in_excel(canonical_smiles, 'screening_chemscrene_50USD_MW500.xlsx')
    if isinstance(compound_info, pd.Series):
        #excel_info_label.config(text=f"Found in Excel: Description - {compound_info['Item Description']}, CAS - {compound_info['CAS']}")
        print(f"Found in Excel: Description - {compound_info['Item Description']}, CAS - {compound_info['CAS']}")
    else:
        excel_info_label.config(text=compound_info)

if __name__ == "__main__":
    main()
