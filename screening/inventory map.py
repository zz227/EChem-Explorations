import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem



def load_and_process_data(file_path_train, file_path_screening, threshold_price=100, threshold_mwt=1000):
    # Load train dataset
    train = pd.read_excel(file_path_train)
    train['Canonical SMILES'] = train['Canonical SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notnull(x) else None)
    train.dropna(subset=['Canonical SMILES'], inplace=True)
    train_blue = train[train['Reactivity'] == 1]
    train_red = train[train['Reactivity'] == -1]
    
    # Load screening dataset
    screening = pd.read_excel(file_path_screening)
    # Apply both price and molecular weight thresholds
    screening_filtered = screening[(screening['Cheapest Unit Price (USD per g/mL)'] < threshold_price) & 
                                   (screening['MWt'] < threshold_mwt)].copy()
    screening_filtered['Canonical SMILES'] = screening_filtered['SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notnull(x) else None)
    screening_filtered.dropna(subset=['Canonical SMILES'], inplace=True)
    screening_grey = screening_filtered
    
    return train_blue, train_red, screening_grey


def generate_fingerprints_and_reduce_dimensions(dfs):
    combined_smiles = pd.concat([df['Canonical SMILES'] for df in dfs if not df.empty])
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2, nBits=1024) for smiles in combined_smiles]
    fingerprints_array = np.array([list(fp) for fp in fingerprints])
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(fingerprints_array)
    
    # t-SNE with explicit parameter settings to avoid FutureWarnings
    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto')
    tsne_result = tsne.fit_transform(fingerprints_array)
    
    return pca_result, tsne_result, combined_smiles.index

def plot_combined_data(pca_result, tsne_result, indices, colors, labels):
    plt.figure(figsize=(12, 6))
    
    # Adjust the color list to include the RGBA tuple for grey with 20% transparency
    colors = ['blue', 'red', (0.5, 0.5, 0.5, 0.02)]  # Adjusting the grey color for 2% transparency
    
    # Plot PCA results
    plt.subplot(1, 2, 1)
    # Ensure grey points are plotted first for them to be in the background
    for i in [2, 0, 1]:  # Reordering to plot grey first, followed by blue and red
        idx = indices[i]
        plt.scatter(pca_result[idx, 0], pca_result[idx, 1], c=colors[i], label=labels[i])
    plt.title('PCA of Molecular Fingerprints')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    # Plot t-SNE results
    plt.subplot(1, 2, 2)
    # Same reordering logic for t-SNE plot
    for i in [2, 0, 1]:  # Reordering to plot grey first, followed by blue and red
        idx = indices[i]
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], c=colors[i], label=labels[i])
    plt.title('t-SNE of Molecular Fingerprints')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main(file_path_train, file_path_screening, price=100, mwt=1000):
    train_blue, train_red, screening_grey = load_and_process_data(file_path_train, file_path_screening, price, mwt)
    pca_result, tsne_result, _ = generate_fingerprints_and_reduce_dimensions([train_blue, train_red, screening_grey])
    
    # Calculate indices for coloring
    len_blue = len(train_blue)
    len_red = len(train_red)
    len_grey = len(screening_grey)
    indices = [range(len_blue), range(len_blue, len_blue + len_red), range(len_blue + len_red, len_blue + len_red + len_grey)]
    
    print(f"Number of blue points: {len_blue}")
    print(f"Number of red points: {len_red}")
    print(f"Number of grey points: {len_grey}")
    
    colors = ['blue', 'red', 'grey']  # Blue for Reactivity 1, Red for Reactivity -1, Grey for Screening
    labels = ['Train Reactivity 1', 'Train Reactivity -1', 'Screening']
    plot_combined_data(pca_result, tsne_result, indices, colors, labels)
    return screening_grey
    
file_name = "Inventory-2024-03-30.xlsx"    
processed_file_name = 'processed_' + file_name    

# Example usage
screening_df = main('SF2. EChem Reaction Screening Dataset.xlsx', processed_file_name,50, 500)
screening_df.to_excel(processed_file_name, index=False) 
print("Processing complete. The new file is saved as:", processed_file_name)
print(screening_df50_500)