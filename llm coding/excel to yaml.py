import pandas as pd
import yaml
from string import ascii_uppercase
from datetime import datetime

# Function to load data from an Excel file
def load_excel_data(filename):
    return pd.read_excel(filename)

# Function to format molecular amounts to a specified precision
def format_mol_amount(mol):
    return round(mol, 8)  # Adjust the precision to 8 decimal places or as needed

# Function to create the YAML content from Excel data
def create_yaml_content(excel_data):
    # Define the size and dimensions of the reaction plate
    n_rows, n_cols, n_wells = 4, 6, 24
    col, row = 1, 1

    # Initialize the base structure of the YAML content
    yaml_dict = {
        'QUEUE_STRING': {
            'containers': {
                'reaction_plate': {
                    'container_name': None,
                    'contents': {},
                    'plate_type': '24 Well Electrochem Plate'
                }
            },
            'date_created': datetime.now().strftime('%m/%d/%Y'),
            'dependency': [],
            'operations': {
                '1': {
                    'agent': 'Lh',
                    'completed': 'no',
                    'container': 'reaction_plate',
                    'details': {
                        'empty_wells': True,
                        'preparation_stages': [],
                        'tip_prep_rinse': True,
                        'washing_frequency': None,
                        'single_transfer': True,
                        'liquid_class': 'Adjusted water free dispense breakoff echem'          

                    },
                    'end_time': None,
                    'operation': 'prepare_wellplate',
                    'start_time': None,
                    'time_est': 3600
                },
                '2': {
                    'agent': 'MC',
                    'completed': 'no',
                    'container': '',
                    'details': {},
                    'end_time': None,
                    'operation': 'complete_queue',
                    'start_time': None,
                    'time_est': 30
                }
            },
            'queue_name': f'echem_{datetime.now().strftime("%Y%m%d")}',
            'status': 'idle'
        }
    }

    # Populate the contents based on Excel data
    for i in range(n_wells):
        well_id = f"{ascii_uppercase[row - 1]}{col}"
        if row == n_rows:
            row = 1
            col += 1
        else:
            row += 1

        if i < len(excel_data):
            row_data = excel_data.iloc[i]
            reagents = []
            for j in range(1, 5):
                reagent_name = row_data.get(f'reagent_{j}')
                reagent_mol = row_data.get(f'reagent_{j} mol')
                if pd.notna(reagent_name) and reagent_mol != 0:
                    reagent_mol = format_mol_amount(float(reagent_mol))
                    reagent_entry = [reagent_name, reagent_mol]
                    if reagent_name.lower() == 'hfip':
                        reagent_entry.append(2)
                    reagents.append(reagent_entry)

            well_data = {
                'confirmation': 'none',
                'final_product': [0, 'no', '', 1],
                'reaction_smiles': '',
                'reagents': reagents,
                'solvents': [
                    [row_data['solvent_1'], int(row_data['solvent_1 frac']), 1]
                ],
                'target_product': None,
                'templates': [],
                'total_volume': int(row_data['total_vol'])
            }
            yaml_dict['QUEUE_STRING']['containers']['reaction_plate']['contents'][well_id] = well_data

    return yaml_dict

# Function to save the constructed YAML to a file
def save_yaml(yaml_dict, output_filename):
    with open(output_filename, 'w') as file:
        yaml.dump(yaml_dict, file, default_flow_style=False, sort_keys=False)

# Load the Excel data
excel_data = load_excel_data('Echem_Auto_Plan.xlsx')

# Create the YAML content directly from the Excel data
final_yaml_dict = create_yaml_content(excel_data)

# Save the YAML output to a file named 'Echem_Queue.yaml'
save_yaml(final_yaml_dict, 'Echem_Queue.yaml')
