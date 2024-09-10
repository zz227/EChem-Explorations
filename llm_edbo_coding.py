import re
import tempfile
import os
import shutil
import anthropic
import openai
from openai import OpenAI
import pandas as pd
import time
import traceback
import glob
import numpy as np
from IPython.utils.io import capture_output
import subprocess
import pandas as pd
import openpyxl
import replicate
import tempfile
import traceback
import time


def get_model_and_api_keys():
    """
    Prompts the user to select an LLM model and then prompts for the corresponding API keys based on the chosen model.

    Returns:
        tuple: A tuple containing the selected model and a dictionary of API keys.
    """
    print("Please enter the LLM model to be used (e.g., 'gpt-3.5-turbo', 'claude-3', 'llama-2'):")
    model = input().strip().lower()

    api_keys = {}
    if 'gpt' in model:
        api_keys['openai_key'] = input("Please provide your OpenAI API key: ").strip()
    if 'claude' in model:
        api_keys['anthropic_key'] = input("Please provide your Anthropic API key: ").strip()
    if 'llama' in model:
        api_keys['replicate_key'] = input("Please provide your Replicate API key: ").strip()

    return model, api_keys


def generate_and_execute_code(user_prompts, model='gpt-3.5-turbo', num_calls=2, max_reflection=0):
    """
    Generate code using the specified model, execute it, and store the results in an Excel file.

    Args:
        user_prompts (list): A list of prompts for generating the Python code.
        model (str): The model to use for code generation. Options are:
            - 'claude-3-opus-20240229'
            - 'claude-3-sonnet-20240229'
            - 'gpt-4-turbo-2024-04-09'
            - 'gpt-4-0613'
            - 'gpt-3.5-turbo' (default)
        num_calls (int): The number of times to generate and execute code (default is 2).
        max_reflection (int): The maximum number of reflection attempts when code execution fails (default is 0).

    Returns:
        pandas.DataFrame: A DataFrame containing the generated code, execution results, response times, number of reflections, and conversation history.
    """
    # Prompt for API keys based on the selected model
    openai_key = anthropic_key = replicate_key = None

    # Load keys from environment or prompt user
    if 'openai_key' in api_keys:
        openai.api_key = api_keys['openai_key']
    if 'anthropic_key' in api_keys:
        anthropic_key = api_keys['anthropic_key']
    if 'replicate_key' in api_keys:
        os.environ["REPLICATE_API_TOKEN"] = api_keys['replicate_key']

    results = []
    for _ in range(num_calls):
        
        print(_)
        # Delete all previous CSV and pkl files
        for file_path in glob.glob('*.csv') + glob.glob('*.pkl'):
            os.remove(file_path)
        
        conversation_history = []
        row_data = {'Model': model}
        execution_result, num_reflections = 0, 0
        for i in range(len(user_prompts)):
            if i > 0 and execution_result != 1:
                row_data[f'Generated Code {i+1}'] = 'N/A'
                row_data[f'Execution Result {i+1}'] = 'N/A'
                row_data[f'Response Time {i+1} (s)'] = 'N/A'
                row_data[f'Number of Reflections {i+1}'] = 'N/A'
                continue
            
            prompt_conversation_history = conversation_history.copy()

            response_content, response_time = chat(model, user_prompts[i], prompt_conversation_history)
            conversation_history.append({"role": "user", "content": user_prompts[i]})
            conversation_history.append({"role": "assistant", "content": response_content})

            code_block = re.search(r'```python\n(.*?)\n```', response_content, re.DOTALL)
            if not code_block:
                code_block = re.search(r'```python(.*?)```', response_content, re.DOTALL)
                
            if not code_block:
                code_block = re.search(r'python\n(.*?)\n', response_content, re.DOTALL)
                
            if not code_block:
                code_block = re.search(r'```\n(.*?)\n```', response_content, re.DOTALL)
            if not code_block:
                # Fall back to generic triple backtick extraction
                code_block = re.search(r'```(.*?)```', response_content, re.DOTALL)
                
            if code_block:
                code = code_block.group(1)
            else:
                code = response_content
                print(f"Warning: No code block found in the response for prompt {i+1}. Attempting to execute the entire response.")

            for j in range(max_reflection + 1):
                execution_result, error_message = run_code(code)
                if execution_result == 1:
                    break
                if j < max_reflection:
                    reflection_prompt = f"Please reflect on the code you previously wrote. There is an error and I cannot run it on my Jupyter Notebook. The error message is:\n{error_message}\nPlease try to catch any bugs or failures to follow the user instruction. In your answer, give me the full revised code. Do not just give the revised part, but the whole code that can be directly copy and paste to run. Make sure you give full code in the python code block, do not miss, comment or abbreviation anything."
                    print(reflection_prompt)
                    prompt_conversation_history = conversation_history.copy()
                    revised_response_content, revised_response_time = chat(model, reflection_prompt, prompt_conversation_history)
                    conversation_history.append({"role": "user", "content": reflection_prompt})                    
                    conversation_history.append({"role": "assistant", "content": revised_response_content})

                    revised_code_block = re.search(r'```python\n(.*?)\n```', revised_response_content, re.DOTALL)

                    if revised_code_block:
                        code = revised_code_block.group(1)

                    response_time += revised_response_time
                    num_reflections += 1

            row_data[f'Generated Code {i+1}'] = code
            row_data[f'Execution Result {i+1}'] = execution_result
            row_data[f'Response Time {i+1} (s)'] = response_time
            row_data[f'Number of Reflections {i+1}'] = num_reflections


        row_data['Conversation History'] = str(conversation_history)
        results.append(row_data)

    df = pd.DataFrame(results)
    df.to_excel('Results_'+model+'.xlsx', index=False)

    return df



def run_code(code):
    """
    Run the provided code in a separate Python process and return the execution result and error message (if any).

    Args:
        code (str): The code to be executed.

    Returns:
        tuple: A tuple containing the execution result (0 for failure, 1 for success) and the error message (if any).
    """


    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, timeout=100)
        os.unlink(temp_file_path)

        if result.returncode == 0:
            return 1, None
        else:
            return 0, result.stderr
    except subprocess.TimeoutExpired:
        print("Error: Code execution exceeded time limit and was terminated.")
        os.unlink(temp_file_path)
        return 0, "Code execution exceeded time limit and was terminated."
    
    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        os.unlink(temp_file_path)
        return 0, error_message



def chat(model, user_prompt, conversation_history, max_retries=3):
    """
    Helper function to chat with the specified model and handle retries.

    Args:
        model (str): The model to use for code generation.
        user_prompt (str): The prompt for generating the Python code.
        conversation_history (list): The history of the conversation.
        max_retries (int): The maximum number of retries if an error occurs (default is 3).

    Returns:
        tuple: A tuple containing the response content and response time.
    """

    retry_count = 0
    pre_prompt= "You are a helpful coding assistant who always writes detailed and executable code without human implementation. Please ensure that you write the complete code so I can copy and paste it directly into a Jupyter notebook to run. Please write all codes in one code block; do not separate them by text explainations. When explanations are necessary, include them as comments in the code. Make sure you use ```python to mark the start of the python code. "
    while retry_count < max_retries:
        try:
            start_time = time.time()
            if model.startswith('claude'):
                client = anthropic.Anthropic(api_key=anthropic_api_key)
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[
                        *conversation_history,
                        {"role": "user", "content": user_prompt}
                    ]
                )
                response_content = response.content[0].text

            elif model.startswith('gpt'):
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": pre_prompt},
                        *conversation_history,
                        {"role": "user", "content": user_prompt}
                    ]
                )
                response_content = response.choices[0].message.content

            elif model.startswith('llama'): 
                # Format the conversation history for Replicate's Llama model
                formatted_history = ""
                for message in conversation_history:
                    role = message["role"]
                    if role == "user":
                        formatted_history += f"User: {message['content']}\n"
                    elif role == "assistant":
                        formatted_history += f"Assistant: {message['content']}\n"
                formatted_prompt = f"{formatted_history}User: {user_prompt}\nAssistant:"
                if model == "llama-3.1":
                    model_name = "meta/meta-llama-3.1-405b-instruct"                
                elif model == "llama-3":
                    model_name = "meta/meta-llama-3-70b-instruct"
                elif model == "llama-2-code":
                    model_name = "meta/codellama-70b-instruct:a279116fe47a0f65701a8817188601e2fe8f4b9e04a518789655ea7b995851bf"
                else:
                    model_name = "meta/codellama-34b-instruct:eeb928567781f4e90d2aba57a51baef235de53f907c214a4ab42adabf5bb9736"
                response = replicate.run(
                    model_name,
                    input={
                        "system_prompt":  pre_prompt,
                        #"max_tokens": 4096,
                        "prompt": formatted_prompt
                    }
                )
                response_content = ''.join(response)

                
            else:
                raise ValueError(f"Unknown model: {model}")
            
            end_time = time.time()
            response_time = end_time - start_time
            
            conversation_history.append({"role": "assistant", "content": response_content})
        
            return response_content, response_time
        except Exception as e:
            retry_count += 1
            
            if retry_count == max_retries:
                error_message = f"Error occurred during code generation: {str(e)}"

                return error_message, 0



def check_csv_conditions(file_names):
    for file_name in file_names:
        try:
            df = pd.read_csv(file_name)
            # Convert all column names to lower case to handle case insensitivity
            df.columns = [col.lower() for col in df.columns]
            
            if "priority" not in df.columns or "yield" not in df.columns:
                print(f"{file_name} does not pass: 'priority' or 'yield' column missing")
                continue  # Skip to the next file

            # Convert 'yield' values to string for a consistent comparison across types
            negative_priority_yields = df[df['priority'] == -1]['yield'].astype(str).tolist()
            valid_yields = ['6', '5', '8', '0']

            if not all(yield_value in valid_yields for yield_value in negative_priority_yields):
                print(f"{file_name} does not pass: Incorrect 'yield' values for priority -1. {negative_priority_yields }")
                continue  # Skip to the next file


            # Define acceptable values for each column as floats
            substrate_concentrations = [0.025, 0.05, 0.075, 0.1, 0.125]
            mediator_eqs = [0, 0.25, 0.5, 0.75, 1]
            mediator_types = ["NHPI", "TCNHPI", "QD", "DABCO", "TEMPO"]
            electrolyte_types = ["LiClO4", "LiOTf", "Bu4NClO4", "Et4NBF4", "Bu4NPF6"]
            co_solvents = [0, 1]
            
            # Check the first five rows for columns where priority is 1
            priority_one_rows = df[df['priority'] == 1].head(5)
            
            # Iterating through each row to check conditions
            for _, row in priority_one_rows.iterrows():
                # Convert numeric column values to float for comparison
                try:
                    row_concentration = float(row[0])
                    row_eqs = float(row[1])
                    row_co_solvent = float(row[4])
                except ValueError:
                    print(f"{file_name} does not pass: Numeric conversion error in data. row_concentration = {row[0]}; row_eqs = {row[1]}; row_co_solvent = {row[4]}")
                    continue

                # Check each column against its respective allowed values
                if row_concentration not in substrate_concentrations:
                    print(f"{file_name} does not pass: Incorrect substrate concentration {row[0]} in the first column")
                    continue  # Skip to the next file
                if row_eqs not in mediator_eqs:
                    print(f"{file_name} does not pass: Incorrect mediator equivalents {row[1]} in the second column")
                    continue  # Skip to the next file
                if row[2] not in mediator_types:
                    print(f"{file_name} does not pass: Incorrect mediator type {row[2]} in the third column")
                    continue  # Skip to the next file
                if row[3] not in electrolyte_types:
                    print(f"{file_name} does not pass: Incorrect electrolyte type {row[3]} in the fourth column")
                    continue  # Skip to the next file
                if row_co_solvent not in co_solvents:
                    print(f"{file_name} does not pass: Incorrect co-solvent {row[4]} in the fifth column")
                    continue  # Skip to the next file

            print(f"{file_name} passes all tests")
            return True

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return False  # If no files pass the test, return False



def process_excel_files(file_names):
    summary_data = []
    
    for file_name in file_names:
        output_file_name = "acc_" + file_name
        if os.path.isfile(output_file_name):
            df = pd.read_excel(output_file_name)
        else:
            df = pd.read_excel(file_name)
            df["Performance"] = False
            
            total_rows = len(df)
            print(f"Processing {file_name} with {total_rows} rows")
            
            
            for index, row in df.iterrows():
                print(f"Processing row {index + 1}/{total_rows}")
                if row["Execution Result 1"] == 1 and row["Execution Result 2"] == 1:
                    combined_code = row['Generated Code 1'] + "\n" + row['Generated Code 2']
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
                        temp_file.write(combined_code)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Clean CSV files in folder before executing the code
                        for file in os.listdir('.'):
                            if file.endswith('.csv'):
                                os.remove(file)
                        
                        result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, timeout=100)
                        csv_files_after = set(os.listdir('.'))
                        generated_csv_files = [file for file in csv_files_after if file.endswith('.csv')]
                        print(generated_csv_files)
                        if check_csv_conditions(generated_csv_files):
                            df.at[index, 'Performance'] = True
                        else:
                            print(f"Error executing code at index {index} in file {file_name}: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        print(f"Error: Code execution at index {index} in file {file_name} exceeded time limit and was terminated.")
                        
                    finally:
                        os.unlink(temp_file_path)
            
            # Calculate 'Adjusted Time' for the entire DataFrame
            df["Total Time"] = df["Response Time 1 (s)"] + df["Response Time 2 (s)"]
            df["Adjusted Time"] = df["Total Time"] / (df["Number of Reflections 1"] + df["Number of Reflections 2"] + 2)
            df.to_excel(output_file_name, index=False)
        
        # Calculate summaries
        total_rows = len(df)
        executed_rows = len(df[(df["Execution Result 1"] == 1) & (df["Execution Result 2"] == 1)])
        valid_rows = df['Performance'].sum()
        correctness_metric = valid_rows / total_rows
        one_conversation_rows = len(df[(df["Execution Result 1"] == 1) & (df["Number of Reflections 1"] == 0) & (df["Execution Result 2"] == 1) & (df["Number of Reflections 2"] == 0)])
        
        summary_row = {
            "Model Name": file_name.split("_")[1],
            "Avg Time": df["Adjusted Time"].mean(),
            "Code Length": df.apply(lambda x: len(str(x['Generated Code 1']).split()) + len(str(x['Generated Code 2']).split()), axis=1).mean(),
            "Code Executability (one conversation)": one_conversation_rows / total_rows,
            "Code Executability (with reflection)": executed_rows / total_rows,
            "Correctness": correctness_metric
        }
        
        summary_data.append(summary_row)
    

    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel("summary.xlsx", index=False)
    
    return "summary.xlsx"


def process_and_summarize_results():
    """
    This function processes all generated Excel files (starting with "Results_") in the current folder, 
    calculates performance metrics for each file, and summarizes the results in a new Excel file.
    
    The function evaluates the accuracy of machine learning models by reading the generated code from the 
    "Generated Code 1" column, executing it, and extracting the accuracy from the output. The performance 
    and other statistics are summarized and saved to an "acc_" prefixed Excel file for each individual 
    result and a "summary.xlsx" file for the overall summary.
    """

    # Automatically collect all "Results_xxxxxxx.xlsx" files in the current directory
    file_names = glob.glob("Results_*.xlsx")

    # Ensure there are files to process
    if not file_names:
        print("No 'Results_.xlsx' files found for processing.")
        return
    
    # Call the existing process_excel_files function to process and evaluate each file
    summary_file = process_excel_files(file_names)
    
    print(f"Processing complete. Summary file generated: {summary_file}")
    
# Main Execution
if __name__ == "__main__":
    # Ask for model and API keys
    model, api_keys = get_model_and_api_keys()

    # Example prompts for ML code generation
    user_prompts = [
    """Below is an example for edbo package (EDBO+. Bayesian reaction optimization as a tool for chemical synthesis) from https://github.com/doyle-lab-ucla/edboplus. Please carefully read it
```
from edbo.plus.optimizer_botorch import EDBOplus

#### This is a tutorial that covers the basics for running EDBO+: from designing a combinatorial space to running the Bayesian Optimizer.

## 1. Creating a search scope using EDBO+.


##### To run EDBO+ we need to first create a reaction scope (search space) in a .csv format with all the possible combinations that you want to consider for our optimization. 
##### You can "manually" create a reaction scope using any spreadsheet editor (such as Excel, Libreoffice Calc, ...) but we have also created a tool to help you generating combinatorial search spaces. 
##### For instance, lets say that we want to consider 3 solvents ($\bf{THF}$, $\bf{Toluene}$, $\bf{DMSO}$), 4 temperatures ($\bf{-10}$, $\bf{0}$, $\bf{10}$, $\bf{25}$) and 3 concentration levels ($\bf{0.1}$, $\bf{0.2}$, $\bf{1.0}$). We can introduce these in the EDBO+ scope generator in a dictionary form as follows:


reaction_components = {
    'solvent': ['THF', 'Toluene', 'DMSO'],
    'T': [-10, 0, 10, 25],
    'concentration': [0.1, 0.2, 1.0]
}


##### Now we need to pass the previous dictionary to the $\bf{generate\_reaction\_scope}$ tool in the EDBOplus class.

EDBOplus().generate_reaction_scope(
    components=reaction_components, 
    filename='my_optimization.csv',
    check_overwrite=False
)

Generating reaction scope...
solvent	T	concentration
0	THF	-10	0.1
1	THF	-10	0.2
2	THF	-10	1.0
3	THF	0	0.1
4	THF	0	0.2
5	THF	0	1.0
6	THF	10	0.1
7	THF	10	0.2
8	THF	10	1.0
9	THF	25	0.1
10	THF	25	0.2
11	THF	25	1.0
12	Toluene	-10	0.1
13	Toluene	-10	0.2
14	Toluene	-10	1.0
15	Toluene	0	0.1
16	Toluene	0	0.2
17	Toluene	0	1.0
18	Toluene	10	0.1
19	Toluene	10	0.2
20	Toluene	10	1.0
21	Toluene	25	0.1
22	Toluene	25	0.2
23	Toluene	25	1.0
24	DMSO	-10	0.1
25	DMSO	-10	0.2
26	DMSO	-10	1.0
27	DMSO	0	0.1
28	DMSO	0	0.2
29	DMSO	0	1.0
30	DMSO	10	0.1
31	DMSO	10	0.2
32	DMSO	10	1.0
33	DMSO	25	0.1
34	DMSO	25	0.2
35	DMSO	25	1.0

##### We can always load/read the previously generated reaction scope using any spreadsheet editor but in this case we will use Pandas for that:

import pandas as pd
df_scope = pd.read_csv('my_optimization.csv')  # Load csv file.

##### Now we can check the number of combinations in the reaction scope:

n_combinations = len(df_scope)
print(f"Your reaction scope has {n_combinations} combinations.")


Your reaction scope has 36 combinations.

##### Of course, this is a very small reaction scope  as it is meant to be a toy model to demonstrate how EDBO+ works.

## 2. First steps, initializing EDBO+ (in absence of training data).

##### We are going to execute EDBO+ to suggest some initial samples. 
##### Since we have not collected any experimental data (observations) yet, EDBO+ will suggest a set of initial experiments based on an feature space sampling method, in this case the CVT sampling method (see:http://kmh-lanl.hansonhub.com/uncertainty/meetings/gunz03vgr.pdf).
##### In this example the $\bf{objective}$ is to maximize the reaction $\bf{yield}$ and $\bf{enantioselectivity}$ but at the same time we want to minimize the amount one of a given $\bf{side~product}$ in this reaction. We also need to introduce the name of the csv file containing our reaction scope (in our case this was $\bf{my\_optimization.csv}$). Now we can execute the algorithm using the $\bf{run}$ command in EDBOplus:


EDBOplus().run(
    filename='my_optimization.csv',  # Previously generated scope.
    objectives=['yield', 'ee', 'side_product'],  # Objectives to be optimized.
    objective_mode=['max', 'max', 'min'],  # Maximize yield and ee but minimize side_product.
    batch=3,  # Number of experiments in parallel that we want to perform in this round.
    columns_features='all', # features to be included in the model.
    init_sampling_method='cvtsampling'  # initialization method.
)



The following columns are categorical and will be encoded using One-Hot-Encoding: ['solvent']
Sampling type:  selection 


Number of unique samples returned by sampling algorithm: 3
Creating a priority list using random sampling: cvtsampling

	solvent	T	concentration	yield	ee	side_product	priority
32	DMSO	10	1.0	PENDING	PENDING	PENDING	1
8	THF	10	1.0	PENDING	PENDING	PENDING	1
19	Toluene	10	0.2	PENDING	PENDING	PENDING	1
0	THF	-10	0.1	PENDING	PENDING	PENDING	0
26	DMSO	-10	1.0	PENDING	PENDING	PENDING	0
21	Toluene	25	0.1	PENDING	PENDING	PENDING	0
22	Toluene	25	0.2	PENDING	PENDING	PENDING	0
23	Toluene	25	1.0	PENDING	PENDING	PENDING	0
24	DMSO	-10	0.1	PENDING	PENDING	PENDING	0
25	DMSO	-10	0.2	PENDING	PENDING	PENDING	0


##### EDBO+ has created a column for each objective and added $\bf{PENDING}$ values to all of them so you can track the experiments that you have been collecting during the optimization campaign.
##### We can also see that EDBO+ has created a new $\bf{priority}$ column. This column is used to distinguish between high and low priority samples. The top entries (with $\bf{priority=1}$) highlight the next suggested samples.


##### We can check now the first 5 experiments in the scope by reading the $\bf{my\_optimization.csv}$ file:


df_edbo = pd.read_csv('my_optimization.csv')
df_edbo.head(5)

solvent	T	concentration	yield	ee	side_product	priority
0	DMSO	10	1.0	PENDING	PENDING	PENDING	1
1	THF	10	1.0	PENDING	PENDING	PENDING	1
2	Toluene	10	0.2	PENDING	PENDING	PENDING	1
3	THF	-10	0.1	PENDING	PENDING	PENDING	0
4	DMSO	-10	1.0	PENDING	PENDING	PENDING	0


## 3. Adding training data in EDBO+.

##### Note: We will use Python and Pandas to add new training data in this example. But you can always edit and add new data into the '.csv' file using any spreedsheet editor (such as Excel, Libreoffice Calc, ...) if that's more convinient for you.

##### Let's open again the $\bf{my\_optimization.csv}$ file we generated before:

df_edbo = pd.read_csv('my_optimization.csv')
df_edbo.head(5)

	solvent	T	concentration	yield	ee	side_product	priority
0	DMSO	10	1.0	PENDING	PENDING	PENDING	1
1	THF	10	1.0	PENDING	PENDING	PENDING	1
2	Toluene	10	0.2	PENDING	PENDING	PENDING	1
3	THF	-10	0.1	PENDING	PENDING	PENDING	0
4	DMSO	-10	1.0	PENDING	PENDING	PENDING	0


##### We can fill the first out entry in the previous dataframe with the "observed" values using Pandas:


df_edbo.loc[0, 'yield'] = 20.5
df_edbo.loc[0, 'ee'] = 40
df_edbo.loc[0, 'side_product'] = 0.1

##### We can check that we have filled out the first entry with our "observed data":

df_edbo.head(5)

	solvent	T	concentration	yield	ee	side_product	priority
0	DMSO	10	1.0	20.5	40	0.1	1
1	THF	10	1.0	PENDING	PENDING	PENDING	1
2	Toluene	10	0.2	PENDING	PENDING	PENDING	1
3	THF	-10	0.1	PENDING	PENDING	PENDING	0
4	DMSO	-10	1.0	PENDING	PENDING	PENDING	0

##### We can also fill out the second entry with their corresponding "observations":

df_edbo.loc[1, 'yield'] = 50.3
df_edbo.loc[1, 'ee'] = 10
df_edbo.loc[1, 'side_product'] = 0.2


df_edbo.head(5)

solvent	T	concentration	yield	ee	side_product	priority
0	DMSO	10	1.0	20.5	40	0.1	1
1	THF	10	1.0	50.3	10	0.2	1
2	Toluene	10	0.2	PENDING	PENDING	PENDING	1
3	THF	-10	0.1	PENDING	PENDING	PENDING	0
4	DMSO	-10	1.0	PENDING	PENDING	PENDING	0

##### Now we can save our dataset as $\bf{my\_optimization\_round0.csv}$:

df_edbo.to_csv('my_optimization_round0.csv', index=False)


## 4. Running EDBO+ with training data.

##### First let's check our previous data (which include some $\bf{yield}$, $\bf{ee}$ and $\bf{side\_product}$ observations, which will be used to train the model):

df_edbo_round0 = pd.read_csv('my_optimization_round0.csv')
df_edbo_round0.head(5)

	solvent	T	concentration	yield	ee	side_product	priority
0	DMSO	10	1.0	20.5	40	0.1	1
1	THF	10	1.0	50.3	10	0.2	1
2	Toluene	10	0.2	PENDING	PENDING	PENDING	1
3	THF	-10	0.1	PENDING	PENDING	PENDING	0
4	DMSO	-10	1.0	PENDING	PENDING	PENDING	0


##### Now that we have introduced some "observations" in our $\bf{my\_optimization\_round0.csv}$ file, we can execute EDBO+ to suggest samples using these "observations" as training data.

EDBOplus().run(
    filename='my_optimization_round0.csv',  # Previous scope (including observations).
    objectives=['yield', 'ee', 'side_product'],  # Objectives to be optimized.
    objective_mode=['max', 'max', 'min'],  # Maximize yield and ee but minimize side_product.
    batch=3,  # Number of experiments in parallel that we want to perform in this round.
    columns_features='all', # features to be included in the model.
    init_sampling_method='cvtsampling'  # initialization method.
)





The following columns are categorical and will be encoded using One-Hot-Encoding: ['solvent']
Using EHVI acquisition function.
Using hyperparameters optimized for continuous variables.
Using hyperparameters optimized for continuous variables.
Using hyperparameters optimized for continuous variables.
Number of QMC samples using SobolQMCNormalSampler sampler: 512
Acquisition function optimized.
Predictions obtained and expected improvement obtained.
solvent	T	concentration	yield	ee	side_product	priority
19	THF	-10	0.2	PENDING	PENDING	PENDING	1.0
35	DMSO	25	1.0	PENDING	PENDING	PENDING	1.0
10	DMSO	0	0.2	PENDING	PENDING	PENDING	1.0
7	Toluene	25	1.0	PENDING	PENDING	PENDING	0.0
6	Toluene	25	0.2	PENDING	PENDING	PENDING	0.0
5	Toluene	25	0.1	PENDING	PENDING	PENDING	0.0
17	Toluene	10	1.0	PENDING	PENDING	PENDING	0.0
2	Toluene	10	0.2	PENDING	PENDING	PENDING	0.0
18	Toluene	10	0.1	PENDING	PENDING	PENDING	0.0


##### Again the samples suggested by EDBO+ have $\bf{priority = +1}$. In addition, we assign $\bf{priority = -1}$ to the experiments that we have already run (these are at the bottom of the dataset).

```


please read the above example, and help me write code for optimization of my reaction yield. Below are the possible choices of synthesis parameters: 

```
Substrate Concentration 0.025 M, 0.05 M, 0.075 M, 0.1M, 0.125M

Mediator eq. 0, 0.25, 0.5, 0.75, 1

Mediator type NHPI, TCNHPI, QD, DABCO, TEMPO

Electrolyte type LiClO4, LiOTf , Bu4NClO4, Et4NBF4, Bu4NPF6

Co solvent HFIP 0, 1

``` 
First, the code should suggest the 5 initial experiments I should start with and I only want to maximize the reaction yield. If possible, make sure you write all your code in one code block so that I can copy and paste it to Jupyter Notebook to execute directly. You do not need to have further explanation on the codes or other instructions.




"""
,
    
 """


Now I have completed the first 5 reactions in the lab, the observed yields are "6, 5, 5, 8, 0" for suggested reactions 1 to 5, their conditions are "
0.1	0.5	TCNHPI	Bu4NPF6	0	
0.075	0.5	QD	Et4NBF4	1	
0.075	0.5	QD	LiOTf	1	
0.075	0.5	NHPI	LiClO4	1	
0.05	0.5	TEMPO	Et4NBF4	0	
"
please write the code to help me suggest the next 5 experiments for me. Make sure you import related modules and please write code in one code block.

 """

]

    
    # Run the code generation and execution process
    generate_and_execute_code(user_prompts, model=model, num_calls=100, max_reflection=2)
    
    #Process and summarize the generated results
    process_and_summarize_results()