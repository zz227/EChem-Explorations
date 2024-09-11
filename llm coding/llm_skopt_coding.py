import re
import tempfile
import os
import shutil
import anthropic
import openai
import pandas as pd
import time
import traceback
import glob
import numpy as np
from IPython.utils.io import capture_output
import subprocess
import pandas as pd
import openpyxl
import csv
import random
import itertools
import numpy as np
import time
from itertools import product
import replicate



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
        model (str): The model to use for code generation. Examples:
            - 'llama-3.1'
            - 'llma-3'
            - 'claude-3-opus-20240229'
            - 'claude-3-sonnet-20240229'
            - 'claude-3-5-sonnet-20240620'
            - 'gpt-4-turbo-2024-04-09'
            - 'gpt-4-0613'
            - 'gpt-4o'
            - 'gpt-4o-mini'
            - 'gpt-3.5-turbo' (default)
        num_calls (int): The number of times to generate and execute code.
        max_reflection (int): The maximum number of reflection attempts when code execution fails.

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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if os.path.exists('Results_'+model+'.xlsx'):
        df.to_excel('Results_'+model+timestamp+'.xlsx', index=False)
    else:
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
    print("run code")

    

    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, timeout=100)  # Timeout set to 100 seconds 
        os.unlink(temp_file_path)

        if result.returncode == 0:
            print("complete code running")
            return 1, None
        else:
            print("not complete code running")
            return 0, result.stderr
    except subprocess.TimeoutExpired:
        print("Error: Code execution exceeded time limit and was terminated.")
        os.unlink(temp_file_path)
        return 0, "Code execution exceeded time limit and was terminated."
    except Exception as e:
        error_message = traceback.format_exc()
        os.unlink(temp_file_path)
        print(error_message)
        print("error message, return 0")
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
                print("start")
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "system", "content": pre_prompt}] + conversation_history + [{"role": "user", "content": user_prompt}]
                )
                
                response_content = response.choices[0].message.content
                print("response collect")
                
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
                        "max_tokens": 4096,
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
                print(error_message)
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
                print(f"{file_name} does not pass: Incorrect 'yield' values for priority -1")
                continue  # Skip to the next file

            # Check the first five rows for columns where priority is 1
            priority_one_rows = df[df['priority'] == 1].head(5)

            # Define acceptable values for each column as floats
            substrate_concentrations = [0.025, 0.05, 0.075, 0.1, 0.125]
            mediator_eqs = [0, 0.25, 0.5, 0.75, 1]
            mediator_types = ["NHPI", "TCNHPI", "QD", "DABCO", "TEMPO"]
            electrolyte_types = ["LiClO4", "LiOTf", "Bu4NClO4", "Et4NBF4", "Bu4NPF6"]
            co_solvents = [0, 1]

            # Prepare to check 'yield' column for all rows where priority is 1
            yield_values = df[df['priority'] == 1]['yield'].astype(str).tolist()
            if not all(yield_value == 'PENDING' for yield_value in yield_values):
                print(f"{file_name} does not pass: 'yield' values for priority 1 are not 'PENDING'")
                continue  # Skip to the next file

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
            return True  # Return True immediately upon successful validation of a file

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

    # Load the prompts from combined JSON file (prompts.json)
    with open("prompts.json", "r") as json_file:
        prompt_data = json.load(json_file)

    # Extract user prompts for file2
    user_prompts = prompt_data["skopt_prompts"]

    # Run the code generation and execution process
    generate_and_execute_code(user_prompts, model=model, num_calls=100, max_reflection=2)
    
    #Process and summarize the generated results
    process_and_summarize_results()