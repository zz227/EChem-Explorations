"""
This script interacts with various LLMs (e.g., GPT-3.5, GPT-4, Claude, and LLaMA) to generate and execute Python code for machine learning tasks. It uses the user's selected LLM model and prompts the user for necessary API keys based on the model. The results of generated code execution are stored in an Excel file.

Requirements:
- openai for GPT models
- anthropic for Claude models
- replicate for LLaMA models
- pandas, subprocess, tempfile, re for general file handling and code execution
- rdkit for chemical descriptor generation (used in the ML task)

The main function `generate_and_execute_code` interacts with the LLM and stores results in a pandas DataFrame.
"""

import re
import tempfile
import os
import glob
import shutil
import pandas as pd
import time
import traceback
import glob
import numpy as np
from IPython.utils.io import capture_output
import subprocess
import pandas as pd
import replicate
import anthropic
import openai
import json



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
            prompt_conversation_history.append({"role": "user", "content": user_prompts[i]})
            prompt_conversation_history.append({"role": "assistant", "content": response_content})

            code_block = re.search(r'```python\n(.*?)\n```', response_content, re.DOTALL)
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
                    reflection_prompt = f"Please reflect on the code you previously wrote. There is an error and I cannot run it on my Jupyter Notebook. The error message is:\n{error_message}\nPlease try to catch any bugs or failures to follow the user instruction. In your answer, give me the full revised code."
                    prompt_conversation_history.append({"role": "user", "content": reflection_prompt})
                    revised_response_content, revised_response_time = chat(model, reflection_prompt, prompt_conversation_history)
                    prompt_conversation_history.append({"role": "assistant", "content": revised_response_content})

                    revised_code_block = re.search(r'```python\n(.*?)\n```', revised_response_content, re.DOTALL)

                    if revised_code_block:
                        code = revised_code_block.group(1)

                    response_time += revised_response_time
                    num_reflections += 1

            row_data[f'Generated Code {i+1}'] = code
            row_data[f'Execution Result {i+1}'] = execution_result
            row_data[f'Response Time {i+1} (s)'] = response_time
            row_data[f'Number of Reflections {i+1}'] = num_reflections
            conversation_history = prompt_conversation_history.copy()

        row_data['Conversation History'] = str(conversation_history)
        results.append(row_data)

    df = pd.DataFrame(results)
    df.to_excel(f'Results_{model}.xlsx', index=False)

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
        
        result = subprocess.run(['python', temp_file_path], capture_output=True, text=True)
        os.unlink(temp_file_path)

        if result.returncode == 0:
            return 1, None
        else:
            return 0, result.stderr
    except Exception as e:
        error_message = traceback.format_exc()
        os.unlink(temp_file_path)
        return 0, error_message


def chat(model, user_prompt, conversation_history, max_retries=3):
    """
    Helper function to chat with the specified model and handle retries.

    Args:
        model (str): The model to use for code generation.
        user_prompt (str): The prompt for generating the Python code.
        conversation_history (list): The history of the conversation.
        max_retries (int): The maximum number of retries if an error occurs.

    Returns:
        tuple: A tuple containing the response content and response time.
    """
    retry_count = 0
    pre_prompt = "You are a helpful coding assistant who always writes detailed and executable code without human implementation."

    while retry_count < max_retries:
        try:
            start_time = time.time()
            if model.startswith('claude'):
                client = anthropic.Anthropic(api_key=anthropic_key)
                response = client.completions.create(
                    model=model,
                    prompt=conversation_history + [{"role": "user", "content": user_prompt}],
                    max_tokens=4096
                )
                response_content = response["completion"]
            elif model.startswith('gpt'):
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "system", "content": pre_prompt}] + conversation_history + [{"role": "user", "content": user_prompt}]
                )
                response_content = response.choices[0].message["content"]
            elif model.startswith('llama'):
                formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
                model_name = "meta/llama-2-70b-chat" if model == "llama-2" else "meta/codellama-34b-instruct:eeb928567781f4e90d2aba57a51baef235de53f907c214a4ab42adabf5bb9736"
                response = replicate.run(model_name, input={"prompt": f"{formatted_history}User: {user_prompt}\nAssistant:"})
                response_content = ''.join(response)

            response_time = time.time() - start_time
            return response_content, response_time
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                return f"Error: {str(e)}", 0

def process_excel_files(file_names):
    summary_data = []
    
    for file_name in file_names:
        # Check if the "acc_" file already exists
        output_file_name = "acc_" + file_name
        if os.path.isfile(output_file_name):
            # If the file exists, read it instead of the original file
            df = pd.read_excel(output_file_name)
        else:
            # If the file doesn't exist, read the original file
            df = pd.read_excel(file_name)
            
            # Create a new column "Performance" and initialize it with an empty string
            df["Performance"] = ""
            
            # Filter rows where "Execution Result 1" is 1
            filtered_df = df[df["Execution Result 1"] == 1]
            
            total_rows = len(df)
            print(f"Processing {file_name} with {total_rows} rows")
            
            # Iterate over the filtered rows
            for index, row in filtered_df.iterrows():
                print(f"Processing row {index + 1}/{total_rows}")
                code = row["Generated Code 1"]
                
                # Create a temporary file to write the code
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
                    temp_file.write(code)
                    temp_file_path = temp_file.name
                
                try:
                    # Execute the code using subprocess
                    result = subprocess.run(['python', temp_file_path], capture_output=True, text=True, timeout=100)
                    os.unlink(temp_file_path)
                    
                    if result.returncode == 0:
                        # Code executed successfully
                        output = result.stdout.split('\n')
                        
                        # Find the accuracy value from the captured output
                        accuracy_values = []
                        for line in output:
                            match = re.search(r"accuracy:\s*(\d+(\.\d+)?)", line, re.IGNORECASE)
                            if match:
                                accuracy_values.append(float(match.group(1)))
                        
                        # Update the "Performance" column with the highest accuracy value
                        if accuracy_values:
                            df.at[index, "Performance"] = max(accuracy_values)
                    else:
                        # Code execution encountered an error
                        print(f"Error executing code at index {index} in file {file_name}: {result.stderr}")
                except subprocess.TimeoutExpired:
                    print(f"Error: Code execution at index {index} in file {file_name} exceeded time limit and was terminated.")
                    os.unlink(temp_file_path)

    
                except Exception as e:
                    # Error occurred while executing the code
                    error_message = traceback.format_exc()
                    os.unlink(temp_file_path)
                    print(f"Error executing code at index {index} in file {file_name}: {error_message}")
                    
            
            # Save the updated DataFrame back to the Excel file
            df.to_excel(output_file_name, index=False)
        
        # Generate summary data for the current file
        model_name = file_name.split("_")[1]
        total_rows = len(df)
        executed_rows = len(df[df["Execution Result 1"] == 1])
        
        # Count rows where "Number of Reflections 1" is 0 within the filtered DataFrame
        one_conversation_rows = len(df[(df["Execution Result 1"] == 1) & (df["Number of Reflections 1"] == 0)])
        
        # Handle empty or non-numeric values in the "Performance" column
        avg_accuracy = pd.to_numeric(df["Performance"], errors='coerce').mean()
        
        # Calculate the average time considering the "Number of Reflections 1"
        df["Adjusted Time"] = df["Response Time 1 (s)"] / (df["Number of Reflections 1"] + 1)
        avg_time = df["Adjusted Time"].mean()
        
        correctness = len(df[pd.to_numeric(df["Performance"], errors='coerce') > 0.85]) /  total_rows #executed_rows
        
        # Calculate the average code length
        code_length = df["Generated Code 1"].apply(lambda x: len(str(x).split())).mean()
        
        summary_row = {
            "Model Name": model_name,
            "Avg Time": avg_time,
            "Code Length": code_length,
            "Code Executability (one conversation)": one_conversation_rows / total_rows,
            "Code Executability (with reflection)": executed_rows / total_rows,
            "Correctness": correctness,
            "Avg Accuracy of ML Models": avg_accuracy
        }
        
        summary_data.append(summary_row)
    
    # Create a summary DataFrame from the summary data
    summary_df = pd.DataFrame(summary_data)
    
    # Reorder the columns
    column_order = ["Model Name", "Avg Time", "Code Length", "Code Executability (one conversation)",
                    "Code Executability (with reflection)", "Correctness", "Avg Accuracy of ML Models"]
    summary_df = summary_df[column_order]
    
    # Save the summary DataFrame to an Excel file
    summary_file_name = "summary.xlsx"
    summary_df.to_excel(summary_file_name, index=False)
    
    return summary_file_name

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
    # Load the prompt from combined JSON file
    with open("prompts.json", "r") as json_file:
        prompt_data = json.load(json_file)

    # Extract user prompts for file1
    user_prompts = prompt_data["ml_prompts"]

    
    # Run the code generation and execution process
    generate_and_execute_code(user_prompts, model=model, num_calls=100, max_reflection=2)
    
    #Process and summarize the generated results
    process_and_summarize_results()