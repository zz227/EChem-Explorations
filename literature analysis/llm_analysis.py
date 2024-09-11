# process_pdfs.py
"""
This script processes PDF files using GPT-4 to analyze their content. 
It requires the user to input their OpenAI API key. The PDFs themselves 
are not included due to copyright restrictions, but you can download them 
via DOI links provided in an Excel spreadsheet.

Dependencies:
- os
- re
- pandas
- PyPDF2
- tiktoken
- openai

Make sure to install the necessary libraries with:
pip install pandas PyPDF2 tiktoken openai

Usage:
1. Place the PDF files in a folder.
2. Ensure the analysis_prompt.json file is available for the long prompt.
3. Run the script and provide the necessary OpenAI API key when prompted.

"""

# Importing the required libraries
import os
import re
import pandas as pd
import PyPDF2
import tiktoken  # Import tiktoken for token counting
import openai
import json

# Ask the user to input their OpenAI API key
my_api_key = input("Please enter your OpenAI API key: ")

# Initialize OpenAI client
openai.api_key = my_api_key

# Load the tokenizer for GPT-4 model
encoding = tiktoken.encoding_for_model("gpt-4")

# Function to count tokens using tiktoken
def count_tokens(text, encoding_name="cl100k_base"):
    """Returns the number of tokens in a text string using the specified encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

# Define a function to remove content after "References" or "Acknowledgement" if found in the last 30% of the text
def clean_text(text):
    """Removes content after 'References' or 'Acknowledgement' if found in the last 30% of the document."""
    # Calculate the index that represents 70% of the text
    cutoff_index = int(len(text) * 0.7)
    
    # Find the index of "References" or "Acknowledgement" regardless of case
    match_references = re.search(r'\b(references|acknowledgements?)\b', text[cutoff_index:], re.IGNORECASE)
    
    if match_references:
        # If found in the last 30%, remove everything after the match
        ref_index = cutoff_index + match_references.start()
        text = text[:ref_index]
    
    return text

# Define a function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Define a function to reduce content if it exceeds the token limit
def reduce_content(text, target_token_count=6000):
    """
    Reduces the length of the content by removing 5% of the initial length from the back
    at a time, up to 20 iterations, until the token count is below the predefined limit.
    """
    encoding_name = "cl100k_base"  # Encoding used for token counting
    max_iterations = 20  # Maximum iterations to remove 5% of the text
    reduction_percent = 0.05  # 5% reduction of the initial length each time

    # Step 1: Calculate the current number of tokens
    current_token_count = count_tokens(text, encoding_name)
    initial_length = len(text)
    
    # Step 2: Remove 5% of the initial length up to 20 times, or until below the token limit
    for _ in range(max_iterations):
        if current_token_count <= target_token_count:
            break  # Stop if the token count is below the limit
        
        # Remove the last 5% of the initial length of the text
        reduce_amount = int(initial_length * reduction_percent)
        text = text[:-reduce_amount]
        
        # Recalculate the token count
        current_token_count = count_tokens(text, encoding_name)
    
    return text

# Define a function to process each PDF and send it to the GPT-4 model
def process_pdf(pdf_path, prompt_long):
    """Processes a single PDF file, cleans the text, checks token limits, and sends it to GPT-4."""
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Step 1: Clean the text by removing "References" or "Acknowledgements" if present in the last 30%
    cleaned_text = clean_text(pdf_text)
    
    # Step 2: Create the system and user messages for the GPT model
    system_prompt = "You are a helpful assistant."
    complete_prompt = f"{prompt_long}\n\nManuscript:\n{cleaned_text}"
    
    # Step 3: Count tokens for system and user prompts
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": complete_prompt}
    ]
    prompt_token_count = count_tokens(complete_prompt, encoding_name="cl100k_base")
    
    # Step 4: Check if the token count exceeds the limit (6000) and reduce if necessary
    target_token_count = 6000
    if prompt_token_count > target_token_count:
        # Reduce the manuscript content if token limit exceeded
        reduced_cleaned_text = reduce_content(cleaned_text, target_token_count)
        complete_prompt = f"{prompt_long}\n\nManuscript:\n{reduced_cleaned_text}"
        # Update the message and recount tokens
        messages[1]['content'] = complete_prompt
        prompt_token_count = count_tokens(complete_prompt, encoding_name="cl100k_base")
    
    # Send the prompt to GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    
    # Extract the response
    gpt_output = response['choices'][0]['message']['content']
    return gpt_output

# Define a function to parse the GPT-4 response into a structured format (csv-ready)
def parse_gpt_output(gpt_output, pdf_file_name):
    """Parses the GPT-4 response and returns a structured dictionary."""
    # Use regular expressions to extract specific parts of the output
    overall_answer = re.search(r'Overall Answer:\s*(Yes|No)', gpt_output)
    part_a_answer = re.search(r'Part A:\s*(Yes|No)', gpt_output)
    reasoning_a = re.search(r'Reasoning A:\s*(.*)', gpt_output)
    part_b_answer = re.search(r'Part B:\s*(Yes|No)', gpt_output)
    reasoning_b = re.search(r'Reasoning B:\s*(.*)', gpt_output)
    part_c_answer = re.search(r'Part C:\s*(Yes|No)', gpt_output)
    reasoning_c = re.search(r'Reasoning C:\s*(.*)', gpt_output)
    
    # Extract the matched text or set as empty string if not found
    overall_answer = overall_answer.group(1) if overall_answer else ''
    part_a_answer = part_a_answer.group(1) if part_a_answer else ''
    reasoning_a = reasoning_a.group(1) if reasoning_a else ''
    part_b_answer = part_b_answer.group(1) if part_b_answer else ''
    reasoning_b = reasoning_b.group(1) if reasoning_b else ''
    part_c_answer = part_c_answer.group(1) if part_c_answer else ''
    reasoning_c = reasoning_c.group(1) if reasoning_c else ''
    
    # Return as a dictionary (to be used as a row in a DataFrame)
    return {
        'PDF File Name': pdf_file_name,
        'Overall Answer': overall_answer,
        'Part A Answer': part_a_answer,
        'Reasoning A': reasoning_a,
        'Part B Answer': part_b_answer,
        'Reasoning B': reasoning_b,
        'Part C Answer': part_c_answer,
        'Reasoning C': reasoning_c
    }

# Main processing function to handle all PDFs in a folder
def process_pdfs_in_folder(folder_path, prompt_long):
    """Processes all PDFs in the specified folder and saves the results in a DataFrame."""
    # List to store the parsed outputs
    results = []
    
    # Loop over all PDF files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            
            # Process each PDF and get the GPT output
            gpt_output = process_pdf(pdf_path, prompt_long)
            
            # Parse the GPT output
            parsed_output = parse_gpt_output(gpt_output, file_name)
            
            # Append the parsed output to the results list
            results.append(parsed_output)
    
    # Convert the results into a DataFrame and return
    df = pd.DataFrame(results)
    return df

# Load the long prompt from the analysis_prompt.json file
def load_prompt_from_json(json_file_path):
    """Loads the long prompt from the analysis_prompt.json file."""
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        prompt_long = data['prompt_long']
    return prompt_long

if __name__ == "__main__":
    # Set the folder path where PDFs are stored
    folder_path = input("Please enter the folder path where the PDFs are stored: ")
    json_file_path = 'analysis_prompt.json'  # Replace with the path to the analysis_prompt.json

    # Load the long prompt from JSON
    prompt_long = load_prompt_from_json(json_file_path)

    # Process all PDFs in the folder and get the results in a DataFrame
    df_results = process_pdfs_in_folder(folder_path, prompt_long)

    # Save the DataFrame to a CSV file
    df_results.to_csv('results.csv', index=False)

    # Display the results
    print(df_results.head())
