# ML + LLM for Electrochemical Reactions 

<img src="image/project.png" width="40%" height="40%">


This repository contains code and datasets associated with the research preprint **[Integrating Machine Learning and Large Language Models to Advance Exploration of Electrochemical Reactions](https://chemrxiv.org/engage/chemrxiv/article-details/66cf7e1120ac769e5f241caf)**. This project demonstrates the synergistic potential of machine learning (ML) and large language models (LLMs) for the exploration, prediction, and optimization of electrochemical C-H oxidation reactions.

## Overview

In this study, we developed a framework combining ML and LLMs to facilitate rapid screening of electrochemical reactions, predict substrate reactivity, and optimize reaction conditions. The repository provides:
- Code for literature analysis using LLMs to assign Yes/No label based on prompt and give justifications.
- ML models and code to predict reactivity and site selectivity of C-H oxidation reactions.
- Datasets from both experimental results and literature mining.

## Repository Structure

### 1. `literature_analysis`
This folder contains code and data related to the semantic analysis of scientific literature, utilizing LLMs to extract relevant information for electrochemical C-H oxidation reactions.
- **SF1. Literature Screening Dataset.xlsx**: Contains summarized results and DOI links to the PDFs associated with this study.
- **analysis_prompt.json**: A JSON file containing the prompts used for LLM analysis.
- **llm_analysis.py**: Python script to process PDFs with OpenAI GPT-4. It extracts and cleans text from PDF files, limits token length, and sends it to the LLM for analysis.

### 2. `llm_coding`
This folder contains the core ML and LLM interaction code for predicting reaction outcomes and optimizing synthesis conditions.
- **SF3. Auto Coding Dataset.xlsx**: Dataset generated through LLMs for auto-coding ML tasks.
- **echem_train.xlsx** & **echem_test.xlsx**: Training and test datasets used for ML models related to electrochemical C-H oxidation reactions.
- **llm_ml_coding.py**: A script for interacting with various LLMs (GPT-4, Claude, LLaMA) to generate and execute Python code for ML tasks.
- **llm_edbo_coding.py** & **llm_skopt_coding.py**: Similar to `llm_ml_coding.py`, these scripts are for generating code for synthesis optimization tasks.
- **prompts.json**: Contains specific prompts used by the LLMs to generate machine learning and optimization code.

### 3. `screening`
This folder contains trained machine learning models used for electrochemical reaction screening.
- **trained chemprop models**: Includes models for reactivity and selectivity prediction.
- **classical ML models**:
- **SF2. EChem Reaction Screening Dataset.xlsx**: Dataset containing experimental screening results.
- **classical ML.py**: Python script to interact with classical ML models for predicting reaction outcomes.

### 4. `results`
This folder contains the final datasets and results generated during this study.
- **SF1. Literature Screening Dataset.xlsx**: Dataset from the literature analysis phase.
- **SF2. EChem Reaction Screening Dataset.xlsx**: Results from reaction screening.
- **SF3. Auto Coding Dataset.xlsx**: Dataset generated from LLM-guided code generation.
- **SF4. EChem Reaction Optimization Dataset.xlsx**: Results from the reaction optimization process.

## Requirements

### Python Libraries
Install the following dependencies before running the scripts:
```bash
pip install pandas PyPDF2 tiktoken openai anthropic replicate rdkit numpy
```

### API Keys
The code interacts with several LLMs (GPT, Claude, LLaMA). You will need API keys from OpenAI, Anthropic, and Replicate to execute these scripts. The scripts will prompt for these keys when necessary.

## Usage

### Literature Analysis
To process and analyze literature data:
1. Place the relevant PDFs in a folder.
2. Ensure the `analysis_prompt.json` file is available in the working directory.
3. Run the `llm_analysis.py` script to analyze the PDFs. The results will be saved in CSV format.

```bash
python literature_analysis/llm_analysis.py
```

### Machine Learning Model Training
For training and testing ML models on the provided datasets:
1. Review the datasets in `echem_train.xlsx` and `echem_test.xlsx`.
2. Use the `llm_ml_coding.py` script to generate code for ML tasks via LLM interactions.

```bash
python llm_coding/llm_ml_coding.py
```

### Reaction Screening and Optimization
To perform screening and optimization of reaction conditions:
1. Use the `screening_analysis.py` script to analyze screening data.
2. The `llm_edbo_coding.py` and `llm_skopt_coding.py` scripts can be used for generating and evalauting synthesis optimization tasks similar to `llm_ml_coding.py`.

## Results

The results from running the LLM-guided code generation and the subsequent execution of ML tasks are stored in the `results` folder. Summary files include:
- **results.csv**: Output of the screening and optimization processes.
- **summary.xlsx**: Summary of the performance of different LLMs in code generation and ML task execution.



## License 

The input prompt generation script is distributed under the MIT open source license (see [`LICENSE.txt`](LICENSE.txt))


## Contributing

If you have any questions/comments/feedback, please feel free to reach out to any of the authors.


## Acknowledgements

This material is based upon work supported by Pfizer.
We extend our gratitude to the Machine Learning for Pharmaceutical Discovery and Synthesis Consortium for their support.

## References
For this work:

> ChemRxiv <br/>
> Integrating Machine Learning and Large Language Models to Advance Exploration of Electrochemical Reactions <br/>
> [https://arxiv.org/abs/2303.08774](https://doi.org/10.26434/chemrxiv-2024-pk105-v2) <br/>




For GPT-4: 

> GPT-4 Technical Report <br/>
> OpenAI <br/>
> https://arxiv.org/abs/2303.08774 <br/>

# echem
