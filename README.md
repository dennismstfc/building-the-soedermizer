# Södermizer: Automatic Gender-Neutral to Gendered Text Translation

## Project Overview
The Södermizer project aims to develop a pipeline for building a comprehensive text corpus consisting of non-gendered and gendered versions of German sentences. Using this corpus, we train a language model to automatically translate gendered text into non-gendered text, addressing the nuances of gender in the German language.

## What is Gendering?
In the German language, nouns and pronouns are typically gendered, meaning they are categorized as masculine, feminine, or neuter. For instance, the word "teacher" can be "Lehrer" (masculine) or "Lehrerin" (feminine). In recent years, there has been a movement to use more inclusive language that avoids gender-specific terms, often by using gender-neutral forms or combining both masculine and feminine forms (e.g., "Lehrer*innen" or "Lehrer/-innen"). This practice, known as "gendering," aims to make language more inclusive of all genders, but it also introduces complexities in text processing and translation.

## Goals
1. **Training Language Models**: Our ultimate goal is to train a language model, dubbed "Södermizer," to perform automatic translation of non-gendered text into gendered text accurately. This involves fine-tuning pre-existing models to specialize in this specific translation task.

## Pipeline Description
TOODO

# Usage
1. **Create a virtual environment**:
    ```bash
    python -m venv .env
    ```

2. **Activate the virtual environment**:
    - On **Windows**:
        ```bash
        .\.env\Scripts\activate
        ```
    - On **Linux/MacOS**:
        ```bash
        source .env/bin/activate
        ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Create Dataset
1. **Run the dataset script**:
    ```bash
    python dataset.py
    ```

**NOTE:** It is important to have an existing Java 8 or higher installation in order to perform grammar correction.

## Train model
1. **Install PyTorch**: 
```bash 
pip3 install torch --index-url https://download.pytorch.org/whl/cu124
```
**NOTE:** Make sure to select the correct CUDA version. For further information look into the [documentation](https://pytorch.org/get-started/locally/).


2. 

#### Why Södermizer?
The name "Södermizer" is inspired by Markus Söder, a Bavarian politician known for his strong opposition to gender-neutral language reforms. By naming this model Södermizer, the societal and political discourse surrounding gendered language is playfully acknowledged, while also addressing the technical challenges in automating these translations.