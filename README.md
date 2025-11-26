# Resume Parser + Categorizer

**Authors:** Trinity Thiele, Alison Tintera, Mustapha Ceesay, Ryan Fantigrossi  

## Overview

Large companies and recruiters rarely have the time to manually review every resume they receive. Instead, they rely on automated systems and machine learning models to screen candidates. Unfortunately, many qualified applicants are filtered out—not because they lack skills, but because their resumes aren’t formatted or phrased in ways these algorithms can easily understand.

This project aims to build an intelligent **Resume Parser + Categorizer** that:
- Classifies resumes into appropriate job categories
- Provides constructive, model-backed feedback to help applicants improve their resumes for specific roles.

## Motivation

Many resume-screening algorithms struggle with inconsistent formats, noisy text, and varied phrasing of skills and experience. Our tool bridges that gap by:

- Parsing resumes in multiple formats  
- Extracting meaningful features (skills, experience, etc.)  
- Categorizing resumes into job fields  
- Using an LLM to explain the classification and provide tailored feedback

This helps job seekers better understand how automated hiring systems “see” their resumes and how to refine them before applying.

## Methodology / Pipeline

1. **Text Extraction**  
   - Extract raw text from uploaded documents (PDF, DOCX, TXT).

2. **Preprocessing**  
   - Clean and normalize the text (remove noise, standardize formatting, etc.).

3. **Feature Extraction (spaCy)**  
   - Use spaCy to extract structured features (e.g., skills, experience, entities).  
   - Export features in JSON format for downstream processing.

4. **Text Vectorization (Bag of Words)**  
   - Convert resume text into numerical vectors.  
   - Identify words and phrases most relevant to each job category.

5. **Feature Fusion**  
   - Combine text vectors with extracted structured features (skills, experience, etc.).

6. **Model Training**  
   - Train a classification model (e.g., Logistic Regression) on the combined features to predict resume job categories.

7. **Evaluation & Optimization**  
   - Evaluate accuracy, precision/recall, and other metrics.  
   - Tune hyperparameters and improve performance.

8. **LLM-Based Feedback**  
   - Use a Large Language Model (LLM) to explain why a resume was assigned to a given category.  
   - Provide actionable feedback on strengths, gaps, and potential improvements.

## Goals

We aim to build a system that:

-  Supports multiple resume formats: **PDF, DOCX, TXT**  
- Achieves **classification accuracy above 90%**  
- Parses and categorizes each resume in **under 3 seconds**

## Project Structure

.  
└── resume-parser/  
    ├── README.md  
    ├── .gitignore  
    ├── data/  
    │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── raw/ # Raw data  
    │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── processed/ # Cleaned data  
    │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── training/ # Processed data for training  
    │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── testing/ # Processed data for testing  
    ├── models/  
    │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── spacy_data # Where all the spaCy stuff is stored  
    │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── spacy_layout_data # Where all the spaCy Layout stuff is stored  
    │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── bag_of_words # Where all the Bag of Words stuff is stored  
    ├── nlp/  
    │&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── spacy_loader # Where spaCy is being used  
    └── script/  
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_model # Where the model is being trained  
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── test_model # Where the model is being tested  
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── parse_resume # Parsing the resumes  
