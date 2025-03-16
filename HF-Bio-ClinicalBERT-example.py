# -*- coding: utf-8 -*-
"""
This is an example of integrating the Hugging Face Transformers library with ipywidgets to create an interactive
interface for analyzing health data using a pre-trained clinical BERT model. This is to
give a demonstration of how to use the Hugging Face Transformers library in a Jupyter Notebook environment.

This template could be used to integrate with other publicly available models or custom models for various NLP
tasks using the BERT transformers in the Clinical or HealthCare domain.

"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import ipywidgets as widgets
from IPython.display import display, clear_output

# Using a publicly available clinical model
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
health_monitor = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Broad Disease Mapping
broad_disease_mapping = {
    "LABEL_0": "No significant condition",
    "LABEL_1": "Cardiovascular Diseases (e.g., hypertension, heart disease)",
    "LABEL_2": "Metabolic and Endocrine Disorders (e.g., diabetes, thyroid issues)",
    "LABEL_3": "Respiratory Diseases (e.g., asthma, COPD)",
    "LABEL_4": "Neurological Conditions (e.g., stroke, epilepsy)",
    "LABEL_5": "Infectious Diseases (e.g., influenza, COVID-19)",
    "LABEL_6": "Oncological Conditions (e.g., cancers)",
    "LABEL_7": "Gastrointestinal Disorders (e.g., IBS, Crohn’s disease)",
    "LABEL_8": "Musculoskeletal Disorders (e.g., arthritis, osteoporosis)",
    "LABEL_9": "Immunological/Autoimmune Disorders (e.g., lupus, rheumatoid arthritis)"
}

# Function to Analyze Health Data
def analyze_health_data(input_text):
    prediction = health_monitor(input_text)[0]
    disease_prediction = broad_disease_mapping.get(prediction["label"], "Unknown Condition")
    output_str = (
        f"Raw Model Output: {prediction}\n"
        f"Interpreted Prediction: {disease_prediction}\n"
        f"Confidence Score: {prediction['score']*100:.2f}%"
    )
    return output_str

# Interactive Interface Using ipywidgets
input_text = widgets.Textarea(
    value='Enter patient health data here...',
    placeholder='Type the clinical notes or patient report',
    description='Health Data:',
    disabled=False,
    layout=widgets.Layout(width='100%', height='100px')
)

# Button widget to trigger the analysis
analyze_button = widgets.Button(
    description='Analyze',
    disabled=False,
    button_style='',  # Options: 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to analyze the health data',
    icon='check'
)

# Output widget to display the results
output_area = widgets.Output()

def on_analyze_button_clicked(b):
    with output_area:
        clear_output()
        input_data = input_text.value
        result = analyze_health_data(input_data)
        print(result)

analyze_button.on_click(on_analyze_button_clicked)

display(input_text, analyze_button, output_area)