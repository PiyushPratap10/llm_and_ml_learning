import langextract as lx
import os

KEY = os.getenv('GEMINI_API_KEY')

input_text = """
The patient was prescribed Lisinopril and Metformin last month.
He takes the Lisinopril 10mg daily for hypertension, but often misses
his Metformin 500mg dose which should be taken twice daily for diabetes.
"""

prompt_description = """
Extract medications with their details, using attributes to group related information:

1. Extract entities in the order they appear in the text
2. Each entity must have a 'medication_group' attribute linking it to its medication
3. All details about a medication should share the same medication_group value
"""


examples = [
    lx.data.ExampleData(
        text="Patient takes Aspirin 100mg daily for heart health and Simvastatin 20mg at bedtime.",
        extractions=[
            # First Medication Group
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Aspirin",
                attributes={"medication_group": "Aspirin"}  
            ),
            lx.data.Extraction(
                extraction_class="dosage",
                extraction_text="100mg",
                attributes={"medication_group": "Aspirin"}
            ),
            lx.data.Extraction(
                extraction_class="frequency",
                extraction_text="daily",
                attributes={"medication_group": "Aspirin"}
            ),
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text="heart health",
                attributes={"medication_group": "Aspirin"}
            ),

            # Second medication group
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Simvastatin",
                attributes={"medication_group": "Simvastatin"}
            ),
            lx.data.Extraction(
                extraction_class="dosage",
                extraction_text="20mg",
                attributes={"medication_group": "Simvastatin"}
            ),
            lx.data.Extraction(
                extraction_class="frequency",
                extraction_text="at bedtime",
                attributes={"medication_group": "Simvastatin"}
            )
        ]
    )
]

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt_description,
    examples=examples,
    model_id="gemini-2.5-pro",
    api_key=KEY 
)

print(f"Input text: {input_text.strip()}\n")
print("Extracted Medications:")

medication_groups = {}
for extraction in result.extractions:
    if not extraction.attributes or "medication_group" not in extraction.attributes:
        print(f"Warning: Missing medication_group for {extraction.extraction_text}")
        continue

    group_name = extraction.attributes["medication_group"]
    medication_groups.setdefault(group_name, []).append(extraction)

for med_name, extractions in medication_groups.items():
    print(f"\n* {med_name}")
    for extraction in extractions:
        position_info = ""
        if extraction.char_interval:
            start, end = extraction.char_interval.start_pos, extraction.char_interval.end_pos
            position_info = f" (pos: {start}-{end})"
        print(f"  • {extraction.extraction_class.capitalize()}: {extraction.extraction_text}{position_info}")

lx.io.save_annotated_documents(
    [result],
    output_name="medical_ner_extraction2.jsonl",
    output_dir="./langextract"
)

html_content = lx.visualize("langextract/medical_ner_extraction2.jsonl")
with open("medical_relationship_visualization.html", "w",encoding='utf-8') as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)

print("Interactive visualization saved to medical_relationship_visualization.html")