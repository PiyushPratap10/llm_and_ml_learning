import langextract as lx
import os

KEY = os.getenv('GEMINI_API_KEY')

# Define extraction task with examples
instructions = """
Extract person details from text:
- Full name
- Job title  
- Key action performed
"""

example = lx.data.ExampleData(
    text="Dr. Sarah Johnson, the lead researcher, discovered a new compound.",
    extractions=[
        lx.data.Extraction(
            extraction_class="person",
            extraction_text="Dr. Sarah Johnson", 
            attributes={
                "title": "lead researcher",
                "action": "discovered a new compound"
            }
        )
    ]
)

# Extract from new text
result = lx.extract(
    text_or_documents="Engineer Alice Williams designed the software architecture.",
    prompt_description=instructions,
    examples=[example],
    model_id="gemini-2.5-flash",
    api_key=KEY
)

# Access structured results with source grounding
for extraction in result.extractions:
    print(f"{extraction.extraction_class}: {extraction.extraction_text}")
    print(f"Attributes: {extraction.attributes}")
    # print(f"Source position: {extraction.char_start}-{extraction.char_end}")

# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir="./langextract")

# Generate the visualization from the file
html_content = lx.visualize("langextract/extraction_results.jsonl")
with open("visualization.html", "w", encoding="utf-8") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)