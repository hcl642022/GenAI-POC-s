import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] 
from google.cloud import documentai_v1 as documentai

# Set these values
project_id = "deutschebank-aipocs"
location = "us"  # or "us-central1"
processor_id = "153123c888659c2c"
file_path = "gs://invoice_parser_db_bk/DB_POC_data/Grand_Hotel_Data.pdf"
mime_type = "application/pdf"
 
# Create the API client
client = documentai.DocumentProcessorServiceClient()
name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
 
# Read the PDF invoice
with open(file_path, "rb") as document_file:
    document_content = document_file.read()
 
# Prepare the request
raw_document = documentai.RawDocument(content=document_content, mime_type=mime_type)
request = documentai.ProcessRequest(name=name, raw_document=raw_document)
 
# Process the document
result = client.process_document(request=request)
document = result.document
 
# Display extracted fields
print("Extracted Fields:")
for entity in document.entities:
    print(f"{entity.type_}: {entity.mention_text}")
