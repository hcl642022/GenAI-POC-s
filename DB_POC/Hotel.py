from google.cloud import documentai_v1 as documentai
import os
 
# Service account key setup (optional if already set)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\Users\aniket.nandk\Desktop\Db_Pocs\operations.json"
# Update these values
project_id = "deutschebank-aipocs"
location = "us"  # or "us-central1"
processor_id = "153123c888659c2c"
gcs_input_uri = "gs://invoice_parser_db_bk/DB_POC_data/Grand_Hotel_Data.pdf"
 
# Document AI client
client = documentai.DocumentProcessorServiceClient()
 
# Full processor resource name
name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
 
# GCS input document
gcs_document = documentai.GcsDocument(
    gcs_uri=gcs_input_uri, mime_type="application/pdf"
)
 
input_config = documentai.DocumentInputConfig(gcs_document=gcs_document)
 
request = documentai.ProcessRequest(
    name=name,
    input_documents=documentai.BatchDocumentsInputConfig(gcs_documents=documentai.GcsDocuments(documents=[gcs_document]))
)
 
# Process the document
result = client.process_document(request=request)
document = result.document
 
# Display results
print("Extracted Fields:")
for entity in document.entities:
    print(f"{entity.type_}: {entity.mention_text}")