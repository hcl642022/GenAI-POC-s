import os
import tempfile
import psycopg2
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google.cloud import storage
from google.generativeai import configure, GenerativeModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict, Any
import re
import json # Import the json module for JSON handling

# --- Configuration ---
# Load .env at the top-level
if os.path.exists(".env"):
    load_dotenv()
else:
    print("Warning: .env file not found. Ensure environment variables are set.")

# Database credentials
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# GCS bucket name
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Gemini API Configuration
gemini_embedding_model_name = "models/embedding-001"

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Client information
CLIENT_NAME = "Apex Global Services FZE"

# --- Database Connection ---
def get_db_connection() -> psycopg2.extensions.connection | None:
    """Establishes a connection to the PostgreSQL database."""
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        print("Database credentials (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD) are missing in .env.")
        return None

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port="5432"
        )
        print(f"Successfully connected to Cloud SQL via private IP: {DB_HOST}")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to the database via private IP: {e}")
        print("Ensure your local network can reach the Cloud SQL private IP and that the database is running.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during database connection: {e}")
        return None

# --- PDF Handling ---
def extract_text_from_pdf_local(file_path: str) -> str:
    """Extracts text from a local PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except FileNotFoundError:
        print(f"Error: Local file not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading local PDF {file_path}: {e}")
        return ""

def extract_text_from_pdf_gcs(bucket_name: str, blob_name: str) -> str:
    """Extracts text from a PDF file stored in Google Cloud Storage."""
    if not bucket_name:
        print("GCS bucket name is not configured.")
        return ""

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        fd, local_path = tempfile.mkstemp()
        os.close(fd)

        print(f"Downloading GCS file '{blob_name}' to temporary path: '{local_path}'")
        blob.download_to_filename(local_path)

        print(f"Processing temporary file: '{local_path}'")
        text = extract_text_from_pdf_local(local_path)
        return text
    except Exception as e:
        print(f"Error downloading or processing GCS file '{blob_name}': {e}")
        return ""
    finally:
        if 'local_path' in locals() and local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                print(f"Cleaned up temporary file: '{local_path}'")
            except Exception as cleanup_e:
                print(f"Error during cleanup of '{local_path}': {cleanup_e}")

def get_pdf_paths_from_gcs(bucket_name: str) -> List[str]:
    """Lists all PDF files in the specified GCS bucket."""
    if not bucket_name:
        print("GCS_BUCKET_NAME is not set in .env. Cannot fetch PDFs from GCS.")
        return []

    try:
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name)
        pdf_blobs = [blob.name for blob in blobs if blob.name.lower().endswith('.pdf')]
        return pdf_blobs
    except Exception as e:
        print(f"Error listing blobs in GCS bucket '{bucket_name}': {e}")
        return []

# --- Embedding and Chunking ---
def get_embeddings_model():
    """Initializes and returns the Gemini embeddings model using ADC."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        print("Gemini embeddings model initialized successfully using ADC.")
        return embeddings
    except Exception as e:
        print(f"Error initializing Gemini embeddings model: {e}")
        return None

def split_text_into_chunks(text: str) -> List[str]:
    """Splits text into manageable chunks."""
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def generate_embeddings(text_chunks: List[str], embeddings_model) -> List[List[float]]:
    """Generates embeddings for a list of text chunks."""
    if not embeddings_model:
        print("Embeddings model not initialized. Cannot generate embeddings.")
        return []
    if not text_chunks:
        print("No text chunks provided for embedding generation.")
        return []

    try:
        print(f"Generating embeddings for {len(text_chunks)} chunks...")
        embeddings = embeddings_model.embed_documents(text_chunks)
        print(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# --- Database Operations ---
def insert_client_if_not_exists(conn, client_name: str) -> int | None:
    """Inserts client into clients table if it doesn't exist."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT client_id FROM clients WHERE client_name = %s", (client_name,))
        client_id_record = cursor.fetchone()

        if client_id_record is None:
            cursor.execute("INSERT INTO clients (client_name) VALUES (%s) RETURNING client_id", (client_name,))
            client_id = cursor.fetchone()[0]
            conn.commit()
            print(f"Inserted client: '{client_name}' with ID: {client_id}")
        else:
            client_id = client_id_record[0]
            print(f"Client '{client_name}' already exists with ID: {client_id}")
        return client_id
    except psycopg2.Error as e:
        print(f"Database error inserting client '{client_name}': {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()

def insert_risk_dossier_corpus(conn, client_id: int, review_year: int, chunks_with_metadata: List[Dict[str, Any]]):
    """Inserts text chunks, embeddings, and metadata into the risk_dossier_corpus table."""
    if not chunks_with_metadata:
        print("No chunks with metadata to insert.")
        return

    cursor = conn.cursor()
    # Modified SQL to include the metadata column
    insert_sql = """
        INSERT INTO risk_dossier_corpus (client_id, review_year, chunk_text, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (client_id, review_year, chunk_text)
        DO NOTHING; -- Avoid duplicates
    """
    try:
        successful_inserts = 0
        for item in chunks_with_metadata:
            if not item or 'text' not in item or 'embedding' not in item or 'source_file' not in item:
                print(f"Skipping invalid chunk item: {item}")
                continue

            embedding_data = item['embedding']
            if not isinstance(embedding_data, list) or not all(isinstance(x, (int, float)) for x in embedding_data):
                print(f"Skipping chunk due to invalid embedding format. Text (start): '{item['text'][:50]}...'")
                continue

            # Convert the metadata dictionary to a JSON string for insertion into JSONB column
            metadata_json = json.dumps({"source_file": item['source_file']})

            cursor.execute(insert_sql, (client_id, review_year, item['text'], embedding_data, metadata_json))
            if cursor.rowcount > 0: # Check if a new row was inserted
                successful_inserts += 1

        conn.commit()
        print(f"Successfully inserted/updated {successful_inserts} new chunks for client ID {client_id}, year {review_year}.")

    except psycopg2.Error as e:
        print(f"Database error inserting chunks for client ID {client_id}, year {review_year}: {e}")
        conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred during chunk insertion for client ID {client_id}, year {review_year}: {e}")
        conn.rollback()
    finally:
        cursor.close()

# --- Main Ingestion Logic ---
def extract_year_from_filename(filename: str) -> int | None:
    """Extracts the year from the filename, e.g., '...-2020.pdf'"""
    match = re.search(r'-(\d{4})\.pdf$', filename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            print(f"Could not parse year from filename '{filename}'")
            return None
    else:
        print(f"Filename '{filename}' does not match expected year format.")
        return None

def ingest_pdfs_from_gcs():
    """Ingests PDFs from GCS and inserts them into the database."""
    conn = get_db_connection()
    if not conn:
        print("Failed to get database connection. Aborting GCS ingestion.")
        return

    client_id = insert_client_if_not_exists(conn, CLIENT_NAME)
    if not client_id:
        print("Failed to get or create client ID. Aborting GCS ingestion.")
        conn.close()
        return

    embeddings_model = get_embeddings_model()
    if not embeddings_model:
        print("Failed to initialize embeddings model. Aborting GCS ingestion.")
        conn.close()
        return

    gcs_pdf_files = get_pdf_paths_from_gcs(GCS_BUCKET_NAME)

    if not gcs_pdf_files:
        print("No PDF files found in the specified GCS bucket.")
        conn.close()
        return

    # This dictionary will store chunks grouped by year, with their source file
    # Structure: { year: [ { "text": "...", "embedding": [...], "source_file": "..." }, ... ], ... }
    chunks_by_year: Dict[int, List[Dict[str, Any]]] = {}

    for gcs_file_path in gcs_pdf_files:
        review_year = extract_year_from_filename(gcs_file_path)
        if review_year is None:
            print(f"Skipping file '{gcs_file_path}' due to invalid year format.")
            continue

        print(f"\n--- Processing GCS file: '{gcs_file_path}' for year {review_year} ---")
        pdf_text = extract_text_from_pdf_gcs(GCS_BUCKET_NAME, gcs_file_path)

        if not pdf_text:
            print(f"Skipping file '{gcs_file_path}' due to no text extracted or an error occurred during processing.")
            continue

        text_chunks = split_text_into_chunks(pdf_text)

        if not text_chunks:
            print(f"No text chunks generated from '{gcs_file_path}'.")
            continue

        embeddings = generate_embeddings(text_chunks, embeddings_model)

        current_file_chunk_data = []
        if len(text_chunks) == len(embeddings):
            for i, chunk_text in enumerate(text_chunks):
                current_file_chunk_data.append({
                    "text": chunk_text,
                    "embedding": embeddings[i],
                    "source_file": gcs_file_path # Add the source file to the chunk data
                })
            
            if review_year not in chunks_by_year:
                chunks_by_year[review_year] = []
            chunks_by_year[review_year].extend(current_file_chunk_data)
            print(f"Grouped {len(current_file_chunk_data)} chunks for year {review_year}.")
        else:
            print(f"Mismatch between number of chunks ({len(text_chunks)}) and embeddings ({len(embeddings)}) for '{gcs_file_path}'. Skipping embeddings for this file.")

    # Now, insert the collected chunks by year
    for year, chunks in chunks_by_year.items():
        print(f"\nAttempting to insert {len(chunks)} total chunks into the database for year {year}...")
        insert_risk_dossier_corpus(conn, client_id, year, chunks)

    conn.close()
    print("\nGCS data ingestion process completed.")

# This part is only executed when ingest_data.py is run directly as a script
if __name__ == "__main__":
    print("Running ingest_data.py as a script...")
    ingest_pdfs_from_gcs()
    print("ingest_data.py script finished.")