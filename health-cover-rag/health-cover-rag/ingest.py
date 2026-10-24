import os
import pickle
import re

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DOCUMENT_FOLDER = "documents"
VECTOR_FOLDER = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


def extract_documents():
    documents = []

    for filename in os.listdir(DOCUMENT_FOLDER):

        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(DOCUMENT_FOLDER, filename)

        print(f"\nReading: {filename}")

        reader = PdfReader(filepath)

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if not text:
                continue

            text = clean_text(text)

            page_chunks = chunk_text(
                text,
                CHUNK_SIZE,
                CHUNK_OVERLAP
            )

            for chunk_number, chunk in enumerate(page_chunks, start=1):

                documents.append(
                    {
                        "text": chunk,
                        "source": filename,
                        "page": page_number,
                        "chunk": chunk_number,
                    }
                )

    return documents


def main():

    os.makedirs(VECTOR_FOLDER, exist_ok=True)

    documents = extract_documents()

    print(f"\nTotal chunks created: {len(documents)}")

    texts = [doc["text"] for doc in documents]

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings.astype("float32"))

    faiss.write_index(
        index,
        os.path.join(VECTOR_FOLDER, "health_index.faiss")
    )

    with open(
        os.path.join(VECTOR_FOLDER, "chunks.pkl"),
        "wb"
    ) as file:

        pickle.dump(documents, file)

    print("\n--------------------------------")
    print("Knowledge base successfully built!")
    print("--------------------------------")

    print(f"Documents processed: {len(set(d['source'] for d in documents))}")
    print(f"Chunks stored: {len(documents)}")


if __name__ == "__main__":
    main()