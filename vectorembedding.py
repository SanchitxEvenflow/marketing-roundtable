import os
import re
from neo4j import GraphDatabase
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ==========================
# Configuration
# ==========================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

MARKDOWN_FOLDER = "Copywrite Research/Copywrite Research"

# ==========================
# Clients
# ==========================

client = genai.Client(api_key=GOOGLE_API_KEY)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ==========================
# Embedding Function
# ==========================

EMBEDDING_MODEL = "gemini-embedding-2"


def get_embedding(text: str, file_name: str = "?"):
    print(f"Requesting embedding: model={EMBEDDING_MODEL} file={file_name}")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values


# ==========================
# Extract Persona Name
# ==========================

def extract_name(content: str):
    match = re.search(r"# (.*?) Copywriter Profile", content)

    if match:
        return match.group(1).strip()

    return "Unknown"


# ==========================
# Import One Persona
# ==========================

def import_persona(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    name = extract_name(content)
    file_name = os.path.basename(file_path)

    print(f"Embedding: {name} (file={file_name})")

    embedding = get_embedding(content, file_name=file_name)

    with driver.session(database=NEO4J_DATABASE) as session:

        result = session.execute_write(lambda tx: tx.run(
            """
            MERGE (p:Persona {file_name:$file_name})

            SET
                p.name = $name,
                p.full_profile = $content,
                p.raw_md = $content,
                p.embedding = $embedding,
                p.imported_at = datetime()

            RETURN p.name, p.file_name
            """,
            name=name,
            content=content,
            embedding=embedding,
            file_name=file_name,
        ).single())

    print(f"Imported: {result['p.name']} (file={result['p.file_name']}, db: {NEO4J_DATABASE})")


# ==========================
# Import All Markdown Files
# ==========================

def main():

    if not os.path.exists(MARKDOWN_FOLDER):
        print(f"Folder not found: {MARKDOWN_FOLDER}")
        return

    md_files = [f for f in os.listdir(MARKDOWN_FOLDER) if f.endswith(".md")]
    print(f"Found {len(md_files)} .md files in '{MARKDOWN_FOLDER}'")

    imported = 0
    for filename in md_files:

        file_path = os.path.join(MARKDOWN_FOLDER, filename)

        try:
            import_persona(file_path)
            imported += 1

        except Exception as e:
            print(f"Failed to import {filename}")
            print(e)

    with driver.session(database=NEO4J_DATABASE) as session:
        count = session.run("MATCH (p:Persona) RETURN count(p) AS total").single()["total"]
        print(f"\nFinal Persona count in db '{NEO4J_DATABASE}': {count}")

    driver.close()

    print(f"\n{imported}/{len(md_files)} personas imported successfully.")


# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":
    main()