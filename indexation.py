import json
import os

import chromadb
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, VECTOR_DB_NAME

load_dotenv()


class VectorDB:
    def __init__(self, vector_db_name, chunks=None):
        if os.path.exists(vector_db_name):
            self.load_vector_db(vector_db_name)

        elif chunks:
            self.create_vector_db(vector_db_name, chunks)

        else:
            raise (
                Exception(
                    "Can't initiate vector db object ! please give a path to a vector db / chuncks."
                )
            )

    def create_vector_db(self, vector_db_name, chuncks):
        print("Création de ma base de donnée vectorielle")
        print(f"Embedding model : {EMBEDDING_MODEL_NAME}")
        self.sentence_transformer_object = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.chroma = chromadb.PersistentClient(path=vector_db_name)
        collection = self.chroma.get_or_create_collection(
            name="Movies",
            metadata={
                "embedding_model": EMBEDDING_MODEL_NAME,
            },
        )

        embeddings = self.get_embeddings(chuncks)

        collection.add(
            ids=[f"chunck_{id_chunck}" for id_chunck in range(len(chuncks))],
            documents=chuncks,
            embeddings=embeddings,
            metadatas=[{"source": "doc1.pdf"} for _ in range(len(chuncks))],
        )

    def load_vector_db(self, vector_db_name):
        print("Chargement de ma base données vectorielle")
        self.chroma = chromadb.PersistentClient(path=vector_db_name)
        collection_info = self.chroma.get_collection("Movies")
        EMBEDDING_MODEL_NAME = collection_info.metadata["embedding_model"]
        print(f"Embedding model : {EMBEDDING_MODEL_NAME}")
        self.sentence_transformer_object = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def get_embeddings(self, chuncks):
        embeddings = self.sentence_transformer_object.encode(
            chuncks, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        ).tolist()
        return embeddings

    def retrieve(self, question, n=3):
        embedded_question = self.get_embeddings([question])[0]

        collection = self.chroma.get_or_create_collection("Movies")

        results = collection.query(query_embeddings=[embedded_question], n_results=n)

        return results["documents"], results["metadatas"]


def prepare_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(by="popularity", ascending=False).copy()

    defaults = {
        "budget": 0,
        "genres": "[]",
        "homepage": "",
        "id": 0,
        "keywords": "[]",
        "original_language": "",
        "original_title": "",
        "overview": "",
        "popularity": 0,
        "production_companies": "[]",
        "production_countries": "[]",
        "release_date": "",
        "revenue": 0,
        "runtime": 0,
        "spoken_languages": "[]",
        "status": "",
        "tagline": "",
        "title": "",
        "vote_average": 0,
        "vote_count": 0,
    }
    for col, val in defaults.items():
        df[col] = df[col].fillna(val)

    print("Top movies by popularity:")
    for _, row in df.head(10).iterrows():
        print(f"  {row['title']} — {row['popularity']:.2f}")

    return df


def chunking(row: dict) -> tuple[str, str, str]:
    try:
        genres_str = ", ".join(g["name"] for g in json.loads(row["genres"]))
    except (KeyError, json.JSONDecodeError):
        genres_str = ""

    try:
        keywords_str = ", ".join(k["name"] for k in json.loads(row["keywords"]))
    except (KeyError, json.JSONDecodeError):
        keywords_str = ""

    joined_fields = "\n".join(
        [
            f"Titre: {row['title']}",
            f"Date de sortie: {row['release_date']}",
            f"Langue originale: {row['original_language']}",
            f"Note: {row['vote_average']}/10 ({row['vote_count']} votes)",
            f"Durée: {row['runtime']} minutes",
            f"Genres: {genres_str}",
            f"Keywords: {keywords_str}",
            f"Synopsis: {row['overview']}",
        ]
    )
    return joined_fields, keywords_str, genres_str


def get_embedding(embedder: SentenceTransformer, chunk: str):
    if not chunk or not isinstance(chunk, str):
        raise ValueError("Chunk must be a non-empty string.")
    return embedder.encode(
        [chunk], batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )[0]


def retrieve(
    question: str, embedder: SentenceTransformer, collection, n: int = 3
) -> tuple[list, list]:
    embedded_question = get_embedding(embedder, question)
    results = collection.query(query_embeddings=[embedded_question], n_results=n)
    return results["documents"], results["metadatas"]


def main():
    df = prepare_csv(r"data\tmdb_5000_movies.csv")

    chunks, metadatas = [], []
    for row_index, row in df.iterrows():
        joined_fields, keywords_str, genres_str = chunking(row)
        chunks.append(joined_fields)
        metadatas.append(
            {
                "id": int(row_index),
                "title": row["title"],
                "vote_average": row["vote_average"],
                "original_language": row["original_language"],
                "release_date": row["release_date"],
                "genres": genres_str,
                "keywords": keywords_str,
            }
        )
    vector_db_object = VectorDB(vector_db_name=VECTOR_DB_NAME, chunks=chunks)

    docs, metas = vector_db_object.retrieve("Movie in france >7", n=10)
    for doc, _ in zip(docs[0], metas[0]):
        print(doc)


if __name__ == "__main__":
    main()
