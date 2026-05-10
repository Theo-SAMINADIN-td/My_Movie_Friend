import json
import os

import chromadb
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, VECTOR_DB_NAME

load_dotenv()


class VectorDB:
    def __init__(self, vector_db_name: str, collection_name: str = "Movies"):
        """Initialize the connection to ChromaDB and SentenceTransformer."""
        self.vector_db_name = vector_db_name
        self.collection_name = collection_name
        
        self.chroma = chromadb.PersistentClient(path=vector_db_name)
        
        self.collection = self.chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": EMBEDDING_MODEL_NAME},
        )
        
        model_name = self.collection.metadata.get("embedding_model", EMBEDDING_MODEL_NAME)
        print(f"Embedding model : {model_name}")
        self.sentence_transformer_object = SentenceTransformer(model_name)

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def ingest(self, chunks: list[str], metadatas: list[dict]):
        print(f"Ingesting {len(chunks)} documents into the vector database...")
        embeddings = self.get_embeddings(chunks)
        
        ids = [str(meta.get("id", idx)) for idx, meta in enumerate(metadatas)]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def get_embeddings(self, chunks: list[str]) -> list[list[float]]:
        embeddings = self.sentence_transformer_object.encode(
            chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        ).tolist()
        return embeddings

    def retrieve(self, question: str, n: int = 3) -> tuple[list, list]:
        embedded_question = self.get_embeddings([question])[0]
        results = self.collection.query(query_embeddings=[embedded_question], n_results=n)
        return results["documents"], results["metadatas"]

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def build_chunks_from_csv(csv_path: str):
        df = VectorDB.prepare_csv(csv_path)
        chunks, metadatas = [], []
        for row_index, row in df.iterrows():
            joined_fields, keywords_str, genres_str = VectorDB.chunking(row)
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
        return chunks, metadatas


def main():
    chunks, metadatas = VectorDB.build_chunks_from_csv(r"data\tmdb_5000_movies.csv")
    vector_db_object = VectorDB(vector_db_name=VECTOR_DB_NAME)
    if vector_db_object.is_empty():
        vector_db_object.ingest(chunks, metadatas)

    docs, metas = vector_db_object.retrieve("Movie in france >7", n=10)
    for doc, _ in zip(docs[0], metas[0]):
        print(doc)


if __name__ == "__main__":
    main()
