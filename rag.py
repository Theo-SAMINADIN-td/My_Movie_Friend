import json
import os

from dotenv import load_dotenv
from groq import Groq

from config import LLM_MODEL_NAME, VECTOR_DB_NAME
from indexation import VectorDB, prepare_csv, chunking


class RAG:
    def __init__(self, vector_db_name: str, csv_path: str = None):
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.vector_db_object = self._load_or_build_db(vector_db_name, csv_path)

    def _load_or_build_db(self, vector_db_name: str, csv_path: str = None) -> VectorDB:
        if os.path.exists(vector_db_name):
            print(f"[RAG] Loading existing DB: {vector_db_name}")
            return VectorDB(vector_db_name)

        if csv_path is None:
            raise ValueError(
                f"DB '{vector_db_name}' not found. Provide csv_path to build it automatically."
            )

        print(f"DB not found, building from {csv_path}...")
        df = prepare_csv(csv_path)

        chunks, metadatas = [], []
        for _, row in df.iterrows():
            joined_fields, keywords_str, genres_str = chunking(row)
            chunks.append(joined_fields)
            metadatas.append({
                "id": int(row.name),
                "title": row["title"],
                "vote_average": row["vote_average"],
                "original_language": row["original_language"],
                "release_date": row["release_date"],
                "genres": genres_str,
                "keywords": keywords_str,
            })

        return VectorDB(vector_db_name=vector_db_name, chunks=chunks)

    @staticmethod
    def read_file(file_path):
        with open(file_path, "r") as file:
            return file.read()

    def build_context(self, question):
        context = RAG.read_file(file_path="context.txt")
        chunks = self.vector_db_object.retrieve(question, n=3)[0]
        return context.replace("{{Chuncks}}", str(chunks))

    def answer_question(self, question):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.build_context(question)},
                {"role": "user", "content": question},
            ],
            model=LLM_MODEL_NAME,
        )
        return chat_completion.choices[0].message.content


if __name__ == "__main__":
    rag_object = RAG(
        vector_db_name=VECTOR_DB_NAME,
        csv_path=r"data\tmdb_5000_movies.csv",
    )
    response = rag_object.answer_question("Un film à propos de l'espace")
    print(response)