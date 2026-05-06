import os

os.environ["HF_HOME"] = "./models"
import json

import chromadb
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


class CacheManager:
    def __init__(self, chunks_cache_path: str = None, embedding_cache_path: str = None):
        # self.chunks_cache =_get_chunks_cache()
        pass

    def _save_cache(self):
        pass

    def _retrieve(self):
        pass

    def _get_chunks_cache(self):
        # _get_cache()
        pass

    def _get_embeddings_cache(self):
        pass

    def _get_cache(self, cache_path: str):
        pass


def prepare_csv(path: str):
    movie_dataframe = pd.read_csv(path)

    # Tri les films par popularité
    movie_dataframe = movie_dataframe.sort_values(by="popularity", ascending=False)
    movie_dataframe = movie_dataframe.head().copy()

    # Fill les valeurs manquantes
    # budget,genres,homepage,id,keywords,original_language,original_title,overview,popularity,production_companies,production_countries,release_date,revenue,runtime,spoken_languages,status,tagline,title,vote_average,vote_count
    # 237000000,"[{""id"": 28, ""name"": ""Action""}, {""id"": 12, ""name"": ""Adventure""}, {""id"": 14, ""name"": ""Fantasy""}, {""id"": 878, ""name"": ""Science Fiction""}]",http://www.avatarmovie.com/,19995,"[{""id"": 1463, ""name"": ""culture clash""}, {""id"": 2964, ""name"": ""future""}, {""id"": 3386, ""name"": ""space war""}, {""id"": 3388, ""name"": ""space colony""}, {""id"": 3679, ""name"": ""society""}, {""id"": 3801, ""name"": ""space travel""}, {""id"": 9685, ""name"": ""futuristic""}, {""id"": 9840, ""name"": ""romance""}, {""id"": 9882, ""name"": ""space""}, {""id"": 9951, ""name"": ""alien""}, {""id"": 10148, ""name"": ""tribe""}, {""id"": 10158, ""name"": ""alien planet""}, {""id"": 10987, ""name"": ""cgi""}, {""id"": 11399, ""name"": ""marine""}, {""id"": 13065, ""name"": ""soldier""}, {""id"": 14643, ""name"": ""battle""}, {""id"": 14720, ""name"": ""love affair""}, {""id"": 165431, ""name"": ""anti war""}, {""id"": 193554, ""name"": ""power relations""}, {""id"": 206690, ""name"": ""mind and soul""}, {""id"": 209714, ""name"": ""3d""}]",en,Avatar,"In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.",150.437577,"[{""name"": ""Ingenious Film Partners"", ""id"": 289}, {""name"": ""Twentieth Century Fox Film Corporation"", ""id"": 306}, {""name"": ""Dune Entertainment"", ""id"": 444}, {""name"": ""Lightstorm Entertainment"", ""id"": 574}]","[{""iso_3166_1"": ""US"", ""name"": ""United States of America""}, {""iso_3166_1"": ""GB"", ""name"": ""United Kingdom""}]",2009-12-10,2787965087,162,"[{""iso_639_1"": ""en"", ""name"": ""English""}, {""iso_639_1"": ""es"", ""name"": ""Espa\u00f1ol""}]",Released,Enter the World of Pandora.,Avatar,7.2,11800
    movie_dataframe["budget"] = movie_dataframe["budget"].fillna(0)
    movie_dataframe["genres"] = movie_dataframe["genres"].fillna("[]")
    movie_dataframe["homepage"] = movie_dataframe["homepage"].fillna("")
    movie_dataframe["id"] = movie_dataframe["id"].fillna(0)
    movie_dataframe["keywords"] = movie_dataframe["keywords"].fillna("[]")
    movie_dataframe["original_language"] = movie_dataframe["original_language"].fillna(
        ""
    )
    movie_dataframe["original_title"] = movie_dataframe["original_title"].fillna("")
    movie_dataframe["overview"] = movie_dataframe["overview"].fillna("")
    movie_dataframe["popularity"] = movie_dataframe["popularity"].fillna(0)
    movie_dataframe["production_companies"] = movie_dataframe[
        "production_companies"
    ].fillna("[]")
    movie_dataframe["production_countries"] = movie_dataframe[
        "production_countries"
    ].fillna("[]")
    movie_dataframe["release_date"] = movie_dataframe["release_date"].fillna("")
    movie_dataframe["revenue"] = movie_dataframe["revenue"].fillna(0)
    movie_dataframe["runtime"] = movie_dataframe["runtime"].fillna(0)
    movie_dataframe["spoken_languages"] = movie_dataframe["spoken_languages"].fillna(
        "[]"
    )
    movie_dataframe["status"] = movie_dataframe["status"].fillna("")
    movie_dataframe["tagline"] = movie_dataframe["tagline"].fillna("")
    movie_dataframe["title"] = movie_dataframe["title"].fillna("")
    movie_dataframe["vote_average"] = movie_dataframe["vote_average"].fillna(0)
    movie_dataframe["vote_count"] = movie_dataframe["vote_count"].fillna(0)

    print("First ten movies after sorting by popularity:")
    for _, row in movie_dataframe.head(10).iterrows():
        print(f"{row['title']} {row['popularity']}")

    return movie_dataframe


def chunking_embedding_dataframe(DataFrame: pd.DataFrame, embedder):
    documents = []
    metadatas = []
    embeddings = []

    # Je crée un chunking simple en concaténant les champs pertinents pour chaque film
    for row_index, row in DataFrame.iterrows():
        joined_fields, keywords_str, genres_str = chunking(row_index=row_index, row=row)

        documents.append(joined_fields)
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

        # TODO Add chunking and embeds cache logic
        embeddings.append(get_embedding(embedder=embedder, chunk=joined_fields))

    # 5 premiers films
    for i in range(5):
        print(documents[i])
        print("Metadata:", metadatas[i])
        print("\n")

    return documents, embeddings, metadatas


def chunking(row_index: int, row: dict):
    # Extraction le nom des genres
    try:
        genres_data = json.loads(row["genres"])
        genres_list = [genre["name"] for genre in genres_data]
        genres_str = ", ".join(genres_list)
    except KeyError as e:
        print(f"Error occurred while processing genres for row {row_index}: {e}")
        genres_str = ""
    # Extraction des keywords
    try:
        keywords = json.loads(row["keywords"])
        keywords_list = [keyword["name"] for keyword in keywords]
        keywords_str = ", ".join(keywords_list)
    except KeyError as e:
        print(f"Error occurred while processing keywords for row {row_index}: {e}")
        keywords_str = ""

    fields_list = [
        f"Titre: {row['title']}",
        f"Date de sortie: {row['release_date']}",
        f"Langue originale: {row['original_language']}",
        f"Note: {row['vote_average']}/10 ({row['vote_count']} votes)",
        f"Durée: {row['runtime']} minutes",
        f"Genres: {genres_str}",
        f"Keywords: {keywords_str}",
        f"Synopsis: {row['overview']}",
    ]
    joined_fields = "\n".join(fields_list)

    return joined_fields, keywords_str, genres_str


def get_embedding(embedder, chunk: str):
    embeddings = embedder.encode(
        chunk, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    ).tolist()
    return embeddings


def retrieve(question, sentence_transformer_object, collection, n=3):
    embedded_question = get_embedding(sentence_transformer_object, question)[0]

    results = collection.query(query_embeddings=[embedded_question], n_results=n)

    return results["documents"], results["metadatas"]


def main():
    embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    movie_dataframe = prepare_csv(path="data/tmdb_5000_movies.csv")

    documents, embeddings, metadatas = chunking_embedding_dataframe(
        movie_dataframe, embedder
    )
    ## TODO : Stocker les chunks dans un cache local pour éviter de devoir les recalculer à chaque fois
    ## TODO : Stocker les embeddings dans un cache local aussi

    ## Ajouter le if cache existe déjà, charger les chunks depuis le cache au lieu de les recalculer

    vector_db = chromadb.PersistentClient(path="./chroma_db")
    collection = vector_db.create_collection(name="Movies")

    collection.add(
        ids=[f"chunck_{id_chunck}" for id_chunck in range(len(documents))],
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    results = retrieve(
        question="Insterstellar", embedder=embedder, collection=collection
    )
    print(results)

    return results


if __name__ == "__main__":
    main()
