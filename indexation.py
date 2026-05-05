import pandas as pd
import json 
def main():
    df = pd.read_csv("data/tmdb_5000_movies.csv")
    
    # Tri les films par popularité
    df = df.sort_values(by="popularity", ascending=False)
    df = df.head().copy()
    
    # Fill les valeurs manquantes
    # budget,genres,homepage,id,keywords,original_language,original_title,overview,popularity,production_companies,production_countries,release_date,revenue,runtime,spoken_languages,status,tagline,title,vote_average,vote_count
    # 237000000,"[{""id"": 28, ""name"": ""Action""}, {""id"": 12, ""name"": ""Adventure""}, {""id"": 14, ""name"": ""Fantasy""}, {""id"": 878, ""name"": ""Science Fiction""}]",http://www.avatarmovie.com/,19995,"[{""id"": 1463, ""name"": ""culture clash""}, {""id"": 2964, ""name"": ""future""}, {""id"": 3386, ""name"": ""space war""}, {""id"": 3388, ""name"": ""space colony""}, {""id"": 3679, ""name"": ""society""}, {""id"": 3801, ""name"": ""space travel""}, {""id"": 9685, ""name"": ""futuristic""}, {""id"": 9840, ""name"": ""romance""}, {""id"": 9882, ""name"": ""space""}, {""id"": 9951, ""name"": ""alien""}, {""id"": 10148, ""name"": ""tribe""}, {""id"": 10158, ""name"": ""alien planet""}, {""id"": 10987, ""name"": ""cgi""}, {""id"": 11399, ""name"": ""marine""}, {""id"": 13065, ""name"": ""soldier""}, {""id"": 14643, ""name"": ""battle""}, {""id"": 14720, ""name"": ""love affair""}, {""id"": 165431, ""name"": ""anti war""}, {""id"": 193554, ""name"": ""power relations""}, {""id"": 206690, ""name"": ""mind and soul""}, {""id"": 209714, ""name"": ""3d""}]",en,Avatar,"In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.",150.437577,"[{""name"": ""Ingenious Film Partners"", ""id"": 289}, {""name"": ""Twentieth Century Fox Film Corporation"", ""id"": 306}, {""name"": ""Dune Entertainment"", ""id"": 444}, {""name"": ""Lightstorm Entertainment"", ""id"": 574}]","[{""iso_3166_1"": ""US"", ""name"": ""United States of America""}, {""iso_3166_1"": ""GB"", ""name"": ""United Kingdom""}]",2009-12-10,2787965087,162,"[{""iso_639_1"": ""en"", ""name"": ""English""}, {""iso_639_1"": ""es"", ""name"": ""Espa\u00f1ol""}]",Released,Enter the World of Pandora.,Avatar,7.2,11800
    df['budget'] = df['budget'].fillna(0)
    df['genres'] = df['genres'].fillna("[]")
    df['homepage'] = df['homepage'].fillna("")
    df['id'] = df['id'].fillna(0)
    df['keywords'] = df['keywords'].fillna("[]")
    df['original_language'] = df['original_language'].fillna("")
    df['original_title'] = df['original_title'].fillna("")
    df['overview'] = df['overview'].fillna("")
    df['popularity'] = df['popularity'].fillna(0)
    df['production_companies'] = df['production_companies'].fillna("[]")
    df['production_countries'] = df['production_countries'].fillna("[]")
    df['release_date'] = df['release_date'].fillna("")
    df['revenue'] = df['revenue'].fillna(0)
    df['runtime'] = df['runtime'].fillna(0)
    df['spoken_languages'] = df['spoken_languages'].fillna("[]")
    df['status'] = df['status'].fillna("")
    df['tagline'] = df['tagline'].fillna("")
    df['title'] = df['title'].fillna("")
    df['vote_average'] = df['vote_average'].fillna(0)
    df['vote_count'] = df['vote_count'].fillna(0)

    print("First ten movies after sorting by popularity:")
    for _, row in df.head(10).iterrows():
        print(f"{row['title']} (Popularity: {row['popularity']})")
    
    documents = []
    metadatas = []
    
    # Je crée un chunking simple en concaténant les champs pertinents pour chaque film
    for row_index, row in df.iterrows():
        # Extraction le nom des genres
        try:
            genres_data = json.loads(row['genres'])
            genres_list = [genre['name'] for genre in genres_data]
            genres_str = ", ".join(genres_list)
        except KeyError as e:
            print(f"Error occurred while processing genres for row {row_index}: {e}")
            genres_str = ""
        # Extraction des keywords
        try :
            keywords = json.loads(row['keywords'])
            keywords_list = [keyword['name'] for keyword in keywords]
            keywords_str = ", ".join(keywords_list)
        except KeyError as e:
            print(f"Error occurred while processing keywords for row {row_index}: {e}")
            keywords_str = ""
            
        text_parts = [
            f"Titre: {row['title']}",
            f"Date de sortie: {row['release_date']}",
            f"Langue originale: {row['original_language']}",
            f"Note: {row['vote_average']}/10 ({row['vote_count']} votes)",
            f"Durée: {row['runtime']} minutes",
            f"Genres: {genres_str}",
            f"Keywords: {keywords_str}",
            f"Synopsis: {row['overview']}"
        ]
        text = "\n".join(text_parts)
        
        documents.append(text)
        metadatas.append({
            "id": int(row_index),
            "title": row['title'],
            "vote_average": row['vote_average'],
            "original_language": row['original_language'],
            "release_date": row['release_date'],
            "genres": genres_str,
            "keywords": keywords_str
        })
    
    # 5 premiers films
    for i in range(5):
        print(documents[i])
        print("Metadata:", metadatas[i])
        print("\n")
        
    ## TODO : Stocker les chunks dans un cache local pour éviter de devoir les recalculer à chaque fois
    ## Ajouter le if cache existe déjà, charger les chunks depuis le cache au lieu de les recalculer
    
if __name__ == "__main__":
    main()
