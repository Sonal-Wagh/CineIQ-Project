from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd, numpy as np, pickle, ast, os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import gdown

ARTIFACTS_DIR = "/tmp/cineiq_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

FILES = {
    "tfidf_matrix.npz"   : "1Vpnl2TB1alQo7TPeECvtwD58iuvZ1ItR",
    "tfidf_vectorizer.pkl": "1dit9Gxluscf0ZU6Hw_Rk8jtrXfFe8QyB",
    "user_enc.pkl"       : "19IAV3MsFHTnkgPSc-vppVwpS8ELFHPsh",
    "movie_enc.pkl"      : "1HQSpFRDLMvMYGzxdTrIYI706EpxL3hXM",
    "tmdb_indexed.csv"   : "16NQQLylfr7GGD3KBJquLZqKT5E4nz5AV",
    "movies_df.csv"      : "1FsTBA6ZEYP6nS6QKnfXT6L2Kgjsf0Du9",
    "title_to_idx.csv"   : "1EDDQD3-vF8Ts14WJDv_CIigy7WTXi28H",
}

for filename, file_id in FILES.items():
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            filepath, quiet=False
        )

print("All artifacts ready!")

app = FastAPI(title="CineIQ API", version="1.0")

OUTPUT_DIR = ARTIFACTS_DIR

tfidf_matrix = sp.load_npz(f"{OUTPUT_DIR}/tfidf_matrix.npz")
with open(f"{OUTPUT_DIR}/tfidf_vectorizer.pkl", "rb") as f: tfidf      = pickle.load(f)
with open(f"{OUTPUT_DIR}/user_enc.pkl",         "rb") as f: user_enc   = pickle.load(f)
with open(f"{OUTPUT_DIR}/movie_enc.pkl",        "rb") as f: movie_enc  = pickle.load(f)
tmdb_indexed  = pd.read_csv(f"{OUTPUT_DIR}/tmdb_indexed.csv", index_col=0)
movies_df     = pd.read_csv(f"{OUTPUT_DIR}/movies_df.csv")
title_to_idx  = pd.read_csv(f"{OUTPUT_DIR}/title_to_idx.csv", index_col=0).iloc[:, 0]
analyzer      = SentimentIntensityAnalyzer()

def safe_parse(v):
    try: return ast.literal_eval(v) if isinstance(v, str) else []
    except: return []

tmdb_indexed["genres_list"] = tmdb_indexed["genres_list"].apply(safe_parse)

def normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0

def get_content_sims(idx, n=80):
    query_vec = tfidf_matrix[idx]
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][1:n+1]
    return [(i, sims[i]) for i in top_indices]

class RecRequest(BaseModel):
    seed_title: str
    user_id:    Optional[int]   = 1
    n:          Optional[int]   = 10
    alpha:      Optional[float] = 0.5

class MovieResult(BaseModel):
    rank:          int
    title:         str
    genres:        List[str]
    final_score:   float
    content_score: float
    svd_score:     float
    sentiment:     float
    explanation:   str

@app.get("/health")
def health():
    return {"status": "ok", "movies": len(tmdb_indexed)}

@app.post("/recommend", response_model=List[MovieResult])
def recommend(req: RecRequest):
    t = req.seed_title.lower()
    if t not in title_to_idx.index:
        raise HTTPException(404, f"Movie not found: {req.seed_title}")

    idx  = title_to_idx[t]
    sims = get_content_sims(idx, n=80)

    c = tmdb_indexed.iloc[[i for i, _ in sims]].copy()
    c["cs"] = [s for _, s in sims]

    ml = movies_df[["tmdbId", "movieId"]].dropna()
    if "id" in c.columns:
        c["tmdbId"] = c["id"].astype("Int64")
        c = c.merge(ml, on="tmdbId", how="left")

    c["svd"]  = 3.5
    c["cn"]   = normalize(c["cs"])
    c["ens"]  = c["cn"]
    c["sent"] = c["title"].apply(
        lambda t2: analyzer.polarity_scores(str(t2))["compound"])
    c["fs"]   = 0.85 * c["ens"] + 0.15 * (c["sent"] + 1) / 2

    top = c.nlargest(req.n, "fs").reset_index(drop=True)
    out = []
    for i, row in top.iterrows():
        sm  = tmdb_indexed[tmdb_indexed["title"].str.lower() == t]
        rm  = tmdb_indexed[tmdb_indexed["title"].str.lower() == str(row["title"]).lower()]
        exp = f"Recommended based on your interest in {req.seed_title}."
        if not sm.empty and not rm.empty:
            sg    = set(sm.iloc[0]["genres_list"]); rg = set(rm.iloc[0]["genres_list"])
            sd    = str(sm.iloc[0].get("director", ""))
            rd    = str(rm.iloc[0].get("director", ""))
            parts = []
            if sd and rd and sd.lower() == rd.lower() and sd != "nan":
                parts.append(f"directed by {sd}")
            if sg & rg:
                parts.append("is also a " + " and ".join(list(sg & rg)[:2]) + " film")
            if row["cs"] > 0.3:
                parts.append(f"{row['cs']:.0%} content match")
            if parts:
                exp = "Recommended because it " + ", and ".join(parts[:3]) + "."
        out.append(MovieResult(
            rank=i + 1,
            title=str(row["title"]),
            genres=row["genres_list"] if isinstance(row["genres_list"], list) else [],
            final_score=round(float(row["fs"]), 4),
            content_score=round(float(row["cs"]), 4),
            svd_score=3.5,
            sentiment=round(float(row["sent"]), 4),
            explanation=exp
        ))
    return out

@app.get("/similar")
def similar(title: str, n: int = 10):
    t = title.lower()
    if t not in title_to_idx.index:
        raise HTTPException(404, f"Not found: {title}")
    idx  = title_to_idx[t]
    sims = get_content_sims(idx, n=n)
    return {
        "seed": title,
        "similar": [
            {"title": tmdb_indexed.iloc[i]["title"],
             "genres": tmdb_indexed.iloc[i]["genres_list"],
             "content_score": round(float(s), 4)}
            for i, s in sims
        ]
    }
