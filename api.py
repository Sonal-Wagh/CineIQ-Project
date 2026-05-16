from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd, numpy as np, pickle, ast, os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="CineIQ API", version="1.0")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# ── Load artifacts once at startup ──────────────────────────────
cosine_sim    = np.load(f"{OUTPUT_DIR}/cosine_sim.npy")
U             = np.load(f"{OUTPUT_DIR}/U.npy")
Vt            = np.load(f"{OUTPUT_DIR}/Vt.npy")
reconstructed = U @ Vt
with open(f"{OUTPUT_DIR}/user_enc.pkl",  "rb") as f: user_enc  = pickle.load(f)
with open(f"{OUTPUT_DIR}/movie_enc.pkl", "rb") as f: movie_enc = pickle.load(f)
tmdb_indexed  = pd.read_csv(f"{OUTPUT_DIR}/tmdb_indexed.csv", index_col=0)
movies_df     = pd.read_csv(f"{OUTPUT_DIR}/movies_df.csv")
title_to_idx  = pd.read_csv(f"{OUTPUT_DIR}/title_to_idx.csv", index_col=0).iloc[:, 0]
analyzer      = SentimentIntensityAnalyzer()

def safe_parse(v):
    try: return ast.literal_eval(v) if isinstance(v, str) else []
    except: return []

tmdb_indexed["genres_list"] = tmdb_indexed["genres_list"].apply(safe_parse)

def predict_rating(user_id, movie_id):
    if user_id not in user_enc.classes_ or movie_id not in movie_enc.classes_:
        return 3.5
    u = user_enc.transform([user_id])[0]
    m = movie_enc.transform([movie_id])[0]
    return float(np.clip(reconstructed[u, m], 0.5, 5.0))

def norm(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0

# ── Schemas ──────────────────────────────────────────────────────
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

# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "movies": len(tmdb_indexed)}

@app.post("/recommend", response_model=List[MovieResult])
def recommend(req: RecRequest):
    t = req.seed_title.lower()
    if t not in title_to_idx.index:
        raise HTTPException(404, f"Movie not found: {req.seed_title}")

    idx  = title_to_idx[t]
    sims = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:80]

    c = tmdb_indexed.iloc[[i for i, _ in sims]].copy()
    c["cs"]   = [s for _, s in sims]
    beta      = 1.0 - req.alpha

    ml = movies_df[["tmdbId", "movieId"]].dropna()
    if "id" in c.columns:
        c["tmdbId"] = c["id"].astype("Int64")
        c = c.merge(ml, on="tmdbId", how="left")

    c["svd"]  = c.get("movieId", pd.Series([None] * len(c))).apply(
        lambda m: predict_rating(req.user_id, m) if pd.notnull(m) else 3.5)
    c["cn"]   = norm(c["cs"]); c["sn"] = norm(c["svd"])
    c["ens"]  = req.alpha * c["cn"] + beta * c["sn"]
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
            svd_score=round(float(row["svd"]), 2),
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
    sims = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    return {
        "seed": title,
        "similar": [
            {"title": tmdb_indexed.iloc[i]["title"],
             "genres": tmdb_indexed.iloc[i]["genres_list"],
             "content_score": round(float(s), 4)}
            for i, s in sims
        ]
    }