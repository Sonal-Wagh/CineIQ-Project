import streamlit as st
import pandas as pd, numpy as np
import pickle, ast, os
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gdown

st.set_page_config(page_title="CineIQ", page_icon="🎬", layout="wide")

ARTIFACTS_DIR = "/tmp/cineiq_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

FILES = {
    "cosine_sim.npy"  : "1oxnyuwfc6kI30HzA5Vuw3ofdZB4aRr_a",
    "U.npy"           : "1rXSf3Duw1vFyGawaehUZHcq6eRjkY4sY",
    "Vt.npy"          : "1jzYpaPgETtrABfimfG0paeu_hIqOQ3B3",
    "user_enc.pkl"    : "19IAV3MsFHTnkgPSc-vppVwpS8ELFHPsh",
    "movie_enc.pkl"   : "1HQSpFRDLMvMYGzxdTrIYI706EpxL3hXM",
    "tmdb_indexed.csv": "16NQQLylfr7GGD3KBJquLZqKT5E4nz5AV",
    "movies_df.csv"   : "1FsTBA6ZEYP6nS6QKnfXT6L2Kgjsf0Du9",
    "title_to_idx.csv": "1EDDQD3-vF8Ts14WJDv_CIigy7WTXi28H",
}

for filename, file_id in FILES.items():
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(filepath):
        st.info(f"Downloading {filename}...")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            filepath, quiet=False
        )

OUTPUT_DIR = ARTIFACTS_DIR

@st.cache_resource
def load_all():
    cosine_sim    = np.load(f"{OUTPUT_DIR}/cosine_sim.npy")
    U             = np.load(f"{OUTPUT_DIR}/U.npy")
    Vt            = np.load(f"{OUTPUT_DIR}/Vt.npy")
    reconstructed = U @ Vt
    with open(f"{OUTPUT_DIR}/user_enc.pkl",  "rb") as f: ue = pickle.load(f)
    with open(f"{OUTPUT_DIR}/movie_enc.pkl", "rb") as f: me = pickle.load(f)
    tmdb   = pd.read_csv(f"{OUTPUT_DIR}/tmdb_indexed.csv", index_col=0)
    movies = pd.read_csv(f"{OUTPUT_DIR}/movies_df.csv")
    tidx   = pd.read_csv(f"{OUTPUT_DIR}/title_to_idx.csv", index_col=0).iloc[:, 0]
    va     = SentimentIntensityAnalyzer()
    return cosine_sim, reconstructed, ue, me, tmdb, movies, tidx, va

cosine_sim, reconstructed, user_enc, movie_enc, \
    tmdb_indexed, movies_df, title_to_idx, analyzer = load_all()

def safe_parse(v):
    try: return ast.literal_eval(v) if isinstance(v, str) else (v or [])
    except: return []

tmdb_indexed["genres_list"] = tmdb_indexed["genres_list"].apply(safe_parse)

def predict_rating(user_id, movie_id):
    if user_id not in user_enc.classes_ or movie_id not in movie_enc.classes_:
        return 3.5
    u = user_enc.transform([user_id])[0]
    m = movie_enc.transform([movie_id])[0]
    return float(np.clip(reconstructed[u, m], 0.5, 5.0))

def normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0

def run_pipeline(seed, user_id=1, alpha=0.5, n=10):
    t = seed.lower()
    if t not in title_to_idx.index:
        return None
    idx  = title_to_idx[t]
    sims = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:80]
    c = tmdb_indexed.iloc[[i for i, _ in sims]].copy()
    c["content_score"] = [s for _, s in sims]
    ml = movies_df[["tmdbId", "movieId"]].dropna()
    if "id" in c.columns:
        c["tmdbId"] = c["id"].astype("Int64")
        c = c.merge(ml, on="tmdbId", how="left")
    c["svd_score"] = c.get("movieId", pd.Series([None] * len(c))).apply(
        lambda m: predict_rating(user_id, m) if pd.notnull(m) else 3.5)
    c["cn"] = normalize(c["content_score"])
    c["sn"] = normalize(c["svd_score"])
    c["ensemble_score"] = (1 - alpha) * c["cn"] + alpha * c["sn"]
    c["sent"] = c["title"].apply(
        lambda t2: analyzer.polarity_scores(str(t2))["compound"])
    c["final_score"] = 0.85 * c["ensemble_score"] + 0.15 * (c["sent"] + 1) / 2
    return c.nlargest(n, "final_score").reset_index(drop=True)

# ── UI ───────────────────────────────────────────────────────────
st.title("CineIQ 🎬")
st.caption("Explainable hybrid movie recommendations")

with st.sidebar:
    st.header("Settings")
    user_id = st.number_input("User ID", min_value=1, max_value=162000, value=1)
    alpha   = st.slider("SVD weight", 0.0, 1.0, 0.5, 0.05)
    n_recs  = st.slider("Recommendations", 5, 20, 10)

seed = st.text_input("Enter a movie you like:", "The Dark Knight")

if st.button("Get Recommendations"):
    with st.spinner("Finding recommendations..."):
        res = run_pipeline(seed, user_id, alpha, n_recs)
    if res is None:
        st.error(f"'{seed}' not found. Check spelling.")
    else:
        st.subheader(f"Top {n_recs} picks for '{seed}'")
        for i, row in res.iterrows():
            g = ", ".join(row["genres_list"]) if isinstance(row["genres_list"], list) else ""
            with st.expander(f"{i+1}. {row['title']}  —  score: {row['final_score']:.3f}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Content",   f"{row['content_score']:.2f}")
                c2.metric("SVD",       f"{row['svd_score']:.2f}/5")
                c3.metric("Sentiment", f"{row['sent']:.2f}")
                st.caption(f"Genres: {g}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Genre radar")
            all_g = [g2 for gl in res["genres_list"]
                     for g2 in (gl if isinstance(gl, list) else [])]
            if all_g:
                gc   = pd.Series(all_g).value_counts().head(8)
                cats = gc.index.tolist(); vals = gc.values.tolist()
                cats += cats[:1];         vals += vals[:1]
                fig  = go.Figure(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself", line_color="#5DCAA5"))
                fig.update_layout(showlegend=False, height=300,
                                  margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Score breakdown")
            fig2 = px.bar(res.reset_index(), x="title",
                          y=["content_score", "svd_score"],
                          barmode="group", height=300,
                          color_discrete_sequence=["#5DCAA5", "#7F77DD"])
            fig2.update_xaxes(tickangle=45, tickfont_size=9)
            fig2.update_layout(margin=dict(l=5, r=5, t=5, b=70))
            st.plotly_chart(fig2, use_container_width=True)
