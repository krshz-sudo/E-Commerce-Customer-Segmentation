"""
E-Commerce Customer Segmentation using RFM, K-Means, DBSCAN, PCA and t-SNE

Run:  python src/rfm_segmentation.py
If data/online_retail.csv (UCI Online Retail, columns: InvoiceNo, CustomerID,
InvoiceDate, Quantity, UnitPrice) exists it is used; otherwise a synthetic
transaction dataset with the same schema is generated.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- 1. DATA
def generate_synthetic_transactions(n_customers=3000, seed=SEED):
    """Synthetic online-retail transactions with 5 hidden customer types."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2011-12-09")
    start = end - pd.Timedelta(days=365)
    # (share, orders_lambda, avg_basket_log_mean, recency_scale_days, name)
    types = [
        (0.12, 14, 5.2, 20, "champions"),
        (0.25, 6, 4.4, 60, "loyal"),
        (0.20, 1.5, 3.9, 40, "new"),
        (0.33, 2, 4.0, 220, "hibernating"),
        (0.10, 3, 5.6, 150, "big_spender_at_risk"),
    ]
    rows = []
    cid = 12000
    for share, lam, mu, rec_scale, _ in types:
        for _ in range(int(n_customers * share)):
            cid += 1
            n_orders = max(1, rng.poisson(lam))
            last_gap = min(rng.exponential(rec_scale), 360)
            last_day = end - pd.Timedelta(days=float(last_gap))
            span = rng.uniform(30, 330)
            for k in range(n_orders):
                day = last_day - pd.Timedelta(days=float(rng.uniform(0, span)) if k else 0)
                day = max(day, start)
                n_items = rng.integers(1, 8)
                for _ in range(n_items):
                    rows.append((f"INV{cid}-{k}", cid, day,
                                 int(rng.integers(1, 12)),
                                 float(np.round(np.exp(rng.normal(mu - 2.0, 0.5)), 2))))
    df = pd.DataFrame(rows, columns=["InvoiceNo", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice"])
    return df


def load_transactions():
    path = os.path.join(DATA, "online_retail.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="ISO-8859-1", parse_dates=["InvoiceDate"])
        df = df.dropna(subset=["CustomerID"])
        df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]
        df = df[~df.InvoiceNo.astype(str).str.startswith("C")]
        src = "UCI Online Retail (real)"
    else:
        df = generate_synthetic_transactions()
        src = "Synthetic transactions (same schema as UCI Online Retail)"
    return df, src


# ---------------------------------------------------------------- 2. RFM
def build_rfm(df):
    df = df.copy()
    df["Revenue"] = df.Quantity * df.UnitPrice
    snapshot = df.InvoiceDate.max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    return rfm


def preprocess(rfm):
    logged = np.log1p(rfm)          # tames right skew of F and M
    scaled = StandardScaler().fit_transform(logged)
    return logged, scaled


# ---------------------------------------------------------------- 3. MODELS
def choose_k(X, k_range=range(2, 10)):
    res = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        res.append((k, km.inertia_, silhouette_score(X, km.labels_),
                    davies_bouldin_score(X, km.labels_), calinski_harabasz_score(X, km.labels_)))
    return pd.DataFrame(res, columns=["k", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz"])


def tune_dbscan(X):
    """k-distance heuristic + small grid scored by silhouette (non-noise points)."""
    nn = NearestNeighbors(n_neighbors=10).fit(X)
    kdist = np.sort(nn.kneighbors(X)[0][:, -1])
    best = None
    for eps in np.arange(0.2, 1.01, 0.1):
        for ms in (5, 10, 15, 20):
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            mask = lab != -1
            ncl = len(set(lab[mask]))
            if ncl < 2 or mask.sum() < 0.6 * len(X):
                continue
            s = silhouette_score(X[mask], lab[mask])
            if best is None or s > best[0]:
                best = (s, round(eps, 2), ms, ncl, 1 - mask.mean())
    return kdist, best


# ---------------------------------------------------------------- 4. PERSONAS
def name_personas(profile):
    """Rule-based persona naming from cluster means (original RFM units)."""
    names = {}
    med = profile[["Recency", "Frequency", "Monetary"]].median()
    for c, r in profile.iterrows():
        if r.Recency <= med.Recency and r.Frequency > med.Frequency and r.Monetary > med.Monetary:
            n = "Champions"
        elif r.Recency <= med.Recency and r.Frequency > med.Frequency:
            n = "Loyal Regulars"
        elif r.Recency <= med.Recency:
            n = "New / Promising"
        elif r.Monetary > med.Monetary:
            n = "At-Risk Big Spenders"
        else:
            n = "Hibernating / Lost"
        names[c] = n
    # ensure unique names
    seen = {}
    for c in list(names):
        seen[names[c]] = seen.get(names[c], 0) + 1
        if seen[names[c]] > 1:
            names[c] = f"{names[c]} #{seen[names[c]]}"
    return names


ACTIONS = {
    "Champions": "VIP programme, early access to launches, referral rewards. Do not discount; reward loyalty.",
    "Loyal Regulars": "Cross-sell and bundles, tiered loyalty points, nudge toward Champions via spend thresholds.",
    "New / Promising": "Welcome series, second-purchase incentive within 30 days, onboarding emails.",
    "At-Risk Big Spenders": "Personal win-back outreach, high-value coupon, 'we miss you' campaign, feedback survey.",
    "Hibernating / Lost": "Low-cost reactivation email, clearance offers; suppress after 2 failed attempts to save budget.",
}


def action_for(name):
    return ACTIONS.get(name.split(" #")[0], "Monitor and test targeted offers.")


# ---------------------------------------------------------------- MAIN
def main():
    df, src = load_transactions()
    rfm = build_rfm(df)
    logged, X = preprocess(rfm)
    print(f"Data source : {src}\nCustomers   : {len(rfm)}\n")
    print(rfm.describe().round(2), "\n")

    # --- EDA: skew before/after log
    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    for i, c in enumerate(rfm.columns):
        ax[0, i].hist(rfm[c], bins=40, color="#4C72B0"); ax[0, i].set_title(f"{c} (raw)")
        ax[1, i].hist(logged[c], bins=40, color="#55A868"); ax[1, i].set_title(f"{c} (log1p)")
    plt.tight_layout(); plt.savefig(f"{OUT}/01_distributions.png"); plt.close()

    # --- PCA
    pca_full = PCA().fit(X)
    evr = pca_full.explained_variance_ratio_
    print("PCA explained variance:", evr.round(3), "cumulative:", np.cumsum(evr).round(3))
    pca2 = PCA(n_components=2, random_state=SEED)
    P = pca2.fit_transform(X)
    loadings = pd.DataFrame(pca2.components_.T, index=rfm.columns, columns=["PC1", "PC2"]).round(3)
    print("\nPCA loadings:\n", loadings, "\n")

    # --- K selection
    ks = choose_k(X)
    print(ks.round(3), "\n")
    best_k = int(ks.loc[ks.silhouette.idxmax(), "k"])
    # Marketing needs more than 2-3 actionable groups: restrict to k in 4..6
    # and pick the best silhouette inside that business-usable range.
    best_k = int(ks[ks.k.between(4, 6)].sort_values("silhouette", ascending=False).iloc[0].k)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks.k, ks.inertia, "o-"); ax[0].axvline(best_k, ls="--", c="grey")
    ax[0].set_title("Elbow method (inertia)"); ax[0].set_xlabel("k")
    ax[1].plot(ks.k, ks.silhouette, "o-", c="#C44E52"); ax[1].axvline(best_k, ls="--", c="grey")
    ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
    plt.tight_layout(); plt.savefig(f"{OUT}/02_elbow_silhouette.png"); plt.close()

    # --- K-Means
    km = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(X)
    rfm["KMeans"] = km.labels_
    print(f"Chosen k = {best_k}; silhouette = {silhouette_score(X, km.labels_):.3f}, "
          f"Davies-Bouldin = {davies_bouldin_score(X, km.labels_):.3f}\n")

    # --- DBSCAN
    kdist, best = tune_dbscan(X)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kdist); ax.set_title("k-distance graph (k=10) for DBSCAN eps"); ax.set_ylabel("distance")
    plt.tight_layout(); plt.savefig(f"{OUT}/03_kdistance.png"); plt.close()
    if best:
        s, eps, ms, ncl, noise = best
        db_lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        print(f"DBSCAN best: eps={eps}, min_samples={ms}, clusters={ncl}, noise={noise:.1%}, silhouette={s:.3f}\n")
    else:
        eps, ms = 0.5, 10
        db_lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        print("DBSCAN fallback eps=0.5, min_samples=10\n")
    rfm["DBSCAN"] = db_lab

    # --- t-SNE
    ts = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=SEED)
    T = ts.fit_transform(X)

    def scatter_pair(coords, name, fname):
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for a, lab, title in ((ax[0], rfm.KMeans.values, "K-Means"), (ax[1], rfm.DBSCAN.values, "DBSCAN (-1 = noise)")):
            sc = a.scatter(coords[:, 0], coords[:, 1], c=lab, cmap="tab10", s=8, alpha=.75)
            a.set_title(f"{name}: {title}"); a.set_xlabel(f"{name} 1"); a.set_ylabel(f"{name} 2")
        plt.tight_layout(); plt.savefig(f"{OUT}/{fname}"); plt.close()
    scatter_pair(P, "PCA", "04_pca_clusters.png")
    scatter_pair(T, "t-SNE", "05_tsne_clusters.png")

    # --- Personas
    prof = rfm.groupby("KMeans")[["Recency", "Frequency", "Monetary"]].mean()
    prof["Customers"] = rfm.groupby("KMeans").size()
    prof["Share_%"] = (prof.Customers / prof.Customers.sum() * 100).round(1)
    prof["Revenue_share_%"] = (rfm.groupby("KMeans").Monetary.sum() / rfm.Monetary.sum() * 100).round(1)
    names = name_personas(prof)
    prof["Persona"] = prof.index.map(names)
    prof["Action"] = prof.Persona.map(action_for)
    print(prof.round(1).drop(columns="Action"), "\n")
    rfm["Persona"] = rfm.KMeans.map(names)

    # heatmap of relative cluster means
    rel = prof[["Recency", "Frequency", "Monetary"]] / rfm[["Recency", "Frequency", "Monetary"]].mean()
    fig, ax = plt.subplots(figsize=(6, 0.9 * len(rel) + 1.5))
    im = ax.imshow(rel.values, cmap="RdYlGn_r" if False else "YlGnBu", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(rel.columns)
    ax.set_yticks(range(len(rel))); ax.set_yticklabels([f"{i}: {names[i]}" for i in rel.index])
    for i in range(rel.shape[0]):
        for j in range(3):
            ax.text(j, i, f"{rel.values[i, j]:.2f}x", ha="center", va="center", fontsize=9)
    ax.set_title("Cluster mean vs overall mean (RFM)"); plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(f"{OUT}/06_persona_heatmap.png"); plt.close()

    # persona-labelled t-SNE
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in sorted(names):
        m = rfm.KMeans.values == c
        ax.scatter(T[m, 0], T[m, 1], s=8, alpha=.75, label=names[c])
    ax.legend(markerscale=2); ax.set_title("t-SNE view of customer personas")
    plt.tight_layout(); plt.savefig(f"{OUT}/07_tsne_personas.png"); plt.close()

    # --- Save artefacts
    rfm.to_csv(f"{OUT}/rfm_with_segments.csv")
    prof.to_csv(f"{OUT}/persona_summary.csv")
    ks.to_csv(f"{OUT}/k_selection_metrics.csv", index=False)
    summary = {
        "data_source": src, "customers": int(len(rfm)), "best_k": best_k,
        "kmeans_silhouette": round(float(silhouette_score(X, km.labels_)), 3),
        "pca_var_2comp": round(float(evr[:2].sum()), 3),
        "dbscan": None if not best else {"eps": eps, "min_samples": ms, "clusters": ncl,
                                         "noise_pct": round(noise * 100, 1), "silhouette": round(s, 3)},
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
