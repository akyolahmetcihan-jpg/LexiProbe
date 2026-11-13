import html
import math
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

st.set_page_config(page_title="LexiProbe — CEFR & Text Complexity", page_icon="λ", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1020px !important; }

/* Minimal Top Bar */
.lp-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; background: #ffffff;
    border: 1px solid #e2e8f0; border-radius: 12px;
    margin-bottom: 40px;
}
.lp-brand {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 1.18rem; color: #0f172a; letter-spacing: -0.02em;
}
.lp-glyph {
    font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.92rem;
    color: #0f766e; background: #ccfbf1; border: 1px solid #99f6e4;
    padding: 2px 8px; border-radius: 7px;
}
.lp-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;
    padding: 4px 10px; border-radius: 6px; color: #475569; background: #f1f5f9;
}

/* Clean Hero */
.lp-hero { text-align: center; margin: 0 auto 32px; max-width: 680px; }
.lp-hero h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.85rem !important; font-weight: 700 !important;
    line-height: 1.12 !important; letter-spacing: -0.035em !important;
    color: #0f172a; margin-bottom: 10px;
}
.lp-accent { color: #0d9488; }
.lp-sub { font-size: 1rem; color: #64748b; font-weight: 400; }

/* Console Shell */
.lp-shell {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 18px 20px; margin: 18px 0 24px;
}
.lp-shell-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px; font-size: 0.82rem; color: #64748b;
}
.lp-shell-head b { color: #0f172a; font-size: 0.88rem; }

.lp-dark-window {
    background: #0b1320; color: #e2e8f0; border-radius: 10px;
    padding: 20px 22px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.94rem; line-height: 1.85;
}
.tok-pos {
    background: rgba(16, 185, 129, 0.25); color: #a7f3d0;
    padding: 2px 6px; border-radius: 4px; border-bottom: 2px solid #10b981;
}
.tok-neg {
    background: rgba(244, 63, 94, 0.22); color: #fecdd3;
    padding: 2px 6px; border-radius: 4px; border-bottom: 2px solid #f43f5e;
}

/* KPI Cards */
.lp-kpi {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 14px 18px;
}
.lp-kpi-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
.lp-kpi-val { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: #0f172a; margin-top: 3px; }
.lp-kpi-val.accent { color: #0f766e; }

.note-pill {
    background: #f8fafc; border-left: 3px solid #0d9488;
    padding: 10px 14px; border-radius: 6px; margin-bottom: 14px; font-size: 0.88rem; color: #334155;
}

table.clean { width: 100%; border-collapse: collapse; font-size: .86rem; background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }
table.clean th { text-align: left; background: #f8fafc; padding: 9px 12px; border-bottom: 1px solid #e2e8f0; font-weight: 600; color: #334155; }
table.clean td { padding: 8px 12px; border-bottom: 1px solid #f1f5f9; font-family: 'JetBrains Mono', monospace; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

LEVELS = ["A1–A2 (Basic)", "B1–B2 (Independent)", "C1–C2 (Proficient)"]
SHORT_LEVELS = ["A1–A2", "B1–B2", "C1–C2"]
ACADEMIC_SUBLIST = {
    "analyze", "analysis", "approach", "assessment", "assume", "authority", "available", "benefit",
    "concept", "consistent", "constitutional", "context", "contract", "create", "data", "definition",
    "derived", "distribution", "economic", "environment", "established", "estimate", "evidence",
    "export", "factors", "financial", "formula", "function", "identified", "income", "indicate",
    "individual", "interpretation", "involved", "issues", "labor", "legal", "legislation", "major",
    "method", "methodology", "occur", "percent", "period", "policy", "principle", "procedure",
    "process", "required", "research", "response", "role", "section", "sector", "significant",
    "similar", "source", "specific", "structure", "theory", "variables", "empirical", "hypothesis",
    "paradigm", "phenomenon", "syntactic", "morphological", "cognitive", "correlations", "consequently",
    "furthermore", "nevertheless", "notwithstanding", "subsequently", "epistemological", "jurisdiction"
}
SUBORDINATORS = {"although", "whereas", "whereby", "inasmuch", "nonetheless", "provided", "unless", "because", "while", "since"}


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    vowels = re.findall(r"[aeiouy]{1,2}", w)
    return max(1, len(vowels))


def extract_linguistic_metrics(text: str) -> Dict[str, float]:
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    n_sents = max(1, len(sents))
    n_words = max(1, len(words))
    lower_words = [w.lower() for w in words]

    avg_sent_len = n_words / n_sents
    avg_word_len = sum(len(w) for w in lower_words) / n_words
    syllables = sum(count_syllables(w) for w in lower_words)
    syll_per_word = syllables / n_words
    long_word_ratio = sum(1 for w in lower_words if len(w) >= 7) / n_words
    ttr = len(set(lower_words)) / math.sqrt(2 * n_words)
    awl_ratio = sum(1 for w in lower_words if w in ACADEMIC_SUBLIST) / n_words
    subord_ratio = sum(1 for w in lower_words if w in SUBORDINATORS) / n_sents
    fk_grade = 0.39 * avg_sent_len + 11.8 * syll_per_word - 15.59

    return {
        "Sentence Length (words)": round(avg_sent_len, 2),
        "Word Length (chars)": round(avg_word_len, 2),
        "Syllables per Word": round(syll_per_word, 2),
        "Polysyllabic Ratio (≥7 chars)": round(long_word_ratio, 3),
        "Type-Token Ratio (CTTR)": round(ttr, 2),
        "Academic Word List (AWL)": round(awl_ratio, 3),
        "Subordination Index": round(subord_ratio, 2),
        "Flesch-Kincaid Grade": round(max(0.0, fk_grade), 2),
    }


class LinguisticFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([list(extract_linguistic_metrics(t).values()) for t in X], dtype=float)


CURATED_CORPUS: List[Tuple[str, int]] = [
    ("I wake up at seven o'clock every morning and drink a cup of hot tea.", 0),
    ("My brother lives in a small house near the train station.", 0),
    ("She likes listening to music and watching movies with her friends on weekends.", 0),
    ("We went to the supermarket yesterday because we needed bread, milk, and fresh apples.", 0),
    ("Can you tell me how to go to the nearest bus stop from here?", 0),
    ("The weather is very cold today, so please wear your warm jacket.", 0),
    ("He works in a restaurant as a cook and finishes his job at ten.", 0),
    ("My favorite color is blue and I have two small cats at home.", 0),
    ("They are playing football in the park because the sun is shining.", 0),
    ("I want to buy a new ticket for the afternoon train to Milan.", 0),
    ("She reads a short book every night before she goes to sleep.", 0),
    ("Our teacher is very kind and helps us with easy homework every day.", 0),
    ("There is a big table and four chairs in the kitchen.", 0),
    ("Do you like coffee or do you prefer cold water with lemon?", 0),
    ("Last summer my family visited Rome and ate delicious pizza every evening.", 0),
    ("He cannot come to the party tonight because he is feeling tired.", 0),
    ("I always walk to school with my best friend in the morning.", 0),
    ("The shop opens at nine and closes at six on Saturday.", 0),
    ("She bought a nice red dress for her sister's birthday party.", 0),
    ("Please open the window because this room is very hot.", 0),
    ("We saw many animals at the zoo and took funny photos.", 0),
    ("My father drives a white car to his office every Monday.", 0),
    ("I am learning Italian because I want to travel to Italy next year.", 0),
    ("The children are happy when they eat ice cream in the garden.", 0),
    ("She put her keys and her phone inside the small bag.", 0),
    ("It rained all day yesterday so we stayed home and watched television.", 0),
    ("He has a dog that likes running after balls in the yard.", 0),
    ("Where did you put the blue pen that was on my desk?", 0),
    ("They eat breakfast together at eight before going to work.", 0),
    ("My grandmother makes wonderful soup when we visit her on Sunday.", 0),
    ("I read a short article about economic policy and data yesterday.", 0),
    ("Although working from home offers considerable flexibility, many employees report feeling isolated from their colleagues.", 1),
    ("The local government announced a new public transport plan to reduce traffic congestion in the city center.", 1),
    ("Students who participate in exchange programs usually develop stronger communication skills and cultural awareness.", 1),
    ("If companies invest more in renewable energy, they can significantly lower their long-term operational costs.", 1),
    ("Researchers interviewed over two hundred participants to understand how social media affects sleep quality.", 1),
    ("While online shopping is convenient, traditional stores allow customers to examine products directly before buying.", 1),
    ("The documentary explains how climate change is altering migration patterns among coastal bird species.", 1),
    ("Applicants must submit a motivation letter and two academic references before the registration deadline.", 1),
    ("Despite initial technical difficulties, the software update improved system stability and user satisfaction.", 1),
    ("Learning a second language requires consistent practice, exposure to authentic materials, and patience with mistakes.", 1),
    ("Many universities now offer hybrid courses that combine recorded lectures with interactive seminar discussions.", 1),
    ("The museum exhibition highlights the economic and cultural exchange between Mediterranean port cities.", 1),
    ("Consumers are increasingly choosing sustainable packaging when they compare similar products in supermarkets.", 1),
    ("Before signing the rental agreement, tenants should carefully review the clauses regarding maintenance fees.", 1),
    ("Regular physical exercise not only strengthens the cardiovascular system but also reduces workplace stress.", 1),
    ("The committee postponed the final decision because several members requested additional financial data.", 1),
    ("Tourism brings substantial revenue to historic towns, yet it often increases housing prices for local residents.", 1),
    ("Engineers designed a lightweight battery that extends the driving range of electric vehicles during winter.", 1),
    ("During the internship, she gained practical experience in database management and customer support.", 1),
    ("Journalists must verify information from multiple independent sources before publishing sensitive reports.", 1),
    ("Because public funding was reduced, the research institute had to seek partnerships with private industry.", 1),
    ("The novel explores the conflict between personal ambition and family loyalty in a rapidly changing society.", 1),
    ("When flights are delayed for more than three hours, passengers are entitled to compensation under European rules.", 1),
    ("Doctors recommend limiting screen time in the evening to prevent disruptions to the natural sleep cycle.", 1),
    ("The survey indicates that younger voters prioritize environmental policy and affordable housing.", 1),
    ("By analyzing customer feedback, the product team identified the most frequent usability issues.", 1),
    ("Although the restaurant is slightly expensive, its seasonal menu attracts food enthusiasts from across the region.", 1),
    ("Effective project management depends on clear deadlines, transparent communication, and realistic resource allocation.", 1),
    ("The conference brought together specialists from different fields to discuss ethical guidelines for artificial intelligence.", 1),
    ("Upgrading the railway network will shorten travel times between regional industrial hubs.", 1),
    ("This method is very good for our research project.", 1),
    ("Empirical investigations into second language acquisition demonstrate that morphosyntactic transfer is constrained by typological distance.", 2),
    ("Notwithstanding the proliferation of neural architectures, interpretable symbolic models remain indispensable for diagnostic linguistic annotation.", 2),
    ("The epistemological assumptions underlying distributional semantics warrant critical scrutiny when applied to low-resource diachronic corpora.", 2),
    ("Consequently, regulatory frameworks governing algorithmic decision-making must reconcile institutional transparency with proprietary confidentiality.", 2),
    ("Whereas traditional philological methodologies prioritized canonical manuscripts, computational stylometry enables large-scale authorship attribution.", 2),
    ("Socioeconomic stratification exacerbates disparities in educational attainment, thereby perpetuating intergenerational occupational immobility.", 2),
    ("The treatise elucidates how monetary tightening reverberates through emerging sovereign debt markets under conditions of asymmetric information.", 2),
    ("By leveraging cross-lingual word embeddings, the study quantifies semantic drift across divergent ideological discourses.", 2),
    ("Phenomenological accounts of embodied cognition challenge the Cartesian dichotomy between perceptual representation and motor intentionality.", 2),
    ("Although linear classifiers exhibit lower representational capacity than deep transformers, their feature weights afford direct explanatory traceability.", 2),
    ("Jurisprudential debates surrounding extraterritorial jurisdiction have intensified amidst transnational digital commerce.", 2),
    ("Morphological agglutination in Turkic languages introduces severe data sparsity for unsegmented subword tokenization schemes.", 2),
    ("The macroeconomic equilibrium postulated by neoclassical theory presupposes frictionless capital mobility and rational expectations.", 2),
    ("Syntactic complexity metrics derived from dependency treebanks correlate robustly with holistic assessments of academic writing proficiency.", 2),
    ("Furthermore, diachronic corpus analysis reveals systematic grammaticalization pathways whereby lexical verbs evolve into aspectual auxiliaries.", 2),
    ("Biomedical ontologies facilitate automated hypothesis generation by formalizing hierarchical relations among heterogeneous clinical entities.", 2),
    ("To mitigate confounding variables, the longitudinal protocol employed stratified random sampling across demographically diverse cohorts.", 2),
    ("The aesthetic paradigm of modernist architecture repudiated ornamental historicism in favor of functionalist structural austerity.", 2),
    ("Psycholinguistic eye-tracking experiments indicate that garden-path sentences elicit immediate regressive saccades upon syntactic disambiguation.", 2),
    ("Without rigorous inter-annotator agreement metrics such as Cohen's kappa, supervised training datasets remain susceptible to systematic bias.", 2),
    ("Institutional isomorphism compels organizations within mature fields to adopt homogeneous administrative structures regardless of efficiency.", 2),
    ("The geopolitical ramifications of decarbonization policies necessitate multilateral coordination on carbon border adjustment mechanisms.", 2),
    ("Lexical-functional grammar separates constituent structure from functional grammatical relations, accommodating non-configurational syntax.", 2),
    ("Epigenetic modifications mediated by environmental stressors can induce heritable alterations in transcriptional gene expression.", 2),
    ("Insofar as pragmatic inference relies on shared contextual presuppositions, purely truth-conditional semantics proves inadequate.", 2),
    ("The monetization of attention within algorithmic media ecosystems incentivizes affective polarization over deliberative public discourse.", 2),
    ("Quantitative evaluation on held-out test partitions prevents optimistic overfitting and substantiates generalizability across domains.", 2),
    ("Subcategorization frames restrict the syntactic realization of predicate arguments within formal generative grammars.", 2),
    ("Historiographical revisions of post-industrial urbanization emphasize the interplay between municipal zoning and spatial segregation.", 2),
    ("Counterfactual probing of transformer attention heads isolates the circuit responsible for long-distance subject-verb agreement.", 2),
]


@st.cache_data(show_spinner=False)
def load_corpus() -> Tuple[List[str], List[int], str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("UniversalCEFR/cefr_sp_en", split="train[:600]")
        texts, labels = [], []
        map_lvl = {"A1": 0, "A2": 0, "B1": 1, "B2": 1, "C1": 2, "C2": 2}
        for row in ds:
            txt = row.get("text") or row.get("sentence") or ""
            lvl = str(row.get("cefr_level") or row.get("level") or "")
            if len(txt.split()) >= 5 and lvl in map_lvl:
                texts.append(txt)
                labels.append(map_lvl[lvl])
        if len(set(labels)) == 3 and len(texts) >= 120:
            return texts, labels, f"UniversalCEFR · n={len(texts)}"
    except Exception:
        pass
    texts = [t for t, _ in CURATED_CORPUS]
    labels = [y for _, y in CURATED_CORPUS]
    return texts, labels, f"CEFR Corpus · n={len(texts)}"


def diagnose_misclassification(text: str, true_idx: int, pred_idx: int) -> str:
    m = extract_linguistic_metrics(text)
    if pred_idx > true_idx:
        if m["Academic Word List (AWL)"] > 0.08:
            return "Academic vocabulary (AWL) inside a short, simple sentence."
        return "Long words inflated the score despite simple syntax."
    if m["Sentence Length (words)"] < 12:
        return "Short sentence length lowered the score despite advanced terms."
    return "No subordinating conjunctions; scored lower on syntax."


@st.cache_resource(show_spinner=False)
def train_models():
    texts, labels, source_name = load_corpus()
    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)

    def make_union():
        return FeatureUnion([
            ("tfidf_word", TfidfVectorizer(ngram_range=(1, 2), max_features=2500, sublinear_tf=True)),
            ("tfidf_char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=1500)),
            ("ling", Pipeline([("ext", LinguisticFeatureTransformer()), ("sc", StandardScaler())])),
        ])

    candidates = {
        "Logistic Regression": Pipeline([
            ("feats", make_union()),
            ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
        ]),
        "Linear SVM": Pipeline([
            ("feats", make_union()),
            ("clf", LinearSVC(C=1.0, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("feats", make_union()),
            ("clf", RandomForestClassifier(n_estimators=150, random_state=42)),
        ]),
    }

    benchmarks, trained, misclassified = [], {}, {}
    for name, pipe in candidates.items():
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        train_ms = (time.perf_counter() - t0) * 1000
        preds = pipe.predict(X_te)
        acc = accuracy_score(y_te, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="macro", zero_division=0)
        cm = confusion_matrix(y_te, preds, labels=[0, 1, 2])
        benchmarks.append({
            "Model": name, "Accuracy": round(acc, 2), "Precision": round(p, 2),
            "Recall": round(r, 2), "Macro F1": round(f1, 2),
            "Train (ms)": round(train_ms, 1), "cm": cm,
        })
        trained[name] = pipe
        errs = []
        for txt, yt, yp in zip(X_te, y_te, preds):
            if int(yt) != int(yp):
                errs.append({
                    "Sentence": txt,
                    "True": SHORT_LEVELS[int(yt)],
                    "Predicted": SHORT_LEVELS[int(yp)],
                    "Reason": diagnose_misclassification(txt, int(yt), int(yp)),
                })
        misclassified[name] = errs

    explainer_vec = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
    X_tr_w = explainer_vec.fit_transform(X_tr)
    explainer_clf = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
    explainer_clf.fit(X_tr_w, y_tr)

    return trained, benchmarks, misclassified, (explainer_vec, explainer_clf), source_name, len(X_te)


def explain_tokens(text: str, target_class: int, explainer) -> str:
    vec, clf = explainer
    vocab = vec.vocabulary_
    coefs = clf.coef_[target_class]
    tokens = re.findall(r"\w+|\W+", text)
    out = []
    for tok in tokens:
        low = tok.lower()
        if low in vocab:
            w = coefs[vocab[low]]
            if w > 0.16:
                out.append(f'<span class="tok-pos" title="Weight: +{w:.2f}">{html.escape(tok)}</span>')
                continue
            elif w < -0.16:
                out.append(f'<span class="tok-neg" title="Weight: {w:.2f}">{html.escape(tok)}</span>')
                continue
        out.append(html.escape(tok))
    return "".join(out)


def quick_note(metrics: Dict[str, float]) -> str:
    if metrics["Sentence Length (words)"] <= 11 and metrics["Academic Word List (AWL)"] >= 0.10:
        return "Note: Short sentence, but high Academic Word List (AWL) density."
    if metrics["Sentence Length (words)"] >= 18 and metrics["Academic Word List (AWL)"] == 0:
        return "Note: Long sentence structure with everyday vocabulary."
    return ""


trained_models, benchmarks, misclassified, explainer, corpus_source, n_te = train_models()

# ---------- Minimal Header & Hero ----------
st.markdown(f"""
<div class="lp-nav">
  <div class="lp-brand"><span class="lp-glyph">⟪λ⟫</span> LexiProbe</div>
  <div style="display:flex;align-items:center;gap:8px;">
    <span class="lp-tag">{corpus_source}</span>
    <span class="lp-tag">TF-IDF + Linguistic Features</span>
  </div>
</div>

<div class="lp-hero">
  <h1>CEFR & text complexity<span class="lp-accent">.</span></h1>
  <div class="lp-sub">Statistical proficiency classification using N-gram TF-IDF and readability metrics.</div>
</div>
""", unsafe_allow_html=True)

PRESETS = {
    "C1–C2 · Academic": (
        "Notwithstanding the proliferation of neural architectures, interpretable symbolic models remain "
        "indispensable for diagnostic linguistic annotation and cross-lingual morphosyntactic analysis."
    ),
    "B1–B2 · Intermediate": (
        "Many universities now offer hybrid courses that combine recorded online lectures with interactive "
        "seminar discussions, allowing students to manage their schedules more flexibly."
    ),
    "A1–A2 · Basic": (
        "Yesterday I went to the supermarket with my brother because we wanted to buy fresh bread, milk, "
        "and some apples for breakfast."
    ),
    "Boundary Case · Short Academic": (
        "Empirical data and methodology indicate a significant economic correlation."
    ),
}

col_p, col_m = st.columns([2, 1])
with col_p:
    preset_key = st.selectbox("Sample", list(PRESETS))
with col_m:
    model_name = st.selectbox("Model", list(trained_models.keys()))

user_text = st.text_area("Text", value=PRESETS[preset_key], height=90, label_visibility="collapsed")

if user_text.strip():
    pipe = trained_models[model_name]
    t0 = time.perf_counter()
    pred_idx = int(pipe.predict([user_text])[0])
    infer_ms = (time.perf_counter() - t0) * 1000
    pred_label = LEVELS[pred_idx]

    clf_step = pipe.named_steps["clf"]
    feats_vec = pipe.named_steps["feats"].transform([user_text])
    if hasattr(clf_step, "predict_proba"):
        probs = clf_step.predict_proba(feats_vec)[0]
    else:
        scores = clf_step.decision_function(feats_vec)[0]
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / exp_s.sum()

    metrics = extract_linguistic_metrics(user_text)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="lp-kpi"><div class="lp-kpi-label">CEFR Level</div><div class="lp-kpi-val accent">{SHORT_LEVELS[pred_idx]}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="lp-kpi"><div class="lp-kpi-label">Confidence</div><div class="lp-kpi-val">{probs[pred_idx]*100:.0f}%</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="lp-kpi"><div class="lp-kpi-label">Flesch-Kincaid</div><div class="lp-kpi-val">{metrics["Flesch-Kincaid Grade"]}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="lp-kpi"><div class="lp-kpi-label">Latency</div><div class="lp-kpi-val">{infer_ms:.1f} ms</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="lp-shell">
      <div class="lp-shell-head">
        <b>Token Weights ({SHORT_LEVELS[pred_idx]})</b>
        <span>Green = supports {SHORT_LEVELS[pred_idx]} · Red = pulls to other levels</span>
      </div>
      <div class="lp-dark-window">
        {explain_tokens(user_text, pred_idx, explainer)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    note = quick_note(metrics)
    if note:
        st.markdown(f'<div class="note-pill">{html.escape(note)}</div>', unsafe_allow_html=True)

    tab_ling, tab_eval, tab_err, tab_prob = st.tabs([
        "Linguistic Features",
        "Model Benchmarks",
        "Error Analysis",
        "Probabilities",
    ])

    with tab_ling:
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in metrics.items())
        st.markdown(f'<table class="clean"><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

    with tab_eval:
        cols = ["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Train (ms)"]
        head = "".join(f"<th>{c}</th>" for c in cols)
        b_rows = "".join("<tr>" + "".join(f"<td>{b[c]}</td>" for c in cols) + "</tr>" for b in benchmarks)
        st.markdown(f'<table class="clean"><thead><tr>{head}</tr></thead><tbody>{b_rows}</tbody></table>', unsafe_allow_html=True)
        st.write("")
        cm = next(b["cm"] for b in benchmarks if b["Model"] == model_name)
        cm_head = f"<th>Confusion Matrix ({model_name})</th>" + "".join(f"<th>{l}</th>" for l in SHORT_LEVELS)
        cm_rows = "".join(f"<tr><td><b>{SHORT_LEVELS[i]}</b></td>" + "".join(f"<td>{cm[i, j]}</td>" for j in range(3)) + "</tr>" for i in range(3))
        st.markdown(f'<table class="clean"><thead><tr>{cm_head}</tr></thead><tbody>{cm_rows}</tbody></table>', unsafe_allow_html=True)

    with tab_err:
        errs = misclassified[model_name]
        if not errs:
            st.info("No misclassified sentences on the test split for this model.")
        else:
            e_cols = ["Sentence", "True", "Predicted", "Reason"]
            e_head = "".join(f"<th>{c}</th>" for c in e_cols)
            e_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in e_cols) + "</tr>" for r in errs)
            st.markdown(f'<table class="clean"><thead><tr>{e_head}</tr></thead><tbody>{e_rows}</tbody></table>', unsafe_allow_html=True)

    with tab_prob:
        for lvl, p in zip(LEVELS, probs):
            st.progress(float(p), text=f"{lvl} — {p*100:.1f}%")