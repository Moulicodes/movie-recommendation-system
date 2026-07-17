# Hybrid Content-Based Movie Recommender Engine

An advanced content-based movie recommendation system that utilizes a dual-pipeline natural language processing (NLP) architecture to generate mathematically optimal movie recommendations. By pairing exact metadata profiling with soft semantic plot matching, the engine successfully solves the limitations of classic single-vectorizer recommendation approaches.

## 📊 Dataset Reference
This engine is built and evaluated using the **TMDB 5000 Movie Dataset**, which features rich metadata profiles for approximately 5,000 films.
* **Dataset Link:** https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---

## 🏗️ System Architecture & Pipeline

Unlike standard recommenders that pass all text features through a single vectorizer, this model implements a **parallel, asymmetric dual-vectorizer branch** to handle different classes of data correctly:

### 1. The CountVectorizer Branch (Structured Metadata)
* **Features:** `genres`, `keywords`, `cast`, `crew`
* **Preprocessing Pipeline:** Categorical tags are stripped of standard alphanumeric casing, converted to lowercase, and subjected to strict whitespace compression (e.g., `"Science Fiction"` -> `"sciencefiction"`, `"Johnny Depp"` -> `"johnnydepp"`). This guarantees that multi-word tokens are treated as a single structural identifier rather than separate words. Stemming is explicitly bypassed here to protect the unique names of actors/directors and avoid conceptual collisions (such as "community" and "commune" mistakenly collapsing into the same root).
* **Mathematical Justification:** Categorical tags demand exact, binary matching. Using term-frequency down-weighting (TF-IDF) here introduces a negative dataset bias. For instance, if a rare keyword or a specific actor uniquely connects two franchise blockbusters, TF-IDF would incorrectly penalize these crucial recurring elements simply because those terms appear across hundreds of other entries globally. `CountVectorizer` ensures every single tag match contributes a clean, unpenalized $+1$ to the coordinate space.

### 2. The TF-IDF Branch (Unstructured Prose)
* **Features:** `overview`, `tagline`
* **Preprocessing Pipeline:** Free-form text summaries are standard-normalized, stripped of generic English stop-words (`'english'`), and passed through NLTK's `PorterStemmer` algorithms. This collapses common semantic variations of identical narrative themes down to their linguistic roots (e.g., `"space battle"`, `"space battles"`, and `"space battling"` cleanly unify into `"space"` and `"battl"`).
* **Mathematical Justification:** Unstructured plot synopses suffer heavily from conversational noise and generic "hero's journey" language (words like *"man"*, *"save"*, *"world"*, *"discover"*). Under a simple count approach, two completely unrelated movies might receive a high similarity score just because their taglines share general marketing hype-words. `TfidfVectorizer` calculates the Inverse Document Frequency (IDF) globally across all 5,000 films, automatically zeroing out heavy-frequency noise words while boosting the mathematical impact of unique, story-defining thematic tokens (e.g., *"alien"*, *"samurai"*, *"spy"*).

### 3. Sparse Matrix Fusion & Scoring
Once both individual matrices are formed, they represent separate coordinate spaces for the same movie indexes. Because both matrices have matching dimensions on their row axis (exactly one row per movie), they are fused horizontally using a compressed sparse stack (`scipy.sparse.hstack`) to keep memory consumption low. 

Global similarity scores are then evaluated across the combined multi-dimensional vector space concurrently using the Cosine Similarity formula:

$$\text{Cosine Similarity}(x, y) = \frac{x \cdot y}{\Vert{}x\Vert{} \Vert{}y\Vert{}}$$

This unified scoring step ensures that structural franchise parameters (cast, directors, explicit genres) and narrative context (plot themes) influence the final geometric distance at the exact same time.

---

## 🛠️ Feature Engineering & Preprocessing Deep Dive

To preserve the absolute mathematical signal of the data, specific transformation rules were applied to each core feature:

| Feature Category | Columns Used | Applied Transform | Failure Mode Prevented |
| :--- | :--- | :--- | :--- |
| **Cast & Crew** | `cast`, `crew` | Lowercase + Space Removal | Prevents names like "John Wayne" and "John Ford" from incorrectly matching on the token "John". |
| **Categorical Meta** | `genres`, `keywords` | Lowercase + Tokenization | Prevents broad genre categories from drowning out micro-genres. |
| **Text Prose** | `overview`, `tagline` | Stopword Filtering + Porter Stemming | Prevents matching on structural grammatical filler ("the", "is", "an"). |

---
