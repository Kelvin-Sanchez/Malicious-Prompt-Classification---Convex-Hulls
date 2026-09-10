# Malicious Prompt Classification with Convex Hulls

Research project testing whether convex hulls can classify malicious LLM prompt embeddings, done as part of my M.S. in Cybersecurity (AI concentration) at Johns Hopkins. Full writeup: [`Classification_with_CH.pdf`](Classification_with_CH.pdf).

## Problem

LLM adoption has made prompt-based attacks (jailbreaks, adversarial perturbations) a growing threat. Traditional classifiers (Random Forest, SVM, kNN) are effective but can be circumvented by adversarial examples. Prior work classifies with *approximate* convex hulls to manage the curse of dimensionality. This project instead computes the *exact* convex hull after reducing embeddings to 3D with PCA, to see if a geometric decision boundary can hold up as a defense.

## Paper vs. notebook

This repo contains both the original paper and a follow-up notebook (`MaliciousClassification_CE.ipynb`) that diverges from it in several ways, not just the embedding model. They're best read as two related but distinct experiments.

### Paper (`Classification_with_CH.pdf`)

- Datasets: MPDD, BeaverTails, and Do-Not-Answer, individually and combined.
- Embeddings: OpenAI's `text-embedding-3-large`.
- Convex hull method: one convex hull per class, computed on the PCA-reduced (3D) training points, plus variants that strip 1, 2, and 3 standard deviations of outliers before computing the hull (Outlier RM 1/2/3).
- Benchmarked against Logistic Regression, kNN (k=5), Random Forest, and SVM (linear and RBF kernels), with 5-fold cross-validation.
- Results (from the paper's tables):
  - On MPDD, the baseline convex hull model hit 0.99 precision but only 0.71 recall (F1 0.83) — it rarely misclassified benign prompts as malicious, but missed a meaningful share of actual malicious ones.
  - Performance dropped sharply on BeaverTails and the combined dataset, where benign/malicious embeddings overlap heavily once reduced to 3D (accuracy fell to ~0.45–0.48 for some hull variants).
  - No convex hull variant exceeded the traditional models — Random Forest and kNN both landed around 0.88–0.89 accuracy on MPDD, ahead of every hull variant.
  - Outlier removal traded precision for recall without a clear net win: aggressive removal (Outlier RM 1) boosted recall at the cost of precision and overall accuracy.
  - **Takeaway:** convex hulls didn't outperform standard classifiers, but precision stayed high on MPDD specifically, where the two classes were more geometrically separable. The paper argues that under those conditions, convex hulls could still work as a defense guardrail — not a general-purpose replacement for traditional classifiers.

### Notebook (`MaliciousClassification_CE.ipynb`)

- Dataset: MPDD only — the notebook doesn't load or process BeaverTails or Do-Not-Answer.
- Embeddings: the open-source `BAAI/bge-large-en-v1.5` model via `sentence-transformers`, instead of OpenAI's API — both to see if it changes classification performance and so anyone can rerun the experiment without paying for API access.
- Convex hull method: different from the paper. Instead of one hull per class with statistical outlier trimming, the notebook first clusters each class with DBSCAN, then computes a convex hull (and Delaunay triangulation) per cluster:
  - `ClusterCHClassifier` builds hulls only for benign clusters; anything falling outside every benign hull is classified malicious.
  - `NearestClusterCHClassifier` builds hulls for both classes; if a point falls outside every hull, it's assigned to the class of its nearest hull vertex.
- Dimensionality reduction: both PCA and UMAP (3 components each) are used and compared.
- Benchmarked against Logistic Regression, Random Forest, and SVM — no kNN in this version.
- No 5-fold cross-validation — this is a single train/test split.
- The notebook has no saved cell outputs in this repo (it hasn't been executed and committed with results), so there are no notebook-specific metrics to report yet. Treat it as a work-in-progress variant of the paper's method, not a reproduction of the paper's numbers above.

## Stack

Python, scikit-learn (`DBSCAN`, `RandomForestClassifier`, `LogisticRegression`, `SVC`), SciPy (`ConvexHull`, `Delaunay`), sentence-transformers (`BAAI/bge-large-en-v1.5`), UMAP, Plotly (3D visualization), pandas.

## Future Work (from the paper)

- Combine convex hulls with traditional models: classify the non-overlapping regions geometrically, fall back to a traditional model in the overlap zone.
- Try approximate convex hull methods (reflective hulls, hull-to-hull distance) shown effective in prior work.
- Explore alpha shapes for a tighter geometric envelope than a convex hull provides.
- Measure training/prediction speed against traditional models to see if convex hulls offer a performance advantage even without a classification-accuracy edge.
