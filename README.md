# Malicious Prompt Classification with Convex Hulls

Research project testing whether convex hulls can classify malicious LLM prompt embeddings, done as part of my M.S. in Cybersecurity (AI concentration) at Johns Hopkins. Full writeup: [`Classification_with_CH.pdf`](Classification_with_CH.pdf).

## Problem

LLM adoption has made prompt-based attacks (jailbreaks, adversarial perturbations) a growing threat. Traditional classifiers (Random Forest, SVM, kNN) are effective but can be circumvented by adversarial examples. Prior work classifies with *approximate* convex hulls to manage the curse of dimensionality. This project instead computes the *exact* convex hull after reducing embeddings to 3D with PCA, to see if a geometric decision boundary can hold up as a defense.

## Approach

- Prompts from three datasets (MPDD, BeaverTails, Do-Not-Answer) are embedded with OpenAI's `text-embedding-3-large`.
- Embeddings are reduced to 3 dimensions with PCA.
- A convex hull is computed over each class's training points; Delaunay triangulation deterministically tests whether a new point falls inside a class's hull.
- Outlier-removal variants strip 1, 2, and 3 standard deviations from the training set before computing the hull, to test whether trimming the envelope improves classification.
- Benchmarked against Logistic Regression, kNN, Random Forest, and SVM (linear and RBF) on Accuracy, Precision, Recall, F1, and ROC-AUC, with 5-fold cross-validation.

## Results

- On MPDD, the baseline convex hull model hit 0.99 precision but only 0.71 recall (F1 0.83) — it rarely misclassified benign prompts as malicious, but missed a meaningful share of actual malicious ones.
- Performance dropped sharply on BeaverTails and the combined dataset, where benign/malicious embeddings overlap heavily once reduced to 3D (accuracy fell to ~0.45–0.48 for some hull variants).
- No convex hull variant exceeded traditional models — Random Forest and kNN both landed around 0.88–0.89 accuracy on MPDD, ahead of every hull variant.
- Outlier removal traded precision for recall without a clear net win: aggressive removal (Outlier RM 1) boosted recall at the cost of precision and overall accuracy.

**Takeaway:** convex hulls didn't outperform standard classifiers here, but precision stayed high on MPDD specifically, where the two classes were more geometrically separable. Under those conditions — clearly distinct classes in latent space — the paper argues convex hulls could still work as a defense guardrail; they're not a general-purpose replacement for traditional classifiers.

## Stack

Python, scikit-learn, SciPy (`ConvexHull`, `Delaunay`), OpenAI embeddings, UMAP, Plotly (3D visualization), pandas.

## Future Work (from the paper)

- Combine convex hulls with traditional models: classify the non-overlapping regions geometrically, fall back to a traditional model in the overlap zone.
- Try approximate convex hull methods (reflective hulls, hull-to-hull distance) shown effective in prior work.
- Explore alpha shapes for a tighter geometric envelope than a convex hull provides.
- Measure training/prediction speed against traditional models to see if convex hulls offer a performance advantage even without a classification-accuracy edge.
