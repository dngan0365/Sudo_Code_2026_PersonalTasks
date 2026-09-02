from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC


@dataclass(slots=True)
class ModelFactory:
    """Build sklearn classifiers used in the notebook."""

    random_state: int = 42
    class_weight_balanced: bool = False

    SUPPORTED_MODELS = ("logistic", "svm-linear", "svm-rbf", "svm-poly", "multinomial-nb", "bernoulli-nb")

    def create(self, name: str):
        class_weight = "balanced" if self.class_weight_balanced else None
        if name == "logistic":
            return LogisticRegression(max_iter=1000, random_state=self.random_state, class_weight=class_weight)
        if name == "svm-linear":
            return SVC(kernel="linear", random_state=self.random_state, class_weight=class_weight)
        if name == "svm-rbf":
            return SVC(kernel="rbf", random_state=self.random_state, class_weight=class_weight)
        if name == "svm-poly":
            return SVC(kernel="poly", degree=2, random_state=self.random_state, class_weight=class_weight)
        if name == "multinomial-nb":
            return MultinomialNB()
        if name == "bernoulli-nb":
            return BernoulliNB()
        raise ValueError(f"Unsupported model: {name}")
