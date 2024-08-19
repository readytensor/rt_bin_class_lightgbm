import os
import warnings
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Classifier:
    """A wrapper class for the binary classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "LightGBM_Binary_Classifier"

    def __init__(
        self,
        boosting_type: Optional[str] = "gbdt",
        n_estimators: Optional[int] = 250,
        num_leaves: Optional[int] = 31,
        learning_rate: Optional[float] = 1e-1,
        decision_threshold: Optional[float] = 0.5,
        positive_class_weight: Optional[float] = 1.0,
        **kwargs,
    ):
        """Construct a new binary classifier.

        Args:
            boosting_type (str, optional): boosting_type
                categorical options: gbdt, dart, goss, rf.
                Defaults to 'gbdt'.
            n_estimators (int, optional): The number of trees in the forest.
                Defaults to 100.
            min_samples_split (int, optional): Maximum tree leaves for base learners.
                to split an internal node.
                Defaults to 31.
            learning_rate (float, optional): Learning rate.
                to be at a leaf node.
                Defaults to 1e-1.
            decision_threshold (float, optional): The decision threshold for
                the positive class. Defaults to 0.5.
            positive_class_weight (float, optional): The weight of the positive
                class. Defaults to 1.0.
        """
        self.boosting_type = boosting_type
        self.num_leaves = int(num_leaves)
        self.learning_rate = learning_rate
        self.n_estimators = int(n_estimators)
        self.decision_threshold = decision_threshold
        self.positive_class_weight = positive_class_weight
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> lgb.LGBMClassifier:
        """Build a new binary classifier."""
        model = lgb.LGBMClassifier(
            num_class=1,
            boosting_type=self.boosting_type,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=42,
            class_weight={0: 1, 1: self.positive_class_weight},
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the binary classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(
        self,
        inputs: pd.DataFrame,
        decision_threshold: float = -1,
    ) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
            decision_threshold (Optional float): Decision threshold for the
                positive class.
                Value -1 indicates use the default set when model was
                instantiated.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if decision_threshold == -1:
            decision_threshold = self.decision_threshold
        if self.model is not None:
            prob = self.predict_proba(inputs)
            labels = prob[:, 1] >= decision_threshold
        return labels

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(
        self,
        test_inputs: pd.DataFrame,
        test_targets: pd.Series,
        decision_threshold: float = -1,
    ) -> float:
        """Evaluate the classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
            decision_threshold (Optional float): Decision threshold for the
                positive class.
                Value -1 indicates use the default set when model was
                instantiated.
        Returns:
            float: The accuracy of the classifier.
        """
        if decision_threshold == -1:
            decision_threshold = self.decision_threshold
        if self.model is not None:
            prob = self.predict_proba(test_inputs)
            labels = prob[:, 1] >= decision_threshold
            score = f1_score(test_targets, labels)
            return score

        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"boosting_type: {self.boosting_type}, "
            f"learning_rate: {self.learning_rate}, "
            f"n_estimators: {self.n_estimators}, "
            f"num_leaves: {self.num_leaves})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    decision_threshold: float = -1,
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.
        decision_threshold (Union(optional, float)): Decision threshold
                for predicted label.
                Value -1 indicates use the default set when model was
                instantiated.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test, decision_threshold)


def set_decision_threshold(model: Classifier, decision_threshold: float) -> None:
    """
    Set the decision threshold for the classifier model.

    Args:
        model (Classifier): The classifier model.
        decision_threshold (float): The decision threshold.
    """
    model.decision_threshold = decision_threshold
