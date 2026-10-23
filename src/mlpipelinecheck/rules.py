"""Stable rule catalog for Mlpipelinecheck 0.1.

Texts in this module are the wording used by findings. ``docs/rules.md``
explains the same identifiers. Changing an id is a breaking change.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlpipelinecheck.findings import Category, Confidence, Severity


def rule_anchor(rule_id: str) -> str:
    """Return the documentation anchor for a rule id."""
    return rule_id.lower().replace(".", "-").replace("_", "-")


@dataclass(frozen=True)
class RuleSpec:
    """Static description of one check."""

    id: str
    title: str
    category: Category
    default_severity: Severity
    default_confidence: Confidence
    why_it_matters: str
    recommended_action: str
    limitations: str

    @property
    def docs_url(self) -> str:
        return f"docs/rules.md#{rule_anchor(self.id)}"


def _spec(
    rule_id: str,
    title: str,
    category: Category,
    severity: Severity,
    confidence: Confidence,
    why: str,
    action: str,
    limitations: str,
) -> RuleSpec:
    return RuleSpec(
        id=rule_id,
        title=title,
        category=category,
        default_severity=severity,
        default_confidence=confidence,
        why_it_matters=why,
        recommended_action=action,
        limitations=limitations,
    )


RULES: dict[str, RuleSpec] = {
    spec.id: spec
    for spec in (
        _spec(
            "DATA.EMPTY_DATASET",
            "Dataset has no rows or no feature columns",
            Category.DATA_QUALITY,
            Severity.HIGH,
            Confidence.HIGH,
            "There is nothing to train or evaluate, so any downstream metric would be meaningless.",
            "Provide a table with at least one row and at least one feature column "
            "that is not the target or a declared identifier.",
            "Identifier columns and the target are excluded from the feature set when "
            "they are declared. A dataset can be non-empty and still be unsuitable for other reasons.",
        ),
        _spec(
            "DATA.MISSING_VALUES",
            "Missing values exceed the configured threshold",
            Category.DATA_QUALITY,
            Severity.MEDIUM,
            Confidence.HIGH,
            "Missingness can change which rows support a metric and can leak information "
            "if it is handled differently at train and test time.",
            "Inspect the listed columns and rates. Decide on an explicit policy; "
            "do not assume Mlpipelinecheck dropped or imputed any values.",
            "Missing values are a data-quality signal, not proof of leakage. "
            "The denominator for each rate is the number of rows. "
            "A column exactly at the threshold is not flagged; the rate must exceed it.",
        ),
        _spec(
            "DATA.DUPLICATE_ROWS",
            "Exact duplicate rows are present",
            Category.DATA_QUALITY,
            Severity.MEDIUM,
            Confidence.HIGH,
            "Repeated records can inflate metrics and, when split naively, put the same "
            "record in more than one partition.",
            "Confirm whether repeated rows are legitimate measurements. "
            "If they are not, deduplicate before splitting or use a split that keeps copies together.",
            "This check uses the documented row fingerprint, not the pandas index. "
            "Near-duplicate rows are not detected. "
            "Legitimate repeated events can produce duplicates. "
            "This is not, by itself, proof of leakage.",
        ),
        _spec(
            "DATA.CONSTANT_FEATURE",
            "A feature has no variation",
            Category.DATA_QUALITY,
            Severity.LOW,
            Confidence.HIGH,
            "A constant feature cannot help a model generalize and may indicate a join "
            "or extraction mistake.",
            "Drop intentional constants, or confirm that a near-constant column is expected "
            "before relying on it.",
            "A single unique non-null value is a structural fact. "
            "Near-constant columns are a frequency heuristic and may be benign. "
            "All-null columns are reported as missing values, not as constants. "
            "This is not proof of leakage.",
        ),
        _spec(
            "DATA.HIGH_CARDINALITY_FEATURE",
            "A feature has a very high unique ratio",
            Category.DATA_QUALITY,
            Severity.LOW,
            Confidence.MEDIUM,
            "An undeclared identifier can let a model memorize rows instead of learning "
            "a relationship that holds for new entities.",
            "If the column is an identifier, pass it in id_columns. "
            "Otherwise decide whether the cardinality is expected for that feature.",
            "This is a heuristic, not proof that the column is an identifier or that leakage occurred. "
            "Float columns are excluded unless high_cardinality_include_floats is enabled, "
            "because continuous measurements are often unique. "
            "Declared identifier columns are not flagged.",
        ),
        _spec(
            "DATA.TARGET_IN_FEATURES",
            "The target column is included among features",
            Category.DATA_QUALITY,
            Severity.CRITICAL,
            Confidence.HIGH,
            "A model that receives the target as an input can reproduce the label without "
            "learning the intended relationship, so the evaluation does not estimate generalization.",
            "Remove the target from the feature list. "
            "By default Mlpipelinecheck already excludes a supplied target name from features; "
            "this finding means it was explicitly included or the column label is duplicated.",
            "The check looks at column identity, not at semantic aliases. "
            "A differently named copy is reported as DATA.TARGET_COPY_FEATURE when the values match exactly.",
        ),
        _spec(
            "DATA.TARGET_COPY_FEATURE",
            "A feature is an exact copy of the target",
            Category.DATA_QUALITY,
            Severity.HIGH,
            Confidence.HIGH,
            "An exact copy gives the model the label under another name, so held-out scores "
            "can look strong without generalizing.",
            "Remove the copied column from the features used to train and evaluate.",
            "Only exact comparisons under the documented rules are reported. "
            "Numeric and boolean columns are compared numerically, so 1 and 1.0 match, "
            "and True matches 1. Text is not coerced to numbers, so '1' does not match 1. "
            "Near copies and correlations are not treated as copies and are not proof of leakage.",
        ),
        _spec(
            "DATA.CLASS_IMBALANCE",
            "The classification target is imbalanced",
            Category.DATA_QUALITY,
            Severity.MEDIUM,
            Confidence.HIGH,
            "Accuracy can be high on an imbalanced target while missing the minority class. "
            "A single-class target cannot support classification metrics that need both classes.",
            "Review class counts, the split strategy, and metrics such as balanced accuracy, "
            "macro F1, and class-wise recall. Collect more minority examples if the imbalance is unintended.",
            "Imbalance is not a model failure and is not proof of leakage. "
            "Thresholds are configuration, not universal definitions of imbalance. "
            "A one-class target is reported with high severity because evaluation is structurally limited. "
            "Milder imbalance keeps the catalog default of medium severity.",
        ),
        _spec(
            "DATA.NON_FINITE_VALUES",
            "Numeric columns contain non-finite values",
            Category.DATA_QUALITY,
            Severity.MEDIUM,
            Confidence.HIGH,
            "Infinity values break many estimators and metrics, and they are easy to miss in a mean.",
            "Locate the non-finite cells outside Mlpipelinecheck and correct or remove them explicitly. "
            "Mlpipelinecheck does not impute them.",
            "Missing values are counted separately. "
            "Means and extrema in the report exclude non-finite values and say so. "
            "This is a data-quality signal, not proof of leakage.",
        ),
        _spec(
            "SPLIT.SCHEMA_MISMATCH",
            "Train and test schemas differ",
            Category.SPLIT_INTEGRITY,
            Severity.HIGH,
            Confidence.HIGH,
            "Models and metrics assume the same columns and compatible types. "
            "A mismatched schema can drop features silently or make a comparison invalid.",
            "Align column names and dtypes before evaluating. "
            "The finding lists missing, extra, and incompatible columns.",
            "Column order is ignored. "
            "Integer and floating dtypes are treated as compatible numeric types. "
            "Boolean columns are not treated as compatible with integers. "
            "Object and categorical text-like dtypes are treated as compatible with each other. "
            "Exact row overlap is not claimed when the schema differs.",
        ),
        _spec(
            "SPLIT.EXACT_ROW_OVERLAP",
            "Train and test contain the same record",
            Category.SPLIT_INTEGRITY,
            Severity.HIGH,
            Confidence.HIGH,
            "A record that appears in both partitions is not a held-out example. "
            "Metrics then partly measure memorization of rows the model can see during training.",
            "Remove overlapping records from one partition, or split by a key that keeps all copies together.",
            "Fingerprints ignore the pandas index and hash a canonical encoding of cell values "
            "over the shared column set. Exact overlap is evaluated only when both partitions "
            "have the same columns and compatible dtypes. "
            "Integer-valued floats match integers. The text '1' does not match the number 1. "
            "All missing sentinels match each other. "
            "Rows that differ only by an undeclared identifier still count as different records. "
            "This check does not detect near duplicates.",
        ),
        _spec(
            "SPLIT.GROUP_OVERLAP",
            "The same group appears in train and test",
            Category.SPLIT_INTEGRITY,
            Severity.HIGH,
            Confidence.HIGH,
            "Related rows that share a group can leak information across the split even when "
            "the full rows are not identical.",
            "Assign every row of a group to one partition, then evaluate only on groups that were held out.",
            "Only the supplied group column is used. Mlpipelinecheck cannot discover related entities "
            "that use a different key. Missing group values are ignored and do not form a shared group. "
            "Group identifiers are counted, not copied into the report. "
            "Numeric 1 and 1.0 match; the text '1' does not match the number 1.",
        ),
        _spec(
            "SPLIT.TIME_ORDER_VIOLATION",
            "Test timestamps occur before the training cutoff",
            Category.SPLIT_INTEGRITY,
            Severity.HIGH,
            Confidence.HIGH,
            "If the model is meant to be scored only on later data, earlier test rows are not a future holdout.",
            "Choose a cutoff and place every test timestamp at or after every training timestamp "
            "used to fit the model. Confirm the time column is the decision time, not an irrelevant date.",
            "The check runs only when a time column or a temporal split strategy is supplied. "
            "The default policy flags a test timestamp that is strictly earlier than the maximum "
            "non-null training timestamp. Ties at the cutoff are not flagged. "
            "Missing timestamps are not counted as violations and are reported as counts. "
            "A violation of this policy is not automatic proof that every other form of leakage exists.",
        ),
        _spec(
            "SPLIT.CLASS_DISTRIBUTION_SHIFT",
            "Class proportions differ between train and test",
            Category.SPLIT_INTEGRITY,
            Severity.LOW,
            Confidence.MEDIUM,
            "A large difference in class mix changes how a single accuracy number should be read "
            "and can indicate an unintended split.",
            "Compare the proportions in the report with the sampling plan. "
            "Use stratified splitting when similar class mix is required.",
            "A difference in proportions is not proof of leakage or of a defective split. "
            "The measure is the maximum absolute difference in class proportions, plus total variation "
            "as a summary. It is skipped when either partition has fewer non-null targets than "
            "class_shift_min_rows. Small samples are noisy.",
        ),
        _spec(
            "SPLIT.NUMERIC_DISTRIBUTION_SHIFT",
            "Numeric feature means differ between train and test",
            Category.SPLIT_INTEGRITY,
            Severity.LOW,
            Confidence.MEDIUM,
            "A large location shift can make a model look better or worse for reasons that have "
            "nothing to do with the decision rule learned on the training rows.",
            "Check whether the shift is expected for this population. "
            "Do not treat the finding as a leakage verdict.",
            "The statistic is the absolute standardized mean difference using a pooled standard deviation. "
            "It is descriptive, not a hypothesis test, and it is sensitive to outliers and to comparing many columns. "
            "Columns with too few finite values, or with zero variance, are skipped and listed. "
            "The target, declared identifiers, the group column, and the time column are excluded. "
            "A distribution difference is not proof of leakage.",
        ),
        _spec(
            "SPLIT.SMALL_TEST_PARTITION",
            "The test partition is smaller than the configured minimum",
            Category.SPLIT_INTEGRITY,
            Severity.LOW,
            Confidence.HIGH,
            "A very small test set makes accuracy, error, and distribution comparisons unstable.",
            "Increase the holdout if more labeled rows exist, and read metrics with the sample size in view.",
            "The default threshold is a row count, not a judgment that every small experiment is invalid. "
            "The optional fraction threshold is off unless configured.",
        ),
        _spec(
            "METRIC.ACCURACY_ONLY_IMBALANCED",
            "Accuracy is a weak summary of this classification target",
            Category.METRIC_VALIDITY,
            Severity.MEDIUM,
            Confidence.HIGH,
            "On an imbalanced or single-class target, accuracy can look strong while missing "
            "the outcomes that matter.",
            "Report balanced accuracy, macro or class-wise precision and recall, and the confusion matrix "
            "alongside accuracy.",
            "The finding does not say the model is poor. "
            "It says accuracy alone is easy to misread for this label distribution. "
            "A single-class evaluation target is reported at high severity because class-wise rates "
            "and ROC AUC are not interpretable. Milder imbalance stays at medium severity.",
        ),
        _spec(
            "METRIC.INVALID_AUC_INPUT",
            "ROC AUC cannot be computed from these labels and scores",
            Category.METRIC_VALIDITY,
            Severity.HIGH,
            Confidence.HIGH,
            "Reporting AUC from one class, or from scores that do not match the label set, "
            "produces a number that is not a ranking metric for the intended problem.",
            "Supply scores for a problem with at least two observed classes, with a shape that matches "
            "the documented binary or multiclass convention.",
            "AUC is attempted only when y_score is supplied. "
            "Wrong shapes and non-numeric scores raise an input error instead of a finding. "
            "A one-class target produces this finding when scores were supplied. "
            "Log loss and Brier score are omitted, without being invented, when the values are not probabilities.",
        ),
        _spec(
            "METRIC.SMALL_EVALUATION_SAMPLE",
            "The evaluation sample is smaller than the configured minimum",
            Category.METRIC_VALIDITY,
            Severity.LOW,
            Confidence.HIGH,
            "Metrics on very small samples move a lot when a few predictions change.",
            "Collect a larger holdout where possible, and avoid ranking models on tiny differences.",
            "The threshold is configuration. An interval is present only when bootstrap_samples "
            "was set on an independent prediction audit. This finding does not itself contain a width.",
        ),
        _spec(
            "METRIC.WIDE_CONFIDENCE_INTERVAL",
            "A metric confidence interval is wider than the configured threshold",
            Category.STATISTICAL_UNCERTAINTY,
            Severity.LOW,
            Confidence.MEDIUM,
            "A wide interval means the point metric is a weak summary of performance on similar samples.",
            "Increase the evaluation sample or use a resampling scheme that matches the dependence in the data.",
            "The interval, when requested, is a 95% percentile bootstrap that resamples rows "
            "independently with replacement and does not retrain the model. "
            "It is not valid for grouped or temporal dependence; those requests are rejected "
            "instead of being given an iid interval. "
            "If bootstrap_samples is omitted, this check is not applicable and no interval is reported. "
            "If an interval is computed but confidence_interval_max_width is omitted, the width is "
            "reported and not judged.",
        ),
    )
}

RULE_IDS: frozenset[str] = frozenset(RULES)

DATASET_RULE_IDS: tuple[str, ...] = (
    "DATA.EMPTY_DATASET",
    "DATA.MISSING_VALUES",
    "DATA.DUPLICATE_ROWS",
    "DATA.CONSTANT_FEATURE",
    "DATA.HIGH_CARDINALITY_FEATURE",
    "DATA.TARGET_IN_FEATURES",
    "DATA.TARGET_COPY_FEATURE",
    "DATA.CLASS_IMBALANCE",
    "DATA.NON_FINITE_VALUES",
)

SPLIT_RULE_IDS: tuple[str, ...] = (
    "SPLIT.SCHEMA_MISMATCH",
    "SPLIT.EXACT_ROW_OVERLAP",
    "DATA.DUPLICATE_ROWS",
    "SPLIT.GROUP_OVERLAP",
    "SPLIT.TIME_ORDER_VIOLATION",
    "SPLIT.CLASS_DISTRIBUTION_SHIFT",
    "SPLIT.NUMERIC_DISTRIBUTION_SHIFT",
    "SPLIT.SMALL_TEST_PARTITION",
    "DATA.TARGET_IN_FEATURES",
    "DATA.TARGET_COPY_FEATURE",
    "DATA.MISSING_VALUES",
    "DATA.CLASS_IMBALANCE",
)

PREDICTION_RULE_IDS: tuple[str, ...] = (
    "METRIC.ACCURACY_ONLY_IMBALANCED",
    "METRIC.INVALID_AUC_INPUT",
    "METRIC.SMALL_EVALUATION_SAMPLE",
    "METRIC.WIDE_CONFIDENCE_INTERVAL",
)
