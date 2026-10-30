"""
BeanVision AI — sample-level visual quality assessment.

This module aggregates the individual bean predictions from the
existing MLP classifier.

IMPORTANT:
The score weights and category boundaries are provisional.
This is not an official coffee grade or a validated quality score.
"""

from collections import Counter


# These weights are initial engineering assumptions, NOT official standards.
DEFAULT_DEFECT_PENALTIES = {
    "good": 0,
    "cut": 15,
    "husk": 10,
    "immature": 20,
    "parchment": 10,
    "partial-black": 25,
    "shell": 15,
}


def get_quality_category(score):
    """Convert the provisional visual score into a descriptive category."""

    if score >= 90:
        return "Excellent visual indicators"

    if score >= 75:
        return "Good visual indicators"

    if score >= 60:
        return "Fair visual indicators"

    if score >= 40:
        return "Poor visual indicators"

    return "Very poor visual indicators"


def assess_quality(
    detections,
    classes,
    penalty_weights=None,
    low_confidence_threshold=0.60,
):
    """
    Calculate sample-level statistics and a provisional visual score.

    Args:
        detections:
            List of detection dictionaries from Pipeline.predict().

        classes:
            The configured seven classification labels.

        penalty_weights:
            Optional dictionary mapping defect class names to penalties.

        low_confidence_threshold:
            Predictions below this probability are flagged for review.

    Returns:
        Dictionary containing quality statistics, defect distribution,
        recommendations, and a provisional score.
    """

    weights = dict(DEFAULT_DEFECT_PENALTIES)

    if penalty_weights is not None:
        weights.update(penalty_weights)

    total = len(detections)

    # Count only classes that the classifier is configured to predict.
    class_counts = Counter(
        detection["predicted_class"]
        for detection in detections
        if detection.get("predicted_class") in classes
    )

    # Keep counts for all configured classes, including zero-count classes.
    class_counts = {
        class_name: class_counts.get(class_name, 0)
        for class_name in classes
    }

    good_count = class_counts.get("good", 0)
    classified_count = sum(class_counts.values())
    defect_count = classified_count - good_count

    if classified_count == 0:
        return {
            "available": False,
            "message": (
                "No valid bean classifications are available. "
                "Upload a clearer image containing visible coffee beans."
            ),
            "total_detected": total,
            "classified_beans": 0,
            "good_beans": 0,
            "defective_beans": 0,
            "good_percentage": None,
            "defect_percentage": None,
            "quality_score": None,
            "quality_category": "Unable to assess",
            "defect_distribution": {},
            "recommendations": [
                "Check the image and segmentation detections."
            ],
            "low_confidence_count": 0,
        }

    good_percentage = 100.0 * good_count / classified_count
    defect_percentage = 100.0 * defect_count / classified_count

    # Calculate the mean penalty per classified bean.
    total_penalty = 0.0

    for class_name, count in class_counts.items():
        if class_name == "good":
            continue

        total_penalty += count * float(weights.get(class_name, 0))

    # Score is a provisional visual indicator, not an official grade.
    average_penalty = total_penalty / classified_count
    quality_score = round(
        max(0.0, min(100.0, 100.0 - average_penalty)),
        1,
    )

    # Show the number and proportion of each detected class.
    defect_distribution = {}

    for class_name, count in class_counts.items():
        if class_name == "good":
            continue

        defect_distribution[class_name] = {
            "count": count,
            "percentage": round(
                100.0 * count / classified_count,
                1,
            ),
        }

    # Most common defect category, excluding the "good" class.
    nonzero_defects = {
        name: values
        for name, values in defect_distribution.items()
        if values["count"] > 0
    }

    if nonzero_defects:
        most_common_defect = max(
            nonzero_defects,
            key=lambda name: nonzero_defects[name]["count"],
        )
    else:
        most_common_defect = None

    # Flag low-confidence predictions for human review.
    low_confidence_count = sum(
        1
        for detection in detections
        if detection.get("score", 0.0) < low_confidence_threshold
    )

    recommendations = []

    if defect_count == 0:
        recommendations.append(
            "No defects were predicted in the classified beans. "
            "Review the image to verify the detections."
        )
    else:
        recommendations.append(
            f"{defect_percentage:.1f}% of classified beans were "
            "predicted to belong to defect categories."
        )

        if most_common_defect:
            recommendations.append(
                f"Review the {most_common_defect} category, "
                "which is the most frequent predicted defect."
            )

    if low_confidence_count:
        recommendations.append(
            f"{low_confidence_count} bean prediction(s) have low "
            "confidence and should be checked manually."
        )

    if good_percentage < 50:
        recommendations.append(
            "Less than half of the classified beans were predicted "
            "as good. Inspect the sample carefully."
        )

    # Avoid presenting a precise-looking score without a caveat.
    if classified_count < 20:
        recommendations.append(
            "Small sample warning: the result may not represent "
            "the quality of the entire coffee batch."
        )

    recommendations.append(
        "This is an experimental visual score based on model "
        "predictions and provisional penalty weights, not an "
        "official coffee grade."
    )

    return {
        "available": True,
        "message": "Visual assessment calculated successfully.",
        "total_detected": total,
        "classified_beans": classified_count,
        "good_beans": good_count,
        "defective_beans": defect_count,
        "good_percentage": round(good_percentage, 1),
        "defect_percentage": round(defect_percentage, 1),
        "quality_score": quality_score,
        "quality_category": get_quality_category(quality_score),
        "most_common_defect": most_common_defect,
        "defect_distribution": defect_distribution,
        "recommendations": recommendations,
        "low_confidence_count": low_confidence_count,
        "scoring_method": (
            "100 minus the average configured defect penalty per "
            "classified bean; weights are provisional."
        ),
    }