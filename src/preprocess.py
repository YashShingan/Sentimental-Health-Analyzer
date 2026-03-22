import pandas as pd

def preprocess_data(df):

    # Drop student_id because it is not useful for prediction
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    # Convert categorical columns to numbers
    df = pd.get_dummies(
        df,
        columns=[
            "gender",
            "academic_level",
            "country",
            "most_used_platform",
            "affects_academic_performance",
            "relationship_status",
            "conflicts_over_social_media"
        ],
        drop_first=True
    )

    return df