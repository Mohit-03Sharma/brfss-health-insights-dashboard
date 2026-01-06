# aggregations.py
import pandas as pd


def add_person_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate person counts and sample sizes from BRFSS summary data.
    
    IMPORTANT: In the BRFSS CSV:
    - Sample_Size column = actual person count (persons)
    - Data_value = percentage
    - True sample size = persons * 100 / Data_value
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Sample_Size and Data_value columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'persons' and recalculated 'Sample_Size' columns
        
    Notes:
    ------
    Ensures both columns are numeric; non-numeric entries become NaN and are dropped.
    """
    df = df.copy()

    df["Sample_Size"] = pd.to_numeric(df["Sample_Size"], errors="coerce")
    df["Data_value"]  = pd.to_numeric(df["Data_value"],  errors="coerce")

    df = df.dropna(subset=["Sample_Size", "Data_value"])

    # Sample_Size in CSV is actually the person count
    df["persons"] = df["Sample_Size"].copy()
    # Calculate the true sample size from percentage
    df["Sample_Size"] = df["persons"] * 100.0 / df["Data_value"]
    
    return df


def apply_response_cleaning(df: pd.DataFrame, clean: bool) -> pd.DataFrame:
    """
    Optionally remove low-quality response categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Response column
    clean : bool
        If True, remove responses like 'don't know', 'refused', etc.
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame (or unchanged if clean=False)
    """
    if not clean:
        return df

    if "Response" not in df.columns:
        return df

    patterns = [
        "don't know",
        "dont know",
        "refused",
        "refuse",
        "missing",
        "unknown",
        "not asked",
        "declined",
        "no response",
        "na",
    ]
    pattern = "|".join(patterns)
    mask = df["Response"].astype(str).str.contains(pattern, case=False, na=False)
    
    removed = mask.sum()
    if removed > 0:
        print(f"  ℹ Removed {removed} rows with low-quality responses")
    
    return df[~mask]


def filter_question(df: pd.DataFrame, question_text: str) -> pd.DataFrame:
    """
    Return subset for a single question.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full BRFSS dataset
    question_text : str
        Exact question text to filter
        
    Returns:
    --------
    pd.DataFrame
        Subset containing only the specified question
    """
    return df[df["Question"] == question_text].copy()


def _finalize_agg(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentages, standard deviations, confidence intervals from aggregated data.
    
    Parameters:
    -----------
    agg : pd.DataFrame
        Aggregated data with agg_persons and agg_ss columns
        
    Returns:
    --------
    pd.DataFrame
        Aggregated data with calculated statistics:
        - agg_percent: percentage
        - agg_percent_sdev: standard deviation
        - agg_low_ci_limit: lower 95% CI
        - agg_high_ci_limit: upper 95% CI
        - err: half-width of error bar (for plotting)
    """
    agg = agg.copy()
    
    # Calculate percentage
    agg["agg_percent"] = agg["agg_persons"] * 100.0 / agg["agg_ss"]
    
    # Calculate standard deviation using binomial approximation
    agg["agg_percent_sdev"] = (
        agg["agg_percent"] * (100 - agg["agg_percent"]) / agg["agg_ss"]
    ) ** 0.5
    
    # Calculate 95% confidence intervals (±2 standard deviations)
    agg["agg_low_ci_limit"] = agg["agg_percent"] - 2 * agg["agg_percent_sdev"]
    agg["agg_high_ci_limit"] = agg["agg_percent"] + 2 * agg["agg_percent_sdev"]
    
    # Error bar half-width for plotting
    agg["err"] = agg["agg_high_ci_limit"] - agg["agg_percent"]
    
    return agg


def aggregate_overall(df_q: pd.DataFrame, clean: bool = False) -> pd.DataFrame:
    """
    Aggregate 'Overall' (CAT1) across all years and states.
    
    Parameters:
    -----------
    df_q : pd.DataFrame
        Question-filtered data
    clean : bool, default=False
        Whether to remove low-quality responses
        
    Returns:
    --------
    pd.DataFrame
        Aggregated overall statistics by Response
    """
    df = df_q[df_q["BreakOutCategoryID"] == "CAT1"].copy()
    df = apply_response_cleaning(df, clean)
    df = add_person_counts(df)

    if df.empty:
        return df

    agg = (
        df.groupby("Response", as_index=False)
          .agg(agg_persons=("persons", "sum"),
               agg_ss=("Sample_Size", "sum"))
    )

    agg = _finalize_agg(agg)
    return agg


def aggregate_by_year(df_q: pd.DataFrame, clean: bool = False) -> pd.DataFrame:
    """
    Temporal panel: aggregate by Year using Overall breakout (CAT1).
    
    Parameters:
    -----------
    df_q : pd.DataFrame
        Question-filtered data
    clean : bool, default=False
        Whether to remove low-quality responses
        
    Returns:
    --------
    pd.DataFrame
        Aggregated statistics by Year and Response
    """
    df = df_q[df_q["BreakOutCategoryID"] == "CAT1"].copy()
    df = apply_response_cleaning(df, clean)
    df = add_person_counts(df)

    if df.empty:
        return df

    agg = (
        df.groupby(["Year", "Response"], as_index=False)
          .agg(agg_persons=("persons", "sum"),
               agg_ss=("Sample_Size", "sum"))
    )

    agg = _finalize_agg(agg)
    return agg


def aggregate_by_state(df_q: pd.DataFrame, clean: bool = False) -> pd.DataFrame:
    """
    Geographic panel: aggregate by Locationabbr using Overall breakout (CAT1).
    
    Parameters:
    -----------
    df_q : pd.DataFrame
        Question-filtered data
    clean : bool, default=False
        Whether to remove low-quality responses
        
    Returns:
    --------
    pd.DataFrame
        Aggregated statistics by Locationabbr and Response
    """
    df = df_q[df_q["BreakOutCategoryID"] == "CAT1"].copy()
    df = apply_response_cleaning(df, clean)
    df = add_person_counts(df)

    if df.empty:
        return df

    agg = (
        df.groupby(["Locationabbr", "Response"], as_index=False)
          .agg(agg_persons=("persons", "sum"),
               agg_ss=("Sample_Size", "sum"))
    )

    agg = _finalize_agg(agg)
    return agg


def aggregate_by_breakout_category(df_q: pd.DataFrame, cat_id: str, clean: bool = False) -> pd.DataFrame:
    """
    Generic aggregator for CAT2–CAT6 (gender, age, race, education, income).
    Groups by Break_Out within that category.
    
    Parameters:
    -----------
    df_q : pd.DataFrame
        Question-filtered data
    cat_id : str
        Category ID: CAT2 (gender), CAT3 (age), CAT4 (race), 
        CAT5 (education), CAT6 (income)
    clean : bool, default=False
        Whether to remove low-quality responses
        
    Returns:
    --------
    pd.DataFrame
        Aggregated statistics by Break_Out and Response
    """
    df = df_q[df_q["BreakOutCategoryID"] == cat_id].copy()
    df = apply_response_cleaning(df, clean)
    df = add_person_counts(df)

    if df.empty:
        return df

    agg = (
        df.groupby(["Break_Out", "Response"], as_index=False)
          .agg(agg_persons=("persons", "sum"),
               agg_ss=("Sample_Size", "sum"))
    )

    agg = _finalize_agg(agg)
    return agg