# data_prep.py
import pandas as pd

BRFSS_CSV_PATH = "data/Prevalence_Data.csv"   
BRFSS_PARQUET_PATH = "data/brfss_prevalence.parquet"


def merge_response_id(series: pd.Series) -> pd.Series:
    """
    Merge ResponseID values to consolidate similar response categories.
    
    Parameters:
    -----------
    series : pd.Series
        Series of ResponseID values to merge
        
    Returns:
    --------
    pd.Series
        Series with merged ResponseID values
    """
    mapping = {
        "RESP025": "RESP137",
        "RESP026": "RESP172",
        "RESP029": "RESP141",
        "RESP230": "RESP020",
        "RESP231": "RESP020",
        "RESP232": "RESP020",
        "RESP196": "RESP199",
        "RESP197": "RESP199",
        "RESP198": "RESP199",
        "RESP199": "RESP199",
        "RESP200": "RESP008",
        "RESP194": "RESP005",
        "RESP195": "RESP006",
    }
    return series.replace(mapping)


def merge_response(response_id: pd.Series, response: pd.Series) -> pd.Series:
    """
    Merge Response values to match merged ResponseID categories.
    
    Parameters:
    -----------
    response_id : pd.Series
        Series of ResponseID values (already merged)
    response : pd.Series
        Series of Response text values to merge
        
    Returns:
    --------
    pd.Series
        Series with merged and lowercased Response values
    """
    resp = response.copy()

    resp.loc[response_id.str.contains("RESP137", na=False)] = "Employed"
    resp.loc[response_id.str.contains("RESP172", na=False)] = "Self-employed"
    resp.loc[response_id.str.contains("RESP141", na=False)] = "Homemaker"
    resp.loc[response_id.str.contains("RESP020", na=False)] = "$50,000+"
    resp.loc[response_id.str.contains("RESP199", na=False)] = "A/A Native, Asian,Other"
    resp.loc[response_id.str.contains("RESP008", na=False)] = "Multiracial"
    resp.loc[response_id.str.contains("RESP005", na=False)] = "White"
    resp.loc[response_id.str.contains("RESP006", na=False)] = "Black"

    return resp.str.lower()


def merge_breakout_id(series: pd.Series) -> pd.Series:
    """
    Merge BreakoutID values to consolidate demographic categories.
    
    Parameters:
    -----------
    series : pd.Series
        Series of BreakoutID values to merge
        
    Returns:
    --------
    pd.Series
        Series with merged BreakoutID values
    """
    mapping = {
        "INCOME01": "INCOME1",
        "INCOME02": "INCOME2",
        "INCOME03": "INCOME3",
        "INCOME04": "INCOME4",
        "INCOME05": "INCOME5",
        "INCOME06": "INCOME5",
        "INCOME07": "INCOME5",
        "RACE01": "RACE1",
        "RACE02": "RACE2",
        "RACE08": "RACE3",
        "RACE04": "RACE4",
        "RACE05": "RACE4",
        "RACE06": "RACE4",
        "RACE03": "RACE4",
        "RACE07": "RACE5",
    }
    return series.replace(mapping)


def merge_break_out(breakout_id: pd.Series, break_out: pd.Series) -> pd.Series:
    """
    Merge Break_Out values to match merged BreakoutID categories.
    
    Parameters:
    -----------
    breakout_id : pd.Series
        Series of BreakoutID values (already merged)
    break_out : pd.Series
        Series of Break_Out text values to merge
        
    Returns:
    --------
    pd.Series
        Series with merged Break_Out values
    """
    bo = break_out.copy()

    bo.loc[breakout_id.str.contains("INCOME5", na=False)] = "$50,000+"
    bo.loc[breakout_id.str.contains("RACE1",   na=False)] = "White"
    bo.loc[breakout_id.str.contains("RACE2",   na=False)] = "Black"
    bo.loc[breakout_id.str.contains("RACE3",   na=False)] = "Hispanic"
    bo.loc[breakout_id.str.contains("RACE4",   na=False)] = "A/A Native, Asian,Other"
    bo.loc[breakout_id.str.contains("RACE5",   na=False)] = "Multiracial"

    return bo


def load_brfss(use_parquet: bool = True) -> pd.DataFrame:
    """
    Load and pre-process the BRFSS prevalence summary data.
    
    Parameters:
    -----------
    use_parquet : bool, default=True
        If True, attempt to load from parquet file first
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and processed BRFSS data
        
    Notes:
    ------
    - Reads CSV (or Parquet if available)
    - Applies ResponseID / BreakoutID merges
    - Removes US and UW aggregates
    - Ensures Year is numeric
    """
    if use_parquet:
        try:
            df = pd.read_parquet(BRFSS_PARQUET_PATH)
            print(f"✓ Loaded {len(df):,} rows from parquet cache")
            return df
        except FileNotFoundError:
            print("⚠ Parquet not found, loading from CSV...")

    print("Loading CSV (this may take a moment)...")
    df = pd.read_csv(BRFSS_CSV_PATH, low_memory=False)

    # Apply merges similar to R
    print("Applying response merges...")
    df["ResponseID"] = merge_response_id(df["ResponseID"].astype(str))
    df["Response"]   = merge_response(df["ResponseID"], df["Response"].astype(str))

    print("Applying breakout merges...")
    df["BreakoutID"] = merge_breakout_id(df["BreakoutID"].astype(str))
    df["Break_Out"]  = merge_break_out(df["BreakoutID"], df["Break_Out"].astype(str))

    # Exclude US / UW aggregates
    df = df[~df["Locationabbr"].isin(["US", "UW"])]

    # Ensure Year is numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    # Optionally save to Parquet for faster reloads
    if use_parquet:
        try:
            df.to_parquet(BRFSS_PARQUET_PATH)
            print(f"✓ Saved to parquet cache for faster future loads")
        except Exception as e:
            print(f"⚠ Could not save parquet: {e}")

    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def build_layerQ(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Class / Topic / Question hierarchy for navigation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full BRFSS dataset
        
    Returns:
    --------
    pd.DataFrame
        Unique combinations of Class, Topic, and Question
    """
    layerQ = df[["Class", "Topic", "Question"]].drop_duplicates().reset_index(drop=True)
    print(f"✓ Built question hierarchy: {len(layerQ)} unique questions")
    return layerQ