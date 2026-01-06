# geographic_grouping.py
"""
Helper module for grouping states and demographics into meaningful categories
"""

import pandas as pd
import re

# ============================================================================
# GEOGRAPHIC GROUPINGS
# ============================================================================

# U.S. Census Bureau Regional Divisions
US_REGIONS = {
    'Northeast': {
        'New England': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
        'Middle Atlantic': ['NJ', 'NY', 'PA']
    },
    'Midwest': {
        'East North Central': ['IL', 'IN', 'MI', 'OH', 'WI'],
        'West North Central': ['IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD']
    },
    'South': {
        'South Atlantic': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'DC'],
        'East South Central': ['AL', 'KY', 'MS', 'TN'],
        'West South Central': ['AR', 'LA', 'OK', 'TX']
    },
    'West': {
        'Mountain': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY'],
        'Pacific': ['AK', 'CA', 'HI', 'OR', 'WA']
    },
    'Territories': {
        'U.S. Territories': ['PR', 'VI', 'GU', 'AS', 'MP']
    }
}

# Flattened mappings for easy lookup
def get_state_to_region_mapping(level='region'):
    """
    Get state to region mapping at different granularity levels
    
    Parameters:
    -----------
    level : str
        'region' - 4 main regions (Northeast, Midwest, South, West)
        'division' - 9 census divisions
        'subdivision' - More granular (13 subdivisions)
    
    Returns:
    --------
    dict : State code -> Region/Division name
    """
    mapping = {}
    
    if level == 'region':
        # Simple 4-region grouping
        for region, divisions in US_REGIONS.items():
            for division, states in divisions.items():
                for state in states:
                    mapping[state] = region
    
    elif level == 'division':
        # 9 census divisions
        for region, divisions in US_REGIONS.items():
            for division, states in divisions.items():
                for state in states:
                    mapping[state] = division
    
    elif level == 'subdivision':
        # Most granular - subdivisions within divisions
        for region, divisions in US_REGIONS.items():
            for division, states in divisions.items():
                # Create subdivision names like "Northeast - New England"
                subdivision = f"{region} - {division}"
                for state in states:
                    mapping[state] = subdivision
    
    return mapping


def aggregate_by_region(state_df, level='region'):
    """
    Aggregate state-level data to regional level
    
    Parameters:
    -----------
    state_df : pd.DataFrame
        DataFrame with 'Locationabbr', 'Response', 'agg_persons', 'agg_ss' columns
    level : str
        Granularity level: 'region', 'division', or 'subdivision'
    
    Returns:
    --------
    pd.DataFrame : Aggregated data at regional level
    """
    if state_df is None or state_df.empty:
        return state_df
    
    df = state_df.copy()
    mapping = get_state_to_region_mapping(level)
    
    # Map states to regions
    df['Region'] = df['Locationabbr'].map(mapping)
    
    # Drop unmapped states (if any)
    df = df.dropna(subset=['Region'])
    
    # Aggregate by region and response
    agg = df.groupby(['Region', 'Response'], as_index=False).agg(
        agg_persons=('agg_persons', 'sum'),
        agg_ss=('agg_ss', 'sum')
    )
    
    # Recalculate percentages and confidence intervals
    agg['agg_percent'] = agg['agg_persons'] * 100.0 / agg['agg_ss']
    agg['agg_percent_sdev'] = (
        agg['agg_percent'] * (100 - agg['agg_percent']) / agg['agg_ss']
    ) ** 0.5
    agg['agg_low_ci_limit'] = agg['agg_percent'] - 2 * agg['agg_percent_sdev']
    agg['agg_high_ci_limit'] = agg['agg_percent'] + 2 * agg['agg_percent_sdev']
    agg['err'] = agg['agg_high_ci_limit'] - agg['agg_percent']
    
    return agg


# ============================================================================
# DEMOGRAPHIC GROUPINGS
# ============================================================================

def _normalize_age_label(label: str) -> str:
    """
    Turn messy age text into canonical forms like '18-24', '25-34', '65+'.
    Handles patterns like:
      '18-24', '18 to 24', '18-24 years', '65 years or older', '75+'
    """
    if pd.isna(label):
        return label
    s = str(label).strip().lower()

    # Extract all numbers
    nums = re.findall(r"\d+", s)
    if not nums:
        return s

    # 65+ / 75+ / '65 years or older'
    if "+" in s or "older" in s or "more" in s or "above" in s:
        return f"{nums[0]}+"

    # '18-24', '18 to 24', '18 – 24'
    if len(nums) >= 2:
        return f"{nums[0]}-{nums[1]}"

    # Fallback – just return the first number
    return nums[0]


def group_age_ranges(age_df, level='simple'):
    """
    Group age ranges into broader or narrower categories.

    Parameters
    ----------
    age_df : pd.DataFrame
        DataFrame with 'Break_Out', 'Response', 'agg_persons', 'agg_ss' columns.
    level : {'simple', 'standard', 'detailed'}
    """
    if age_df is None or len(age_df) == 0:
        return age_df

    # Work on a copy
    df = age_df.copy()

    # Define age grouping mappings in canonical form
    age_mappings = {
        'simple': {
            '18-24': ['18-24', '21-25', '21-30'],
            '25-34': ['25-34', '26-35', '31-40'],
            '35-44': ['35-44', '36-45', '40-49', '41-50'],
            '45-54': ['45-54', '46-55', '50-59', '51-60'],
            '55-64': ['55-64', '56-65', '60-64', '60-69', '61-65'],
            '65+':   ['65+', '65-74', '65-75', '70-74', '70-75', '70-80', '75+'],
        },
        'standard': {
            '18-24': ['18-24', '21-25', '21-30'],
            '25-34': ['25-34', '26-35'],
            '35-44': ['35-44', '36-45', '31-40'],
            '45-54': ['45-54', '46-55', '40-49', '41-50'],
            '55-64': ['55-64', '56-65', '50-59', '51-60'],
            '65-74': ['65-74', '65-75', '60-64', '60-69', '61-65', '70-74'],
            '75+':   ['75+', '70-75', '70-80'],
        },
        'detailed': {}  # handled separately
    }

    # Detailed → return as-is
    if level == 'detailed':
        return df

    if level not in age_mappings:
        raise ValueError(f"Invalid level for age grouping: {level}")

    # Normalize the Break_Out values to canonical keys
    df['age_norm'] = df['Break_Out'].astype(str).apply(_normalize_age_label)

    # Build reverse mapping using the same normalization
    reverse_mapping = {}
    for grouped_age, originals in age_mappings[level].items():
        for orig in originals:
            key = _normalize_age_label(orig)
            reverse_mapping[key] = grouped_age

    # Map normalized age to grouped age
    df['Grouped_Age'] = df['age_norm'].map(reverse_mapping)

    # Any ages not covered by mapping keep their original Break_Out label
    df['Grouped_Age'] = df['Grouped_Age'].fillna(df['Break_Out'])

    # Aggregate by grouped age + response
    agg = df.groupby(['Grouped_Age', 'Response'], as_index=False).agg(
        agg_persons=('agg_persons', 'sum'),
        agg_ss=('agg_ss', 'sum'),
    )

    # Recompute percent and 95% CI
    agg['agg_percent'] = agg['agg_persons'] * 100.0 / agg['agg_ss']
    agg['agg_percent_sdev'] = (
        agg['agg_percent'] * (100 - agg['agg_percent']) / agg['agg_ss']
    ) ** 0.5
    agg['agg_low_ci_limit'] = agg['agg_percent'] - 2 * agg['agg_percent_sdev']
    agg['agg_high_ci_limit'] = agg['agg_percent'] + 2 * agg['agg_percent_sdev']
    agg['err'] = agg['agg_high_ci_limit'] - agg['agg_percent']

    # Rename Grouped_Age -> Break_Out so plotting code doesn't change
    agg = agg.rename(columns={'Grouped_Age': 'Break_Out'})

    return agg


def group_income_ranges(income_df, level='simple'):
    """
    Group income ranges into broader categories
    
    Parameters:
    -----------
    income_df : pd.DataFrame
        DataFrame with 'Break_Out' column containing income ranges
    level : str
        'simple' - 3 broad groups (Low, Middle, High)
        'standard' - 5 groups
        'detailed' - All available groups
    
    Returns:
    --------
    pd.DataFrame : Aggregated income data
    """
    if income_df is None or income_df.empty:
        return income_df
    
    df = income_df.copy()
    
    income_mappings = {
        'simple': {
            'Low Income (< $35k)': [
                'Less than $15,000',
                '$15,000-$24,999',
                '$25,000-$34,999'
            ],
            'Middle Income ($35k-$75k)': [
                '$35,000-$49,999',
                '$50,000-$74,999',
                '$50,000-$99,999'
            ],
            'High Income ($75k+)': [
                '$75,000+',
                '$50,000+',
                '$100,000-$199,999',
                '$200,000+'
            ]
        },
        'standard': {
            'Very Low (< $25k)': ['Less than $15,000', '$15,000-$24,999'],
            'Low ($25k-$50k)': ['$25,000-$34,999', '$35,000-$49,999'],
            'Middle ($50k-$100k)': ['$50,000-$74,999', '$50,000-$99,999', '$75,000-$99,999'],
            'Upper Middle ($100k-$200k)': ['$100,000-$199,999'],
            'High ($200k+)': ['$200,000+']
        }
    }
    
    if level == 'detailed':
        return df
    
    # Create reverse mapping
    reverse_mapping = {}
    for grouped_income, original_incomes in income_mappings[level].items():
        for orig in original_incomes:
            reverse_mapping[orig] = grouped_income
    
    df['Grouped_Income'] = df['Break_Out'].map(reverse_mapping)
    df['Grouped_Income'] = df['Grouped_Income'].fillna(df['Break_Out'])
    
    # Aggregate
    agg = df.groupby(['Grouped_Income', 'Response'], as_index=False).agg(
        agg_persons=('agg_persons', 'sum'),
        agg_ss=('agg_ss', 'sum')
    )
    
    agg['agg_percent'] = agg['agg_persons'] * 100.0 / agg['agg_ss']
    agg['agg_percent_sdev'] = (
        agg['agg_percent'] * (100 - agg['agg_percent']) / agg['agg_ss']
    ) ** 0.5
    agg['agg_low_ci_limit'] = agg['agg_percent'] - 2 * agg['agg_percent_sdev']
    agg['agg_high_ci_limit'] = agg['agg_percent'] + 2 * agg['agg_percent_sdev']
    agg['err'] = agg['agg_high_ci_limit'] - agg['agg_percent']
    
    agg = agg.rename(columns={'Grouped_Income': 'Break_Out'})
    
    return agg


def group_education_levels(edu_df, level='simple'):
    """
    Group education levels
    
    Parameters:
    -----------
    edu_df : pd.DataFrame
        DataFrame with 'Break_Out' column containing education levels
    level : str
        'simple' - 3 groups (No College, Some College, College+)
        'standard' - 4 groups (standard breakout)
        'detailed' - All available
    
    Returns:
    --------
    pd.DataFrame : Aggregated education data
    """
    if edu_df is None or edu_df.empty:
        return edu_df
    
    df = edu_df.copy()
    
    edu_mappings = {
        'simple': {
            'No College': ['Less than H.S.', 'H.S. or G.E.D.'],
            'Some College': ['Some post-H.S.', 'Some college'],
            'College Graduate+': ['College graduate', 'College or higher']
        }
    }
    
    if level in ['standard', 'detailed']:
        return df
    
    reverse_mapping = {}
    for grouped_edu, original_edus in edu_mappings[level].items():
        for orig in original_edus:
            reverse_mapping[orig] = grouped_edu
    
    df['Grouped_Edu'] = df['Break_Out'].map(reverse_mapping)
    df['Grouped_Edu'] = df['Grouped_Edu'].fillna(df['Break_Out'])
    
    agg = df.groupby(['Grouped_Edu', 'Response'], as_index=False).agg(
        agg_persons=('agg_persons', 'sum'),
        agg_ss=('agg_ss', 'sum')
    )
    
    agg['agg_percent'] = agg['agg_persons'] * 100.0 / agg['agg_ss']
    agg['agg_percent_sdev'] = (
        agg['agg_percent'] * (100 - agg['agg_percent']) / agg['agg_ss']
    ) ** 0.5
    agg['agg_low_ci_limit'] = agg['agg_percent'] - 2 * agg['agg_percent_sdev']
    agg['agg_high_ci_limit'] = agg['agg_percent'] + 2 * agg['agg_percent_sdev']
    agg['err'] = agg['agg_high_ci_limit'] - agg['agg_percent']
    
    agg = agg.rename(columns={'Grouped_Edu': 'Break_Out'})
    
    return agg


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_granularity_description(category, level):
    """Get human-readable description of granularity level"""
    descriptions = {
        'geography': {
            'state': 'Individual States (50+ locations)',
            'region': 'Census Regions (4 regions)',
            'division': 'Census Divisions (9 divisions)',
            'subdivision': 'Regional Subdivisions (13 areas)'
        },
        'age': {
            'simple': 'Broad Age Groups (6 groups)',
            'standard': 'Standard Age Ranges (8 groups)',
            'detailed': 'All Available Ages (15-20 groups)'
        },
        'income': {
            'simple': 'Income Tiers (3 tiers)',
            'standard': 'Income Brackets (5 brackets)',
            'detailed': 'All Available Ranges (7+ ranges)'
        },
        'education': {
            'simple': 'Education Attainment (3 levels)',
            'standard': 'Standard Breakout (4 levels)',
            'detailed': 'All Available Levels'
        }
    }
    
    return descriptions.get(category, {}).get(level, f"{category} - {level}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Show region mappings
    print("=== REGION LEVEL ===")
    region_map = get_state_to_region_mapping('region')
    for state, region in sorted(region_map.items())[:10]:
        print(f"{state} -> {region}")
    
    print("\n=== DIVISION LEVEL ===")
    division_map = get_state_to_region_mapping('division')
    for state, division in sorted(division_map.items())[:10]:
        print(f"{state} -> {division}")
    
    print("\n=== GRANULARITY DESCRIPTIONS ===")
    for cat in ['geography', 'age', 'income', 'education']:
        print(f"\n{cat.upper()}:")
        for level in ['simple', 'standard', 'detailed']:
            if cat == 'geography':
                level = ['region', 'division', 'subdivision'][
                    ['simple', 'standard', 'detailed'].index(level)
                ]
            desc = get_granularity_description(cat, level)
            print(f"  {level}: {desc}")