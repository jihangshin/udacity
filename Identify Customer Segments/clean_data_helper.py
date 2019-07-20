import numpy as np
import pandas as pd

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def convert_to_nan(df, feat_info_df):
    # Identify missing or unknown data values and convert them to NaNs.
    #series
    sr_origin = feat_info_df['missing_or_unknown']
    for origin in sr_origin.iteritems():
        i_origin = origin[0]
        str_origin = origin[1]
        
        # convert string to list for 'missing_or_unknown' value
        stripped = str_origin.strip('[]')
        ls_missing_or_unknown = stripped.split(',')
        
        # change missing_or_unknown -> NaN
        attribute_name = feat_info_df.loc[i_origin, 'attribute']
        for i in ls_missing_or_unknown:
            # skip the null value 
            if len(i) > 0:
                # convert to int only if i is integer
                if RepresentsInt(i) == True:
                    df.loc[df[attribute_name] == int(i), attribute_name] = np.nan
                else:
                    df.loc[df[attribute_name] == i, attribute_name] = np.nan
    return df

def mark_multi_nonnum(df, feat_info_df):
    categorical_df = feat_info_df[feat_info_df['type'] == 'categorical']
    # Assess categorical variables: which are binary, which are multi-level, and
    # which one needs to be re-encoded?
    category_type_ls = []
    category_dtype_ls = []
    is_re_encode_required_ls = []
    for attribute in categorical_df['attribute']:
    #     print(attribute, azdias_df_lte_30[attribute].dtype, azdias_df_lte_30[attribute].unique())
        unique_ls = df[attribute].unique()
        unique_ls_len = len(unique_ls)    
        if unique_ls_len == 2 : 
            category_type_ls.append("binary")
        elif unique_ls_len > 2 : 
            category_type_ls.append("multi-level")
        else:
            print("singe level exist!! here!!")
            category_type_ls.append("single-level")
            
        dtype_col = df[attribute].dtype
        category_dtype_ls.append(dtype_col)
        
        if dtype_col == "object":
            is_re_encode_required_ls.append("yes")
        elif unique_ls_len > 2 : 
            is_re_encode_required_ls.append("yes")
        else:
            is_re_encode_required_ls.append("no")
        
    #add new column, "category_type" in categorical_df save # of unique categories
    categorical_df['category_type'] = category_type_ls
    categorical_df['category_dtype'] = category_dtype_ls
    categorical_df['is_re_encode_required'] = is_re_encode_required_ls

    return categorical_df, feat_info_df

def re_encode_multi_nonnum_marked(df, categorical_df):
    # Re-encode categorical variable(s) to be kept in the analysis.
    for row in categorical_df.index:
        if categorical_df.loc[row].is_re_encode_required == "yes":
            attribute = categorical_df.loc[row].attribute 
            dummy = pd.get_dummies(df[attribute], prefix=attribute[1:-1])
            
            df = pd.concat([df, dummy], axis=1) #add dummy columns at the end
            df = df.drop(columns=[attribute])  #drop the original column

    return df

def engineer_mixed(df, feat_info_df):
    # Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.

    #create two new column to save new variables
    for val in df['PRAEGENDE_JUGENDJAHRE']:
        if val == 1:
            df['decade'] = pd.Interval(left=1940, right=1949)
            df['movement'] = 0
        elif val == 2:
            df['decade'] = pd.Interval(left=1940, right=1949)
            df['movement'] = 1
        elif val == 3:
            df['decade'] = pd.Interval(left=1950, right=1959)
            df['movement'] = 0
        elif val == 4:
            df['decade'] = pd.Interval(left=1950, right=1959)
            df['movement'] = 1
        elif val == 5:
            df['decade'] = pd.Interval(left=1960, right=1969)
            df['movement'] = 0
        elif val == 6:
            df['decade'] = pd.Interval(left=1960, right=1969)
            df['movement'] = 1
        elif val == 7:
            df['decade'] = pd.Interval(left=1960, right=1969)
            df['movement'] = 1
        elif val == 8:
            df['decade'] = pd.Interval(left=1970, right=1979)
            df['movement'] = 0
        elif val == 9:
            df['decade'] = pd.Interval(left=1970, right=1979)
            df['movement'] = 1

    #exclude PRAEGENDE_JUGENDJAHRE 
    df = df.drop('PRAEGENDE_JUGENDJAHRE', 1)

    # Investigate "CAMEO_INTL_2015" and engineer two new variables.
    tens = []
    ones = []
    for val in df['CAMEO_INTL_2015']:
        tens.append(int(val[0]))
        ones.append(int(val[1]))
        
    df['tens'] = tens
    df['ones'] = ones

    #exclude CAMEO_INTL_2015 
    df = df.drop('CAMEO_INTL_2015', 1)

    #exclude all other mixed type
    mixed_df = feat_info_df[feat_info_df['type'] == 'mixed']
    mixed_attr = []
    for attribute in mixed_df['attribute']:
        mixed_attr.append(attribute)
        try:
            df = df.drop(attribute, 1)
        except KeyError:
            print(attribute, "was already excluded")
    
    return df

# def clean_data(df, feat_info_df):
#     # Put in code here to execute all main cleaning steps:
#     # convert missing value codes into NaNs, ...
#     df = convert_to_nan(df, feat_info_df)
    
#     # remove selected columns and rows, ...
#     #drop the column with most missing values
#     df = df.drop('TITEL_KZ', 1)
#     df_lte_30 = df[df.missing_count <= 30] 
#     # drop na for OneHotEncoder later
#     df_lte_30_nadropped_df = df_lte_30.dropna() # drop NaN rows

#     # select, re-encode, and engineer column values.
#     # mark re-encode necessary features in feat_info_df
#     categorical_df, feat_info_df = mark_multi_nonnum(df_lte_30_nadropped_df, feat_info_df)
#     df_lte_30_nadropped_encoded_df = re_encode_multi_nonnum_marked(df_lte_30_nadropped_df, categorical_df)
#     df_lte_30_nadropped_encoded_df = engineer_mixed(df_lte_30_nadropped_encoded_df, feat_info_df)
    
#     # Return the cleaned dataframe.
#     return df_lte_30_nadropped_encoded_df


