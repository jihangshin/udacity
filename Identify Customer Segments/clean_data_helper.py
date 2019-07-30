import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def convert_to_nan(df, feat_info_df):
    # Identify missing or unknown data values and convert them to NaNs.
    #series
    for row in feat_info_df.iterrows():
        row_val = row[1]
        column = df[row_val['attribute']]
        missing_values = row_val['missing_or_unknown'][1:-1].split(',')
        if missing_values != ['']:
            for mv in missing_values:
                if mv != 'X' and mv != 'XX':
                    column.replace(int(mv), np.nan, inplace = True)
                else:
                    column.replace(mv, np.nan, inplace = True)

        df[row_val['attribute']] = column

    return df

def drop_non_existing_col(feat_info_df):
    # drop 'AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP', 'KBA05_BAUMAX'
    #, bc the columns are not existed anymore in azdias dataframe
    feat_info_df = feat_info_df[feat_info_df.attribute != 'AGER_TYP']
    feat_info_df = feat_info_df[feat_info_df.attribute != 'GEBURTSJAHR']
    feat_info_df = feat_info_df[feat_info_df.attribute != 'TITEL_KZ']
    feat_info_df = feat_info_df[feat_info_df.attribute != 'ALTER_HH']
    feat_info_df = feat_info_df[feat_info_df.attribute != 'KK_KUNDENTYP']
    feat_info_df = feat_info_df[feat_info_df.attribute != 'KBA05_BAUMAX']

    return feat_info_df

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

def _impute_nan(df):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputed = imp.fit_transform(df)
    imputed_df = pd.DataFrame(imputed)

    # Set columns index back to original label
    imputed_df.columns = df.columns

    return imputed_df

def engineer_mixed(df, feat_info_df):
    # Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.

    #create two new column to save new variables
    df['DECADE'] = pd.cut(df['PRAEGENDE_JUGENDJAHRE'], [1,3,5,8,10,14,16], False, [40,50,60,70,80,90]) 
    df['IS_MAINSTREAM_MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE'].isin([1,3,5,8,10,12,14]).astype(float)

    #exclude PRAEGENDE_JUGENDJAHRE 
    df = df.drop('PRAEGENDE_JUGENDJAHRE', 1)

    # Impute NaN values with mean of each colums
    # Perform Imputing here bc 'CAMEO_INTL_2015' contains nan value needed to be replaced before engineering process
    imputed_df = _impute_nan(df)

    # Investigate "CAMEO_INTL_2015" and engineer two new variables.
    tens = []
    ones = []
    for val in imputed_df['CAMEO_INTL_2015']:
        str_val = str(val)
        tens.append(int(str_val[0]))
        ones.append(int(str_val[1]))
        
    imputed_df['tens'] = tens
    imputed_df['ones'] = ones

    #exclude CAMEO_INTL_2015 
    imputed_df = imputed_df.drop('CAMEO_INTL_2015', 1)

    #exclude all other mixed type
    mixed_df = feat_info_df[feat_info_df['type'] == 'mixed']
    mixed_attr = []
    for attribute in mixed_df['attribute']:
        mixed_attr.append(attribute)
        try:
            imputed_df = imputed_df.drop(attribute, 1)
        except KeyError:
            print(attribute, "was already excluded")
    
    return imputed_df



