import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

print('Diagnostic script starting')

df = pd.read_csv('bank-additional.csv', sep=';')
TARGET='y'
X = df.drop(columns=[TARGET]); y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

def handle_pdays(X):
    X = X.copy()
    X['was_contacted_before'] = (X['pdays'] != 999).astype(int)
    X = X.drop(columns=['pdays'])
    return X

X_train = handle_pdays(X_train)

print('\nOriginal X_train columns ({}):'.format(len(X_train.columns)))
print(list(X_train.columns))

nominal_cols = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']
ordinal_cols = ['education']
education_order = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y','high.school', 'professional.course', 'university.degree', 'unknown']

ordinal_enc = OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=-1)
# compatible OneHotEncoder instantiation
try:
    nominal_enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
except TypeError:
    nominal_enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)

cat_transformer = ColumnTransformer(transformers=[('nominal', nominal_enc, nominal_cols), ('ordinal', ordinal_enc, ordinal_cols)], remainder='passthrough')

X_train_encoded = cat_transformer.fit_transform(X_train)
print('\nEncoded array shape:', X_train_encoded.shape)

# build expected column names
ohe = cat_transformer.named_transformers_['nominal']
try:
    nominal_feature_names = list(ohe.get_feature_names_out(nominal_cols))
except Exception:
    nominal_feature_names = []
    for i,col in enumerate(nominal_cols):
        cats = list(ohe.categories_[i])
        for c in cats[1:]:
            nominal_feature_names.append(f"{col}_{c}")

num_passthrough_cols = [col for col in X_train.columns if col not in ordinal_cols + nominal_cols]
all_cols_enc = nominal_feature_names + list(ordinal_cols) + num_passthrough_cols

print('\nConstructed encoded column names (count={}):'.format(len(all_cols_enc)))
print(all_cols_enc)

# create DataFrame and compare
try:
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=all_cols_enc)
    print('\nSuccessfully created DataFrame from encoded array')
except Exception as e:
    print('\nERROR creating DataFrame:', e)
    # try fallback: print shapes
    print('encoded array shape:', X_train_encoded.shape)
    print('len(all_cols_enc):', len(all_cols_enc))

# report differences
missing_originals = [c for c in X_train.columns if c not in all_cols_enc]
print('\nOriginal columns NOT present in constructed encoded columns (expected are categorical ones replaced by encodings):')
print(missing_originals)

unexpected_missing_passthrough = [c for c in num_passthrough_cols if c not in all_cols_enc]
print('\nUnexpected missing passthrough columns:')
print(unexpected_missing_passthrough)

# also check for duplicate or overlapping names
dups = [name for name in all_cols_enc if all_cols_enc.count(name) > 1]
if dups:
    print('\nDuplicate names in all_cols_enc (could cause column overwrites):')
    print(set(dups))
else:
    print('\nNo duplicate names in constructed encoded columns')

print('\nDiagnostic script finished')

