import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from math import pi
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KernelDensity

MAIN_FEATURES = ["RAM","Procesorius","WIFI","Atmintis","Akumuliatorius","Operacine sistema","Tinklas","Ekranas","Gamintojas"]
PROCESEORIAI = [
    # Qualcomm Snapdragon - High End
    "snapdragon 8 gen 3", "snapdragon 8 gen 2", "snapdragon 8 gen 1", "snapdragon 888+", "snapdragon 888",
    "snapdragon 870", "snapdragon 865+", "snapdragon 865", "snapdragon 860", "snapdragon 855+", "snapdragon 855",
    "snapdragon 845", "snapdragon 835",

    # Qualcomm Snapdragon - Mid Range
    "snapdragon 782g", "snapdragon 780g", "snapdragon 778g", "snapdragon 768g",
    "snapdragon 765g", "snapdragon 765", "snapdragon 750g", "snapdragon 732g", "snapdragon 730g", "snapdragon 730",
    "snapdragon 720g", "snapdragon 712", "snapdragon 710", "snapdragon 695", "snapdragon 690",
    "snapdragon 680", "snapdragon 678", "snapdragon 675", "snapdragon 665", "snapdragon 662",
    "snapdragon 660", "snapdragon 652",

    # Qualcomm Snapdragon - Low End
    "snapdragon 480+", "snapdragon 480", "snapdragon 460", "snapdragon 450", "snapdragon 439",
    "snapdragon 435", "snapdragon 430", "snapdragon 425", "snapdragon 415", "snapdragon 400", "snapdragon",

    # MediaTek - Dimensity (High-End)
    "mediatek dimensity 9300+", "mediatek dimensity 9300", "mediatek dimensity 9200+", "mediatek dimensity 9200",
    "mediatek dimensity 9000+", "mediatek dimensity 9000", "mediatek dimensity 8300", "mediatek dimensity 8200",
    "mediatek dimensity 8100", "mediatek dimensity 8000", "mediatek dimensity 7050", "mediatek dimensity 7030",
    "mediatek dimensity 7020", "mediatek dimensity 1300", "mediatek dimensity 1200", "mediatek dimensity 1100",
    "mediatek dimensity 1000+", "mediatek dimensity 1000", "mediatek dimensity 930", "mediatek dimensity 920",
    "mediatek dimensity 900", "mediatek dimensity 820", "mediatek dimensity 810", "mediatek dimensity 800",
    "mediatek dimensity 700", "mediatek dimensity 6100+", "mediatek dimensity 6100", "mediatek dimensity 6080",
    "mediatek dimensity 6020",

    # MediaTek - Helio G (Gaming)
    "mediatek helio g99", "mediatek helio g96", "mediatek helio g95", "mediatek helio g90t", "mediatek helio g90",
    "mediatek helio g88", "mediatek helio g85", "mediatek helio g80", "mediatek helio g70",
    "mediatek helio g37", "mediatek helio g35", "mediatek helio g25",

    # MediaTek - Helio P (Mid-Range)
    "mediatek helio p95", "mediatek helio p90", "mediatek helio p65", "mediatek helio p60",
    "mediatek helio p35", "mediatek helio p22", "mediatek helio p20", "mediatek helio p10",

    # MediaTek - Helio X and A Series
    "mediatek helio x30", "mediatek helio x25", "mediatek helio x20", "mediatek helio x10",
    "mediatek helio a25", "mediatek helio a22", "mediatek mt6765", "mediatek mt6761", "mediatek mt6739","mediatek",

    # Apple
    "apple a17 pro", "apple a16 bionic", "apple a15 bionic", "apple a14 bionic", "apple a13 bionic",
    "apple a12 bionic", "apple a11 bionic", "apple a10 fusion", "apple a9", "apple a8", "apple a7", "apple a6",
    "a17 pro", "a16 bionic", "a15 bionic", "a14 bionic", "a13 bionic", "a12 bionic", "a11 bionic", "a10 fusion",
    "a9", "a8", "a7", "a6",

    # Samsung Exynos
    "exynos 1580",
    "exynos 2400", "exynos 2200", "exynos 2100", "exynos 1380", "exynos 1280", "exynos 1200",
    "exynos 1080", "exynos 990", "exynos 980", "exynos 9825", "exynos 9820", "exynos 9810",
    "exynos 8895", "exynos 8890", "exynos 850", "exynos 7904", "exynos 7885", "exynos 7880",
    "exynos 7870", "exynos 7580", "exynos",

    # Google
    "google tensor g3", "google tensor g2", "google tensor",

    # Unisoc
    "unisoc t820", "unisoc t740", "unisoc t700", "unisoc t618", "unisoc t616", "unisoc t612",
    "unisoc t610", "unisoc t606", "unisoc t310", "unisoc sc9863a", "unisoc sc9832e", "unisoc sc7731e", "unisoc",

    # HiSilicon (Huawei)
    "kirin 9000", "kirin 990", "kirin 985", "kirin 980", "kirin 970", "kirin 960", "kirin 950",
    "kirin 930", "kirin 920", "kirin 910", "kirin 820", "kirin 810", "kirin 710", "kirin",

    # Spreadtrum (Older Unisoc)
    "spreadtrum sc9830", "sc9830",
    
    # Others / Catch-all
    "mali", "powervr", "adreno", "nvidia tegra", "tegra",
    # Add any additional entries below as needed
]

OPERATING_SYSTEMS = [
    # Android-based operating systems
    "android 13", "android 12", "android 11", "android 10", "android 9", "android 8", "android 7", "android 6",
    "android 5", "android 4", "android go", "oxygenos", "one ui", "miui", "coloros", "realme ui", "funtouch os",
    "originos", "emui", "magic ui", "flyme os", "zenui", "nubia ui", "redmagic os", "joyui", "lineageos",
    "grapheneos", "calyxos", "fire os", "kaios","android",
    
    # iOS (Apple)
    "ios 17", "ios 16", "ios 15", "ios 14", "ios 13", "ios 12", "ios 11", "ios 10", "ios 9", "ios 8", "ios",
    
    # HarmonyOS (Huawei)
     "harmonyos 3", "harmonyos 2","harmonyos",
    
    # Windows-based operating systems
    "windows phone", "windows 10 mobile", "windows 8 mobile", "windows 7 mobile",
    
    # Other mobile operating systems
    "blackberry os", "blackberry 10", "sailfish os", "tizen", "ubuntu touch", "firefox os", "symbian", "bada",
    "meego", "palm os", "webos"
]
PHONE_BRANDS = [
    "samsung", "apple", "xiaomi", "redmi", "poco", "mi", "oneplus", "oppo", "vivo", "realme", 
    "honor", "huawei", "motorola", "nokia", "sony", "google", "microsoft", "lg", "htc", "lenovo", 
    "asus", "zte", "meizu", "tecno", "infinix", "sharp", "coolpad", "doogee", "ulefone", "umidigi",
    "blackview", "cat", "energizer", "alcatel", "one touch", "blu", "fairphone", "fair phone",
    "wiko", "micromax", "lava", "karbonn", "gionee", "lyf", "acer", "thomson", "toshiba", "philips"
]
def load_csv(file_path):
    return pd.read_csv(file_path)

def extract_number(value):
    # Use regex to extract numbers (integers or floats)
    match = re.search(r'\d+(\.\d+)?', value)
    if match:
        return float(match.group())  # Convert to float if a match is found
    return None  # Return None if no number is found

def parse_processor_info(text):

    text = text.lower()
    core_map = {
        r"\bocta[\-\s]*core\b": 8,
        r"\bhexa[\-\s]*core\b": 6,
        r"\bquad[\-\s]*core\b": 4,
        r"\bdual[\-\s]*core\b": 2,
        r"\btriple[\-\s]*core\b": 3,
    }

    digit_core_pattern = re.compile(r"(\d+)\s*(branduoliu(?:\w*)|cores?|core)")
    
    core_count = None

    for pattern, core_num in core_map.items():
        if re.search(pattern, text):
            core_count = core_num
            break

    if core_count is None:
        match = digit_core_pattern.search(text)
        if match:
            core_count = int(match.group(1))

    clock_pattern = re.compile(r'(\d+(?:[\.,]\d+))\s*(ghz|mhz)')

    matches = clock_pattern.findall(text)

    speeds = []
    for match in matches:

        raw_value, unit = match

        raw_value = raw_value.replace(",", ".")
        speed_value = float(raw_value)

        if unit == "mhz":
            speed_value /= 1000.0
        speeds.append(speed_value)

    return {
        "core_count": core_count,
        "speeds": speeds
    }


def parse_screen_info(text):
    text = text.lower()
    
    # Extract resolution (e.g. "1920x1080", "1080 × 2400", etc.)
    resolution_pattern = re.compile(r"(\d+)\s*[x×]\s*(\d+)")
    resolution_match = resolution_pattern.search(text)
    screen_resolution = None
    if resolution_match:
        screen_resolution = f"{resolution_match.group(1)}x{resolution_match.group(2)}"
    
    # Extract size (e.g. "6.5 inch", "16,5 cm")
    size_pattern = re.compile(r"(\d+(?:[\.,]\d+)?)\s*(?:inch|in|col|cm)")
    size_match = size_pattern.search(text)
    screen_size = None
    if size_match:
        numeric_str = size_match.group(1).replace(",", ".")
        screen_size = float(numeric_str)
        # Convert from cm to inches if needed
        if "cm" in size_match.group(0):
            screen_size /= 2.54
    
    return (screen_size, screen_resolution)

def preprocess(df):
    for column in df.columns:
        check_column = str(column).lower()

        if "ram" in check_column:

            for idx, value in enumerate(df[column]):
                value = str(value).lower()
                for ram_size in range(2,34,2):
                    if str(ram_size) in value:
                        df.loc[idx,"ram_size_feature"] = ram_size
                    elif str(ram_size * 1024)  in value:
                        df.loc[idx,"ram_size_feature"] = ram_size/1024

        elif any(keyword in check_column for keyword in ["procesorius", "process", "cpu", "dažnis","branduo","core"]):
            for idx, row_value in enumerate(df[column]):
                text_val = str(row_value).lower()
                
                parse_results = parse_processor_info(text_val)

                if parse_results["core_count"] is not None:
                    df.loc[idx, "procesorius_core_count_feature"] = parse_results["core_count"]
                
                if parse_results["speeds"]:
                    avg_speed = sum(parse_results["speeds"]) / len(parse_results["speeds"])
                    df.loc[idx, "procesorius_clock_feature"] = avg_speed

                if "procesorius" in check_column or "processor" in check_column:
                    for processor_name in PROCESEORIAI:
                        if processor_name in text_val:
                            df.loc[idx,"procesorius_feature"] = processor_name
                            break

        elif "atmint" in check_column or "storage" in check_column:
            for idx, value in enumerate(df[column]):
                value = str(value).lower()
                for storage_size in range(32,1024):
                    if str(storage_size) + " gb" in value or str(storage_size) + "gb" in value:
                        df.loc[idx,"storage_feature"] = storage_size
                        break
                for storage_size in range(1, 4):
                    if str(storage_size) + " tb" in value or str(storage_size) + "tb" in value:
                        df.loc[idx,"storage_feature"] = storage_size*1024
                        break
                if "gb" in check_column or "tb" in check_column:
                    for storage_size in range(32,1024):
                        if str(storage_size) in value:
                            df.loc[idx,"storage_feature"] = storage_size
                            break

        elif any(keyword in check_column for keyword in ["operac","android","ios","operat","system","sistem","version","versija"]):
            for idx, value in enumerate(df[column]):
                value = str(value).lower()
                for os in OPERATING_SYSTEMS:
                    if os in value:
                        df.loc[idx, "operating_system_feature"] = os
                        break
        elif any(keyword in check_column for keyword in ["tinklas","tinklo","juost","band","ryšys"]):
            for idx, value in enumerate(df[column]):
                value = str(value).lower()
                for band in ["5g","4g","3g","2g"]:
                    if band in value:
                        df.loc[idx, "band_feature"] = band
                        break
        elif any(keyword in check_column for keyword in ["ekran","screen","raiška", "resolution","rezoliucija"]):
            for idx, value in enumerate(df[column]):
                size, resolution = parse_screen_info(str(value))
                if size:
                    df.loc[idx, "screen_size_feature"] = size
                if resolution:
                    df.loc[idx, "screen_resolution_feature"] = resolution

        elif any(keyword in check_column for keyword in ["gamintojas","brand","manufactur","title","seller"]):
            for idx, value in enumerate(df[column]):
                value_lower = str(value).lower()
                for brand in PHONE_BRANDS:
                    if brand in value_lower:
                        df.loc[idx, "brand_feature"] = brand
                        break
        

        
    return df



def analyse_feature_instance(df):

    columns = df.columns
    column_list = []
    instance_count = []

    for col in columns:
        non_nan_count = df[col].notnull().sum()
        if non_nan_count == 0:
            columns.remove(col)
        else:
            instance_count.append(int(non_nan_count))
            column_list.append(col)



    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts = ax.pie(instance_count,labels=column_list)
    ax.legend(wedges, column_list,loc="upper right")
    plt.savefig("instance_pie.png")

    median_instance_count = (np.median(instance_count), columns[instance_count.index(int(np.median(instance_count)))])
    smallest_instance_count = np.min(instance_count)
    largest_instance_count = np.max(instance_count)
    print(median_instance_count)


    
def screen_resolution_preprocess(value: str):
    """
    Processes screen resolution strings in the format 'width x height'
    and returns the product of width and height. Handles missing or invalid values.
    """
    if not isinstance(value, str) or "x" not in value.lower():
        # Return None or a default value (e.g., 0) for invalid or missing values
        return None

    try:
        # Split the resolution string and calculate the product
        width, height = value.lower().split("x")
        return int(width.strip()) * int(height.strip())
    except (ValueError, AttributeError):
        # Handle cases where the split or conversion fails
        return None

def plot_price(price_list):
    plt.figure(figsize=(10,10))
    plt.scatter(price_list, list(range(price_list.shape[0])))
    plt.xlabel("price")
    plt.ylabel("item")
    plt.title("Price Distribution")

    plt.savefig("Price distribution scatterplot")


df = load_csv("varle_product_info.csv")
df_processed = preprocess(df)
columns_to_drop = [col for col in df_processed.columns if "feature" not in col.lower() and "price" not in col.lower()]
df_processed.drop(columns=columns_to_drop, inplace=True)
df_processed.drop(columns=["procesorius_core_count_feature", "procesorius_clock_feature","screen_size_feature"],inplace=True)
df_processed["screen_resolution_feature"] = df_processed["screen_resolution_feature"].apply(screen_resolution_preprocess)
df_processed = df_processed[df_processed['price'] <= 800]
encoder = OneHotEncoder(handle_unknown="ignore")
scaler = RobustScaler()
standard_scaler = StandardScaler()
#df_processed["price_scaled"] = scaler.fit_transform(df_processed[["price"]])

categorical_columns = ["procesorius_feature", "operating_system_feature", "band_feature","brand_feature","ram_size_feature","storage_feature"]

numerical_columns = [
    "screen_resolution_feature"
]

for column in categorical_columns:
    df_processed[column] = df_processed[column].fillna("Unknown").astype(str)
    
    encoded_array = encoder.fit_transform(df_processed[[column]]).toarray()
    feature_names = encoder.get_feature_names_out([column])
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_processed.index)

    df_processed = pd.concat([df_processed, encoded_df], axis=1)
    df_processed.drop(columns=[column], inplace=True)

for column in numerical_columns:
    df_processed[column] = df_processed[column].fillna(df_processed[column].median())
    df_processed[column] = standard_scaler.fit_transform(df_processed[[column]])


linear_model = LinearRegression()
X = df_processed.drop(columns=["price"])
y = df_processed["price"]
y_log = np.log1p(y)

# Estimate density of price distribution
kde = KernelDensity(bandwidth=50.0).fit(y.values.reshape(-1, 1))
density = np.exp(kde.score_samples(y.values.reshape(-1, 1)))

# Calculate weights (inverse of density)
weights = 1.0 / (density + 1e-10)  # Add small constant to avoid division by zero
weights = weights / np.mean(weights)  # Normalize weights to mean=1

# Create a Series with same index as original data
weights_series = pd.Series(weights.flatten(), index=df_processed.index)

# Then use the Series for indexing with train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear_model.fit(X_train, y_train, sample_weight=weights_series.loc[X_train.index])
y_pred = linear_model.predict(X_test)

# Train model on transformed data
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
linear_model.fit(X_train, y_train_log, sample_weight=weights_series.loc[X_train.index])

# Predict and transform back
y_pred_log = linear_model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # inverse of log1p

# Evaluate on original scale
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.savefig("ActualvsPredicted_Prices_Linear_regression.png")

param_grid = {
    'n_estimators': [20, 50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [1, 2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)

best_rf_model = grid_search.best_estimator_

best_rf_model.fit(X_train, y_train)

y_pred_rf = best_rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Random Forest R² Score: {r2_rf}")

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.savefig("ActualvsPredicted_Prices_Random_Forest.png")


bins = [0, 150, 300, 450, 600, 750, float('inf')] 
labels = ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5", "Tier 6"]
df_processed["price_category"] = pd.cut(df_processed["price"], bins=bins, labels=labels)
df_price = df_processed.filter(["price"], axis=1)
df_processed.drop(columns=["price"], inplace=True)

price_tier_distribution = df_processed["price_category"].value_counts()
print("Original class distribution:")
print(price_tier_distribution)
plt.figure(figsize=(10, 6))
price_tier_distribution.plot(kind='bar')
plt.title('Price Tier Distribution')
plt.xlabel('Price Tier')
plt.ylabel('Count')
plt.savefig("price_tier_distribution.png")

X = df_processed.drop(columns=["price_category"])  
y = df_processed["price_category"]
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("Class distribution after SMOTE:")
print(pd.Series(y_balanced).value_counts())

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_balanced)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_encoded, test_size=0.2, random_state=42)


grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

best_rf_classifier = grid_search.best_estimator_
joblib.dump(best_rf_classifier, "random_forest_classifier.pkl")
joblib.dump(standard_scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")
y_pred = best_rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("Confusion_matrix_RFC.png")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 10, 100]
}

xgb_model = XGBClassifier(random_state=42)

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)


best_xgb_model = grid_search.best_estimator_

y_pred_xgb = best_xgb_model.predict(X_test)

y_pred_xgb_decoded = label_encoder.inverse_transform(y_pred_xgb)

conf_matrix = confusion_matrix(y_test, y_pred_xgb)
print("\nConfusion Matrix:")
print(conf_matrix)

class_labels = label_encoder.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("xgbc/Confusion_matrix_XGBC.png")

train_sizes, train_scores, test_scores = learning_curve(
    best_xgb_model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
plt.plot(train_sizes, test_mean, label="Cross-Validation Score", color="orange")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orange")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.savefig("xgbc/learning_curve_XGBC.png")


accuracy_xgb = accuracy_score(label_encoder.inverse_transform(y_test), y_pred_xgb_decoded)
print(f"XGBoost Accuracy: {accuracy_xgb}")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4] 
}


svc_model = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svc_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

best_svc_model = grid_search.best_estimator_
y_pred_svc = best_svc_model.predict(X_test)

accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"SVC Accuracy: {accuracy_svc}")

conf_matrix_svc = confusion_matrix(y_test, y_pred_svc)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svc, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("svc/Confusion_matrix_SVC.png")

train_sizes, train_scores, test_scores = learning_curve(
    best_svc_model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
plt.plot(train_sizes, test_mean, label="Cross-Validation Score", color="orange")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orange")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (SVC)")
plt.legend()
plt.savefig("svc/learning_curve_SVC.png")

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization type
    'solver': ['liblinear', 'saga'],  # Solvers that support L1 and elasticnet
    'max_iter': [100, 500, 1000]  # Maximum iterations
}

logreg_model = LogisticRegression(random_state=42)

grid_search = GridSearchCV(estimator=logreg_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

best_logreg_model = grid_search.best_estimator_

y_pred_logreg = best_logreg_model.predict(X_test)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg}")

conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_logreg, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("logreg/Confusion_matrix_logreg.png")

train_sizes, train_scores, test_scores = learning_curve(
    best_logreg_model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
plt.plot(train_sizes, test_mean, label="Cross-Validation Score", color="orange")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orange")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (Logistic Regression)")
plt.legend()
plt.savefig("logreg/learning_curve_logreg.png")

# param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [1e-5, 1e-4, 1e-3],
#     'learning_rate': ['constant', 'adaptive'],
#     'max_iter': [200, 500, 1000]
# }


# mlp_model = MLPClassifier(random_state=42)

# grid_search = GridSearchCV(estimator=mlp_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)

# grid_search.fit(X_train, y_train)

# best_mlp_model = grid_search.best_estimator_
# y_pred_mlp = best_mlp_model.predict(X_test)

# accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
# print(f"Neural Network Accuracy: {accuracy_mlp}")

# conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)

# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix_mlp, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.savefig("mlp/Confusion_matrix_mlp.png")

# train_sizes, train_scores, test_scores = learning_curve(
#     best_mlp_model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1
# )

# train_mean = train_scores.mean(axis=1)
# train_std = train_scores.std(axis=1)
# test_mean = test_scores.mean(axis=1)
# test_std = test_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
plt.plot(train_sizes, test_mean, label="Cross-Validation Score", color="orange")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orange")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (mlp)")
plt.legend()
plt.savefig("mlp/learning_curve_mlp.png")

# Logistic Regression
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg, average='weighted')
recall_logreg = recall_score(y_test, y_pred_logreg, average='weighted')
f1_logreg = f1_score(y_test, y_pred_logreg, average='weighted')

# SVC
accuracy_svc = accuracy_score(y_test, y_pred_svc)
precision_svc = precision_score(y_test, y_pred_svc, average='weighted')
recall_svc = recall_score(y_test, y_pred_svc, average='weighted')
f1_svc = f1_score(y_test, y_pred_svc, average='weighted')

# XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb, average='weighted')
recall_xgb = recall_score(y_test, y_pred_xgb, average='weighted')
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')

# MLPClassifier
# accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
# precision_mlp = precision_score(y_test, y_pred_mlp, average='weighted')
# recall_mlp = recall_score(y_test, y_pred_mlp, average='weighted')
# f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')

metrics = {
    'Logistic Regression': [accuracy_logreg, precision_logreg, recall_logreg, f1_logreg],
    'SVC': [accuracy_svc, precision_svc, recall_svc, f1_svc],
    'XGBoost': [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb],
    # 'MLPClassifier': [accuracy_mlp, precision_mlp, recall_mlp, f1_mlp]
}

# Create a DataFrame
metrics_df = pd.DataFrame(metrics, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Prepare data for radar chart
categories = list(metrics_df.index)
values_logreg = metrics_df['Logistic Regression'].tolist()
values_svc = metrics_df['SVC'].tolist()
values_xgb = metrics_df['XGBoost'].tolist()
# values_mlp = metrics_df["MLPClassifier"].tolist()

values_logreg += values_logreg[:1]
values_svc += values_svc[:1]
values_xgb += values_xgb[:1]
# values_mlp += values_mlp[:1]
categories += categories[:1]  

angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values_logreg, label='Logistic Regression', color='blue')
ax.fill(angles, values_logreg, color='blue', alpha=0.1)
ax.plot(angles, values_svc, label='SVC', color='orange')
ax.fill(angles, values_svc, color='orange', alpha=0.1)
ax.plot(angles, values_xgb, label='XGBoost', color='green')
ax.fill(angles, values_xgb, color='green', alpha=0.1)
# ax.plot(angles, values_mlp, label='MLPClassifier', color='red')
# ax.fill(angles, values_mlp, color='red', alpha=0.1)
ax.set_xticks(angles)
ax.set_xticklabels(categories)
plt.title('Model Performance Comparison')
plt.legend()
plt.savefig("Model_Comparison.png")


