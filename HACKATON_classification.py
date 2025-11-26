# necessary libraries

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
# Using RandomForest for better multi-class performance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv(r"/content/sample_data/homelessness_3class_balanced.csv")
#df.head()

X = df.drop(columns=["risk_label","household_id"])
y = df["risk_label"]   # this has 0, 1, 2

# df.info()

numeric_features = ['arrears_amount', 'months_in_arrears', 'uc_change', 'ae_visits_6m',
       'eviction_notice', 'domestic_abuse_flag', 'mental_health_flag',
       'social_care_flag', 'previous_homelessness']

categorical_features = ['employment_status']

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25, 
    random_state=42,
    stratify=y  # IMPORTANT for balanced classes
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(cm)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_proba = clf.predict_proba(X_test)

roc_auc_score(y_test_bin, y_proba, average="macro")
# AUC-ROC measures how well the model separates the classes 1.0 perfect swparation, 0.5 random guessing and <0.5 worse than guessing

# model investigation
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# get feature importance for RandomForest
import numpy as np
ohe = clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)

# Numeric features come as-is
all_feature_names = np.concatenate([numeric_features, cat_feature_names])
all_feature_names

# extracting RandomForest feature importance
feature_importance = clf.named_steps["model"].feature_importances_

# DataFrame for readability
import pandas as pd

importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Feature Importance (RandomForest):")
print(importance_df)

# feature importance dashboard
import pandas as pd
import matplotlib.pyplot as plt

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("RandomForest Feature Importance")
plt.tight_layout()
plt.show()

# heatmap for feature importance (top 10 features)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
plt.title("Top 10 Most Important Features (RandomForest)")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# getting my data back
## risk probability band
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. Load your dataset
# ---------------------------------------------------------
df = pd.read_csv("/content/sample_data/homelessness_3class_balanced.csv")  # <- Replace with your path

# ---------------------------------------------------------
# 2. Get model probabilities
# ---------------------------------------------------------
# Get actual probabilities from trained model
proba = clf.predict_proba(df.drop(columns=["risk_label","household_id"]))

df["low_risk_prob"] = proba[:, 0]
df["medium_risk_prob"] = proba[:, 1]
df["high_risk_prob"] = proba[:, 2]

# ---------------------------------------------------------
# 3. Build Risk Bands
# ---------------------------------------------------------
def assign_risk_band(row):
    probs = {
        "Low Risk": row["low_risk_prob"],
        "Medium Risk": row["medium_risk_prob"],
        "High Risk": row["high_risk_prob"]
    }
    return max(probs, key=probs.get)

df["risk_band"] = df.apply(assign_risk_band, axis=1)

# See first rows
df.head()


# dashboard
import pandas as pd

# ---------------------------------------------------------
# 1. LOAD YOUR DATA
# ---------------------------------------------------------
#df = pd.read_csv("PATH_TO_YOUR_FILE.csv")   # <-- replace this

# Ensure JSON serializable types
df = df.copy()

# Convert the dataframe to JSON for embedding
json_data = df.to_json(orient="records")

# ---------------------------------------------------------
# 2. WRITE HTML + JS DASHBOARD
# ---------------------------------------------------------
html_path = "homelessness_dashboard_randomforest.html"

html = f"""
<html>
<head>
<title>Homelessness Risk Dashboard - RandomForest Model</title>

<!-- CHART.JS FOR PIE + BAR CHARTS -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body {{
    font-family: Arial;
    margin: 25px;
    background: #f4f6f9;
}}

h1 {{
    color: #003366;
}}

.card {{
    padding: 20px;
    background: white;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}}

.risk-button {{
    display: inline-block;
    width: 23%;
    margin-right: 2%;
    padding: 15px;
    background: #eef5ff;
    border-radius: 10px;
    text-align: center;
    cursor: pointer;
    border: 2px solid transparent;
}}

.risk-button:hover {{
    background: #dceaff;
}}

.risk-button.active {{
    border: 2px solid #003366;
    background: #d0e1ff;
}}

input {{
    width: 100%;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 15px;
    border: 1px solid #ccc;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    background: #003366;
    color: white;
    padding: 10px;
}}

td {{
    padding: 8px;
    border-bottom: 1px solid #ddd;
    cursor: pointer;
}}

tr:hover {{
    background: #eef2f7;
}}

.pagination {{
    text-align: center;
    margin-top: 15px;
}}

button {{
    padding: 8px 12px;
    margin: 5px;
    border-radius: 6px;
    border: none;
    cursor: pointer;
}}

.modal {{
    display: none;
    position: fixed;
    z-index: 10;
    padding-top: 80px;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
}}

.modal-content {{
    background: white;
    margin: auto;
    padding: 20px;
    width: 50%;
    border-radius: 12px;
}}

.close {{
    float: right;
    font-size: 28px;
    cursor: pointer;
}}
</style>

</head>
<body>

<h1>🏠 Homelessness Risk Dashboard - RandomForest Model</h1>

<!-- ===================================================== -->
<!-- RISK BAND BUTTONS -->
<!-- ===================================================== -->

<div class="card">
    <h2>Click a Risk Band</h2>

    <div id="btnAll" class="risk-button active">
        <h2 id="allCount"></h2><p>All Households</p>
    </div>

    <div id="btnLow" class="risk-button">
        <h2 id="lowCount"></h2><p>Low Risk</p>
    </div>

    <div id="btnMed" class="risk-button">
        <h2 id="medCount"></h2><p>Medium Risk</p>
    </div>

    <div id="btnHigh" class="risk-button">
        <h2 id="highCount"></h2><p>High Risk</p>
    </div>
</div>

<!-- ===================================================== -->
<!-- CHARTS -->
<!-- ===================================================== -->

<div class="card">
    <h2>Risk Distribution</h2>
    <canvas id="riskPieChart"></canvas>
</div>

<div class="card">
    <h2>Average Arrears by Risk Band</h2>
    <canvas id="arrearsBarChart"></canvas>
</div>

<!-- ===================================================== -->
<!-- SEARCH + TABLE -->
<!-- ===================================================== -->

<div class="card">
    <h2 id="tableTitle">All Households</h2>

    <input
        type="text"
        id="searchInput"
        placeholder="Search households..."
    >

    <table id="riskTable">
        <tr>
            <th>Household ID</th>
            <th>High Risk Probability</th>
            <th>Risk Band</th>
        </tr>
    </table>

    <div class="pagination">
        <button onclick="changePage(-1)">Previous</button>
        <span id="pageLabel"></span>
        <button onclick="changePage(1)">Next</button>
    </div>
</div>

<!-- ===================================================== -->
<!-- DRILL-DOWN MODAL -->
<!-- ===================================================== -->

<div id="detailModal" class="modal">
  <div class="modal-content">
    <span class="close" onclick="closeModal()">&times;</span>
    <h2>Household Details</h2>
    <pre id="detailContent"></pre>
  </div>
</div>

<!-- ===================================================== -->
<!-- JAVASCRIPT LOGIC -->
<!-- ===================================================== -->

<script>
// EMBEDDED DATA
let data = {json_data};

// GLOBAL STATE
let currentPage = 1;
let rowsPerPage = 20;
let currentFilter = "All";

// COUNT SUMMARY
function populateCounts() {{
    document.getElementById("lowCount").innerText = data.filter(x => x.risk_band === "Low Risk").length;
    document.getElementById("medCount").innerText = data.filter(x => x.risk_band === "Medium Risk").length;
    document.getElementById("highCount").innerText = data.filter(x => x.risk_band === "High Risk").length;
    document.getElementById("allCount").innerText = data.length;
}}

// SEARCH FILTER
function applySearchFilter(rows) {{
    let search = document.getElementById("searchInput").value.toLowerCase();
    if (!search) return rows;

    return rows.filter(row =>
        row.household_id.toString().toLowerCase().includes(search) ||
        row.risk_band.toLowerCase().includes(search)
    );
}}

// LOAD TABLE WITH FILTERS
function loadTable(filter) {{
    currentFilter = filter;
    let table = document.getElementById("riskTable");

    table.innerHTML = `
        <tr>
            <th>Household ID</th>
            <th>High Risk Probability</th>
            <th>Risk Band</th>
        </tr>
    `;

    let filtered =
        filter === "All" ? data :
        data.filter(x => x.risk_band === filter);

    filtered = applySearchFilter(filtered);

    // PAGINATION
    let start = (currentPage - 1) * rowsPerPage;
    let end = start + rowsPerPage;
    let paginated = filtered.slice(start, end);

    document.getElementById("pageLabel").innerText =
        "Page " + currentPage + " of " + Math.ceil(filtered.length / rowsPerPage);

    document.getElementById("tableTitle").innerText = filter + " Households";

    paginated.forEach(row => {{
        let tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${{row.household_id}}</td>
            <td>${{Number(row.high_risk_prob).toFixed(3)}}</td>
            <td>${{row.risk_band}}</td>
        `;
        tr.onclick = () => showDetails(row);
        table.appendChild(tr);
    }});
}}

// PAGINATION CONTROLS
function changePage(direction) {{
    currentPage += direction;
    if (currentPage < 1) currentPage = 1;
    loadTable(currentFilter);
}}

// DRILL-DOWN MODAL
function showDetails(row) {{
    document.getElementById("detailContent").innerText =
        JSON.stringify(row, null, 2);
    document.getElementById("detailModal").style.display = "block";
}}
function closeModal() {{
    document.getElementById("detailModal").style.display = "none";
}}

// RISK DISTRIBUTION PIE CHART
function renderCharts() {{
    let low = data.filter(x => x.risk_band === "Low Risk").length;
    let med = data.filter(x => x.risk_band === "Medium Risk").length;
    let high = data.filter(x => x.risk_band === "High Risk").length;

    new Chart(document.getElementById("riskPieChart"), {{
        type: "pie",
        data: {{
            labels: ["Low", "Medium", "High"],
            datasets: [{{
                data: [low, med, high],
                backgroundColor: ["#4caf50", "#ff9800", "#f44336"]
            }}]
        }}
    }});

    // AVERAGE ARREARS PER BAND
    function avg(arr) {{ return arr.reduce((a,b) => a+b, 0) / arr.length; }}

    let avgLow = avg(data.filter(x => x.risk_band === "Low Risk").map(x => x.arrears_amount));
    let avgMed = avg(data.filter(x => x.risk_band === "Medium Risk").map(x => x.arrears_amount));
    let avgHigh = avg(data.filter(x => x.risk_band === "High Risk").map(x => x.arrears_amount));

    new Chart(document.getElementById("arrearsBarChart"), {{
        type: "bar",
        data: {{
            labels: ["Low", "Medium", "High"],
            datasets: [{{
                data: [avgLow, avgMed, avgHigh],
                backgroundColor: ["#4caf50", "#ff9800", "#f44336"]
            }}]
        }}
    }});
}}

populateCounts();
loadTable("All");
renderCharts();

document.getElementById("btnAll").onclick = () => {{ setActive("btnAll"); loadTable("All"); }};
document.getElementById("btnLow").onclick = () => {{ setActive("btnLow"); loadTable("Low Risk"); }};
document.getElementById("btnMed").onclick = () => {{ setActive("btnMed"); loadTable("Medium Risk"); }};
document.getElementById("btnHigh").onclick = () => {{ setActive("btnHigh"); loadTable("High Risk"); }};

function setActive(btnId) {{
    document.querySelectorAll(".risk-button").forEach(btn => btn.classList.remove("active"));
    document.getElementById(btnId).classList.add("active");
}}

</script>

</body>
</html>
"""

with open(html_path, "w") as f:
    f.write(html)

print("RandomForest Dashboard created:", html_path)