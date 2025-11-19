import os
import pandas as pd
import gradio as gr
import psycopg2
from psycopg2.extras import RealDictCursor
import re
from datetime import datetime
import plotly.graph_objects as go

pd.set_option('display.max_rows', None)
os.environ["GRADIO_TELEMETRY_ENABLED"] = "0"

from vanna.chromadb import ChromaDB_VectorStore
from vanna.ollama import Ollama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ====================== PATHS & CONFIG ======================
CHROMA_PATH = "chroma_db"
LOG_DIR = "logs"
STATIC_DIR = "static"
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

DEFAULT_DIAGRAM_PATH = os.path.join(STATIC_DIR, "static_diagram.png")

TRAIN_LOG_FILE = os.path.join(LOG_DIR, "train.log")
SYSTEM_PROMPT_FILE = "system_prompt.txt"
DDL_FILE = "posgresdb/init.sql"
REAL_DB_CONFIG = {
    "host": "localhost",
    "dbname": "apartment_rentals",
    "user": "admin",
    "password": "adminpass",
    "port": "5432"
}

# ====================== DATABASE FUNCTIONS ======================
def clean_generated_sql(sql: str) -> str:
    return re.sub(r'\s+', ' ', sql.strip().replace('\"', '"')).strip()

def fix_quotes(sql: str) -> str:
    return re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', lambda m: f"'{m.group(1)}'", sql)

def execute_query_to_df(sql: str):
    conn = psycopg2.connect(**REAL_DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        df = pd.DataFrame([dict(row) for row in rows]) if rows else pd.DataFrame()
        return df, "PASS"
    except Exception as e:
        return pd.DataFrame(), f"FAIL: {str(e)[:500]}"
    finally:
        cursor.close()
        conn.close()

def analyze_query(sql: str):
    conn = psycopg2.connect(**REAL_DB_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
        plan = cursor.fetchone()[0][0]['Plan']
        total = plan.get("Actual Total Time", 0)
        return {"time_ms": round(total, 5)}
    except:
        return {"time_ms": 9999.99999}
    finally:
        cursor.close()
        conn.rollback()
        conn.close()

# ====================== SCHEMA ======================
def load_ddl_schema():
    try:
        with open(DDL_FILE, "r", encoding="utf-8") as f:
            raw_ddl = f.read()
        lines = [line.rstrip() for line in raw_ddl.split("\n") if line.strip() and not line.strip().startswith("--")]
        cleaned_ddl = "\n".join(lines)
        return f"""
        <div style="background:#1e293b;color:#e2e8f0;padding:24px;border-radius:16px;font-family:'Fira Code',monospace;font-size:14px;line-height:1.6;margin:20px 0;border:1px solid #475569;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
            <h3 style="color:#60a5fa;margin:0 0 16px 0;font-size:20px;">Database Schema – apartment_rentals</h3>
            <pre style="background:#0f172a;padding:20px;border-radius:12px;overflow-x:auto;margin:0;white-space:pre;color:#cbd5e1;border:1px solid #334155;">
<code style="color:inherit;">{cleaned_ddl}</code></pre>
        </div>
        """
    except:
        return "<div style='background:#7f1d1d;color:white;padding:20px;border-radius:12px;text-align:center;'><h3>Schema Not Found</h3><p><code>posgresdb/init.sql</code> missing</p></div>"

# ====================== VANNA SETUP ======================
class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['path'] = CHROMA_PATH
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={
    'model': 'llama3.1:8b',
    'embedding_function': SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
})

def train_schema_from_ddl():
    if not os.path.isdir(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        try:
            with open(DDL_FILE) as f:
                for stmt in [s.strip() for s in f.read().split(";") if s.strip().upper().startswith("CREATE TABLE")]:
                    clean = stmt.split("--")[0].strip()
                    if clean: vn.train(ddl=clean)
        except: pass
train_schema_from_ddl()

current_prompt = open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8").read().strip() if os.path.exists(SYSTEM_PROMPT_FILE) else "You are a PostgreSQL expert. Generate only valid SQL. Use single quotes and aliases."
vn.system_message(current_prompt)

def save_system_prompt(txt):
    txt = txt.strip()
    open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8").write(txt)
    vn.system_message(txt)
    return "System prompt saved and applied!"

# ====================== TRAIN ======================
def train_real(file_obj):
    if not file_obj:
        return "Please upload train.csv"
    df = pd.read_csv(file_obj.name)
    log = f"--- TRAINING STARTED {datetime.now():%Y-%m-%d %H:%M} ---\n"
    open(TRAIN_LOG_FILE, "w", encoding="utf-8").write(log)
    prog = gr.Progress()
    for i, row in df.iterrows():
        prog(i / len(df))
        q, sql = row["question"], fix_quotes(row["sql"])
        try:
            execute_query_to_df(sql)[0]
            vn.train(question=q, sql=sql)
        except: pass
    open(TRAIN_LOG_FILE, "a", encoding="utf-8").write("**TRAINING COMPLETED SUCCESSFULLY**\n")
    return open(TRAIN_LOG_FILE).read()

# ====================== TEST – FINAL WITH YOUR COLUMN NAMES ======================
def test_real(file_obj):
    if not file_obj:
        return "Please upload test.csv", None, None, None

    df = pd.read_csv(file_obj.name)
    results = []
    correct_em = correct_ex = 0
    semantic_count = 0
    semantic_gold_sum = semantic_gen_sum = 0.0
    total = len(df)

    for i, row in df.iterrows():
        no = i + 1
        q = row["question"]
        gold_sql = fix_quotes(row["sql"]).strip()

        gen_sql = "GENERATION FAILED"
        em_status = "GEN FAIL"
        try:
            gen_sql = clean_generated_sql(vn.generate_sql(q))
            em_status = "PASS" if gen_sql.lower() == gold_sql.lower() else "FAIL"
            if em_status == "PASS": correct_em += 1
        except Exception as e:
            gen_sql = f"GEN ERROR: {str(e)[:100]}..."

        gold_df, _ = execute_query_to_df(gold_sql)
        gen_df, gen_status = execute_query_to_df(gen_sql) if "ERROR" not in gen_sql else (pd.DataFrame(), "FAIL")
        ex_status = "PASS" if gold_df.equals(gen_df) else "FAIL"
        if ex_status == "PASS": correct_ex += 1

        gold_time = analyze_query(gold_sql)["time_ms"]
        gen_time = analyze_query(gen_sql)["time_ms"] if gen_status == "PASS" and "GEN ERROR" not in gen_sql else 9999.99999

        if gen_status == "PASS":
            diff = gold_time - gen_time
            ves = "Same" if abs(diff) < 0.00001 else ("Generated" if diff > 0 else "Gold")
        else:
            ves = "Failed"

        if ex_status == "PASS" and em_status == "FAIL" and gen_status == "PASS":
            semantic_count += 1
            semantic_gold_sum += gold_time
            semantic_gen_sum += gen_time

        results.append({
            "No.": no,
            "Question": q[:90] + "..." if len(q) > 90 else q,
            "Gold Query": gold_sql,
            "Generated Query": gen_sql,
            "Exact Matching (EM)": em_status,
            "Execution Accuracy (EX)": ex_status,
            "Gold Time": f"{gold_time:.5f}",
            "Gen Time": "TIMEOUT" if gen_time >= 9999 else f"{gen_time:.5f}",
            "Valid Efficiency Score (VES)": ves
        })

    em_score = correct_em / total * 100
    ex_score = correct_ex / total * 100
    semantic_pct = semantic_count / total * 100 if total > 0 else 0
    avg_gold = semantic_gold_sum / semantic_count if semantic_count else 0
    avg_gen = semantic_gen_sum / semantic_count if semantic_count else 0
    speed_diff = avg_gold - avg_gen

    summary_html = f"""
    <div style="background:#1e293b;color:white;padding:30px;border-radius:16px;text-align:center;margin:20px 0;font-size:21px;">
        <h2>Test Results – {datetime.now():%Y-%m-%d %H:%M}</h2>
        <b>Exact Matching (EM):</b> {correct_em}/{total} ({em_score:.2f}%) | 
        <b>Execution Accuracy (EX):</b> {correct_ex}/{total} ({ex_score:.2f}%)<br><br>
        <b style="font-size:28px;color:#fbbf24;">Different SQL with the same Execution Result:</b> {semantic_count} ({semantic_pct:.2f}%)<br>
        <b style="font-size:22px;color:#a8ffbc;">→ Generated faster by {speed_diff:+.5f} ms on average</b>
    </div>
    """

    fig1 = go.Figure(data=[
        go.Bar(name="EM", x=["Score"], y=[em_score], marker_color="#10b981", text=f"{em_score:.2f}%", textposition="outside"),
        go.Bar(name="EX", x=["Score"], y=[ex_score], marker_color="#f59e0b", text=f"{ex_score:.2f}%", textposition="outside")
    ]).update_layout(title="EM vs EX", height=420, yaxis_range=[0,100], template="plotly_dark")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Gold Query", x=["Time (ms)"], y=[avg_gold], marker_color="#ef4444", text=f"{avg_gold:.5f}"))
    fig2.add_trace(go.Bar(name="Generated Query", x=["Time (ms)"], y=[avg_gen], marker_color="#10b981", text=f"{avg_gen:.5f}"))
    fig2.update_layout(
        title=f"Valid Efficiency Score (VES) – Speed on Semantic Different SQL with the same Execution Result ({semantic_count} queries)",
        barmode="group", height=520, template="plotly_dark", font_size=16
    )
    if semantic_count == 0:
        fig2.add_annotation(text="No Different SQL with the same Execution Result found", x=0.5, y=0.5, xref="paper", yref="paper", font_size=22, showarrow=False)

    df_res = pd.DataFrame(results)
    def highlight_ves(val):
        color = {"Generated": "#10b981", "Gold": "#ef4444", "Same": "#94a3b8", "Failed": "#ef4444"}.get(val, "")
        return f"color:{color};font-weight:bold" if color else ""
    styled = df_res.style.applymap(highlight_ves, subset=["Valid Efficiency Score (VES)"])

    return summary_html, fig1, fig2, styled

# ====================== UI – FINAL MASTERPIECE ======================
with gr.Blocks(theme=gr.themes.Soft(), title="Text_To_SQL – Database Department Rental") as demo:
    gr.Markdown("# Text_To_SQL – Database Department Rental")
    gr.Markdown("**EM • EX • VES – Professional Text-to-SQL Evaluation System**")

    with gr.Tabs():
        with gr.Tab("Test"):
            gr.Markdown("### Upload test.csv → Run Full Evaluation")
            test_file = gr.File(label="Upload test.csv", file_types=[".csv"])
            test_btn = gr.Button("Run Evaluation", variant="primary", size="lg")
            summary = gr.HTML()
            p1 = gr.Plot(); p2 = gr.Plot()
            table = gr.Dataframe(wrap=True, max_height=10000)
            test_btn.click(test_real, test_file, [summary, p1, p2, table])

        with gr.Tab("Train"):
            gr.Markdown("### Train Your Model")
            train_file = gr.File(label="Upload train.csv", file_types=[".csv"])
            train_btn = gr.Button("Start Training", variant="primary")
            train_log = gr.Textbox(label="Training Log", lines=12, value="Ready to train...")
            train_btn.click(train_real, train_file, train_log)

        with gr.Tab("Interact"):
            gr.Markdown("### Ask Questions in Natural Language")
            q = gr.Textbox(label="Your Question", lines=6, placeholder="e.g. Show apartments with rent > 2000")
            gen = gr.Button("Generate SQL", variant="secondary")
            sql = gr.Code(label="Generated SQL", language="sql", lines=10)
            run = gr.Button("Execute Query", variant="primary")
            out = gr.Dataframe(label="Result")
            gen.click(lambda x: clean_generated_sql(vn.generate_sql(x)), q, sql)
            run.click(lambda s: execute_query_to_df(s)[0], sql, out)

        with gr.Tab("Schema"):
            gr.Markdown("### Database Schema (DDL)")
            gr.HTML(load_ddl_schema())

        with gr.Tab("Database Diagram"):
            gr.Markdown("### Database ER Diagram")
            gr.Markdown("_Default: `static/static_diagram.png` • Upload to replace_")
            default_img = DEFAULT_DIAGRAM_PATH if os.path.exists(DEFAULT_DIAGRAM_PATH) else None
            upload = gr.File(label="Upload New Diagram (PNG/JPG/WEBP)", file_types=["image"])
            display = gr.Image(value=default_img, label="Current Database Diagram", height=800, interactive=False)
            upload.change(fn=lambda f: f.name if f else default_img, inputs=upload, outputs=display)

        with gr.Tab("Prompt"):
            gr.Markdown("### Customize System Prompt")
            p = gr.Textbox(value=current_prompt, lines=15, label="System Prompt")
            gr.Button("Save & Apply", variant="primary").click(save_system_prompt, p, gr.Markdown("Prompt saved and applied!"))

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True)