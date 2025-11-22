import os
import pandas as pd
import gradio as gr
import psycopg2
from psycopg2.extras import RealDictCursor
import re
from datetime import datetime
import time
import sys, io

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

# ====================== TRAIN FUNCTION (Streaming log) ======================
def train_real(file_obj, log_txt=""):
    if not file_obj:
        yield log_txt + "Please upload train.csv\n"
        return

    df = pd.read_csv(file_obj.name)
    total = len(df)
    log_txt += f"--- TRAINING STARTED {datetime.now():%Y-%m-%d %H:%M} ---\n"
    yield log_txt

    for i, row in df.iterrows():
        q, sql = row["question"], fix_quotes(row["sql"])
        try:
            execute_query_to_df(sql)
            vn.train(question=q, sql=sql)
            log_txt += f"[{i+1}/{total}] OK: {q}\n"
        except Exception as e:
            log_txt += f"[{i+1}/{total}] FAIL: {q} ({str(e)[:100]})\n"
        yield log_txt
        time.sleep(0.05)

    log_txt += "**TRAINING COMPLETED SUCCESSFULLY**\n"
    yield log_txt

# ====================== INTERACT FUNCTION (Generate + Run SQL + Log) ======================
def interact_generate_run(question, prev_log):
    log_txt = prev_log or ""
    sql = ""
    try:
        # Capture internal print from VN/RAG
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        sql = clean_generated_sql(vn.generate_sql(question))
        sys.stdout = old_stdout
        log_txt += mystdout.getvalue()
    except Exception as e:
        log_txt += f"[{datetime.now():%H:%M:%S}] ERROR generating SQL: {str(e)}\n"
        sql = ""

    log_txt += f"[{datetime.now():%H:%M:%S}] Q: {question}\nGenerated SQL: {sql}\n"

    # Run SQL
    if sql:
        df, status = execute_query_to_df(sql)
        if status != "PASS":
            log_txt += f"[{datetime.now():%H:%M:%S}] SQL Execution Error: {status}\n"
            df = pd.DataFrame()
        else:
            log_txt += f"[{datetime.now():%H:%M:%S}] SQL Executed Successfully, {len(df)} rows returned\n"
    else:
        df = pd.DataFrame()

    log_txt += "\n"
    return sql, df, log_txt

# ====================== UI ======================
with gr.Blocks(title="Text_To_SQL – Database Department Rental") as demo:
    gr.Markdown("# Text_To_SQL – Database Department Rental")
    gr.Markdown("**EM • EX • VES – Professional Text-to-SQL Evaluation System**")

    with gr.Tabs():
        # --------- Train Tab ---------
        with gr.Tab("Train"):
            gr.Markdown("### Train Your Model")
            train_file = gr.File(label="Upload train.csv", file_types=[".csv"])
            train_btn = gr.Button("Start Training", variant="primary")
            train_log = gr.Textbox(label="Training Log", lines=20, interactive=False)
            train_btn.click(fn=train_real, inputs=train_file, outputs=train_log)

        # --------- Interact Tab ---------
        with gr.Tab("Interact"):
            gr.Markdown("### Ask Questions in Natural Language")
            q = gr.Textbox(label="Your Question", lines=6, placeholder="e.g. Show apartments with rent > 2000")
            gen = gr.Button("Generate & Run SQL", variant="secondary")
            sql_out = gr.Textbox(label="Generated SQL", lines=10)
            out_df = gr.Dataframe(label="Query Result")
            interact_log = gr.Textbox(label="Interaction Log", lines=20, interactive=False)
            gen.click(fn=interact_generate_run, inputs=[q, interact_log], outputs=[sql_out, out_df, interact_log])

        # --------- Schema Tab ---------
        with gr.Tab("Schema"):
            gr.Markdown("### Database Schema (DDL)")
            gr.HTML(load_ddl_schema())

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True, enable_queue=True)
