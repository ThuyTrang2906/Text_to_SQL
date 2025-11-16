# text_to_SQL.py – REAL POSTGRESQL + WRAPPED TEXT + SYSTEM PROMPT + ERROR TRUNCATION
import os
import pandas as pd
import gradio as gr
import psycopg2
import time
import re
from datetime import datetime

from vanna.chromadb import ChromaDB_VectorStore
from vanna.ollama import Ollama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ====================== PERSISTENCE ======================
CHROMA_PATH = "chroma_db"
LOG_DIR = "logs"
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TRAIN_LOG_FILE = os.path.join(LOG_DIR, "train.log")
TEST_LOG_FILE  = os.path.join(LOG_DIR, "test.log")
SYSTEM_PROMPT_FILE = "system_prompt.txt"

def load_log(fp):
    return open(fp, "r", encoding="utf-8").read() if os.path.exists(fp) else ""

def save_log(fp, txt):
    open(fp, "w", encoding="utf-8").write(txt)

# ====================== CONFIG ======================
DDL_FILE = "posgresdb/init.sql"

REAL_DB_CONFIG = {
    "host": "localhost",
    "dbname": "apartment_rentals",
    "user": "admin",
    "password": "adminpass",   # CHANGE THIS
    "port": "5432"
}

# ====================== SAFE STRING TRUNCATION ======================
MAX_ERROR_LEN = 500  # Prevent file name too long

def safe_error(msg: str) -> str:
    if len(msg) > MAX_ERROR_LEN:
        return msg[:MAX_ERROR_LEN] + "\n... [truncated]"
    return msg

# ====================== QUOTE & SQL CLEANUP ======================
def fix_quotes(sql: str) -> str:
    """Convert ""value"" → 'value' for PostgreSQL string literals"""
    return re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', lambda m: f"'{m.group(1)}'", sql)

def clean_generated_sql(sql: str) -> str:
    """Light cleanup: strip, remove extra quotes, fix common alias bugs"""
    sql = sql.strip()
    sql = sql.replace('\"', '"')  # Normalize
    sql = re.sub(r'\s+', ' ', sql)  # Collapse whitespace
    return sql

# ====================== VANNA ======================
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


# ====================== DDL TRAINING ======================
def train_schema_from_ddl():
    print("Training schema from init.sql...")
    with open(DDL_FILE, "r", encoding="utf-8") as f:
        ddl = f.read()
    for stmt in [s.strip() for s in ddl.split(";") if s.strip().upper().startswith("CREATE TABLE")]:
        clean = stmt.split("--")[0].strip()
        if clean:
            vn.train(ddl=clean)
    print("DDL training complete!")

train_schema_from_ddl()

# ====================== DB CONNECTION ======================
def connect_real_db():
    conn = psycopg2.connect(**REAL_DB_CONFIG)
    vn.connect_to_postgres(**REAL_DB_CONFIG)
    return conn

# ====================== SYSTEM PROMPT ======================
def load_system_prompt():
    if os.path.exists(SYSTEM_PROMPT_FILE):
        return open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8").read().strip()
    return ("You are a PostgreSQL expert. "
            "Generate only valid SQL. "
            "Use single quotes for strings. "
            "Use proper table aliases. "
            "Never use SELECT *.")

def save_system_prompt(txt):
    txt = txt.strip()
    open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8").write(txt)
    vn.system_message(txt)
    return "System prompt saved and applied."

vn.system_message(load_system_prompt())

# ====================== TRAIN ======================
def train_real(file_obj):
    if not file_obj:
        return "Upload train.csv"
    log = f"--- TRAINING {datetime.now():%Y-%m-%d %H:%M:%S} ---\n"
    save_log(TRAIN_LOG_FILE, log)

    df = pd.read_csv(file_obj.name)
    log += f"Loaded {len(df)} rows.\n\n"
    prog = gr.Progress()
    conn = connect_real_db()

    for i, row in df.iterrows():
        prog(i/len(df), desc=f"Training {i+1}/{len(df)}")
        q, sql = row["question"], fix_quotes(row["sql"])
        try:
            pd.read_sql(sql, conn)
            vn.train(question=q, sql=sql)
            log += f"[OK] {q[:40]}...\n"
        except Exception as e:
            log += f"[ERR] {q[:40]}... | {safe_error(str(e))}\n"
        finally:
            save_log(TRAIN_LOG_FILE, log)
        time.sleep(0.02)

    conn.close()
    log += "\n**TRAINING DONE**\n"
    save_log(TRAIN_LOG_FILE, log)
    return log

# ====================== TEST ======================
def test_real(file_obj):
    if not file_obj:
        return "Upload test.csv", None
    log = f"--- TEST {datetime.now():%Y-%m-%d %H:%M:%S} ---\n"
    save_log(TEST_LOG_FILE, log)

    df = pd.read_csv(file_obj.name)
    log += f"Testing {len(df)} rows.\n\n"
    results, q_pass, r_pass = [], 0, 0
    conn = connect_real_db()

    for _, row in df.iterrows():
        q, exp_sql = row["question"], fix_quotes(row["sql"])

        # Generate
        try:
            gen_sql_raw = vn.generate_sql(q)
            gen_sql = clean_generated_sql(gen_sql_raw)
            q_match = "PASS" if gen_sql.lower() == exp_sql.lower() else "FAIL"
            if q_match == "PASS": q_pass += 1
        except Exception as e:
            gen_sql, q_match = f"GEN ERR: {safe_error(str(e))}", "GEN FAIL"

        # Execute
        try:
            exp_df = pd.read_sql(exp_sql, conn)
            gen_df = pd.read_sql(gen_sql, conn) if "ERR" not in gen_sql else pd.DataFrame()
            r_match = "PASS" if exp_df.equals(gen_df) else "FAIL"
            if r_match == "PASS": r_pass += 1
        except Exception as e:
            r_match = f"EXEC ERR: {safe_error(str(e))}"

        results.append({
            "Question": q,
            "Expected SQL": exp_sql,
            "Generated SQL": gen_sql,
            "Query Match": q_match,
            "Result Match": r_match
        })
        log += f"[{q_match}/{r_match}] {q[:40]}...\n"
        save_log(TEST_LOG_FILE, log)

    conn.close()
    score = f"**QUERY ACC: {q_pass}/{len(df)}** | **RESULT ACC: {r_pass}/{len(df)}**"
    log += f"\n{score}\n"
    save_log(TEST_LOG_FILE, log)
    return score, pd.DataFrame(results)

# ====================== INTERACT ======================
def ask_real(q):
    try:
        raw = vn.generate_sql(q)
        return clean_generated_sql(raw)
    except Exception as e:
        return f"GEN ERR: {safe_error(str(e))}"

def run_real(sql):
    conn = connect_real_db()
    try:
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        return safe_error(f"SQL EXEC ERROR: {str(e)}")

# ====================== UI ======================
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .responsive-results .dataframe {width:100%!important;max-height:70vh!important;overflow:auto;display:block}
    .responsive-results table {width:100%!important;table-layout:fixed;border-collapse:collapse}
    .responsive-results th,.responsive-results td {white-space:normal!important;word-wrap:break-word!important;padding:10px 8px!important;font-size:13px;vertical-align:top}
    .responsive-results thead th {position:sticky;top:0;background:#f0f0f0;z-index:10;border-bottom:2px solid #999}
    .responsive-results .col-question {width:22%}
    .responsive-results .col-sql {width:35%}
    .responsive-results .col-match {width:8%;text-align:center}
    """
) as demo:
    gr.Markdown("# Text-to-SQL – Apartment Rentals (PostgreSQL)")

    with gr.Tabs():
        with gr.Tab("Train"):
            gr.Markdown("Upload `train.csv` → trains on live DB")
            train_file = gr.File(label="train.csv")
            train_btn = gr.Button("Start Training", variant="primary")
            train_log = gr.Textbox(label="Log", lines=20, value=load_log(TRAIN_LOG_FILE))
            train_btn.click(train_real, train_file, train_log)

        with gr.Tab("Test"):
            gr.Markdown("Upload `test.csv` → full comparison")
            test_file = gr.File(label="test.csv")
            test_btn = gr.Button("Run Test", variant="primary")
            test_score = gr.Markdown()
            test_results = gr.Dataframe(label="Results", interactive=False, elem_classes="responsive-results")
            test_log = gr.Textbox(label="Log", lines=15, value=load_log(TEST_LOG_FILE))
            test_btn.click(test_real, test_file, [test_score, test_results]).then(
                lambda: load_log(TEST_LOG_FILE), outputs=test_log
            )

        with gr.Tab("Interact"):
            gr.Markdown("Ask → SQL runs on your DB")
            q_in = gr.Textbox(label="Question", lines=2, placeholder="Apartments managed by Kyle?")
            gen_btn = gr.Button("Generate SQL")
            run_btn = gr.Button("Execute", variant="primary")
            sql_out = gr.Code(label="SQL", language="sql", lines=8)
            res_out = gr.Dataframe(label="Result")
            gen_btn.click(ask_real, q_in, sql_out)
            run_btn.click(run_real, sql_out, res_out)

        with gr.Tab("System Prompt"):
            gr.Markdown("Edit prompt sent to LLM")
            prompt_box = gr.Textbox(
                label="Prompt",
                value=load_system_prompt(),
                lines=12
            )
            save_btn = gr.Button("Save & Apply", variant="primary")
            status = gr.Markdown()
            save_btn.click(save_system_prompt, prompt_box, status)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)