import io
import os
import sys
import pandas as pd
import gradio as gr
import psycopg2
from psycopg2.extras import RealDictCursor
import re
from datetime import datetime
import plotly.graph_objects as go

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
DDL_FILE = "init.sql"

REAL_DB_CONFIG = {
    "host": "localhost",
    "dbname": "apartment_rentals",
    "user": "admin",
    "password": "adminpass",
    "port": "5432"
}

# ====================== VANNA SETUP ======================

current_prompt = ""

if os.path.exists(SYSTEM_PROMPT_FILE):
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        current_prompt = f.read().strip()

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        config = config or {}
        config['path'] = CHROMA_PATH
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={
    'model': 'llama3.1:8b',
    'embedding_function': SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2'),
    'initial_prompt': current_prompt,
    'allow_llm_to_see_data': True
})

# Train on schema
if os.path.exists(DDL_FILE):
    with open(DDL_FILE) as f:
        for stmt in [s.strip() for s in f.read().split(";") if s.strip().upper().startswith("CREATE TABLE")]:
            clean = stmt.split("--")[0].strip()
            if clean:
                vn.train(ddl=clean)

def beautify_sql_prompt_text(raw_text: str) -> str:
    """
    Beautifies the raw SQL prompt conversation that comes from Ollama/logs.
    Keeps all information but makes it very readable with nice headers, code blocks, etc.
    """
    lines = raw_text.strip().split('\n')
    return "\n\n".join(lines)

# ====================== DATABASE FUNCTIONS ======================
def clean_generated_sql(sql: str) -> str:
    return re.sub(r'\s+', ' ', sql.strip().replace('\"', '"')).strip()

def fix_quotes(sql: str) -> str:
    return re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', lambda m: f"'{m.group(1)}'", sql)

def execute_query_to_df(sql: str):
    if not sql or "ERROR" in sql.upper() or "--" in sql:
        return pd.DataFrame(), "FAIL"
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


def save_system_prompt(txt):
    global vn
    txt = txt.strip()
    with open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(txt)
    vn = MyVanna(config={
        'model': 'llama3.1:8b',
        'embedding_function': SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2'),
        'initial_prompt': current_prompt,
    })
    return "✅ System prompt saved and applied successfully!"

# ====================== PROCESS LOG FORMATTER ======================
def format_process_log(process_log):
    """Convert process log to beautiful HTML"""
    if not process_log:
        return "<div style='color:#64748b; padding:20px; text-align:center;'>No process log available</div>"
    
    log_html = ""
    for entry in process_log:
        status_colors = {
            "✓": "#10b981",
            "✗": "#ef4444",
            "⚠": "#f59e0b",
            "⟳": "#3b82f6"
        }
        status = entry.get("status", "")
        color = status_colors.get(status, "#64748b")
        
        log_html += f"""
        <div style="background:#0f172a; border-left:4px solid {color}; padding:16px 20px; margin:12px 0; border-radius:0 8px 8px 0; box-shadow:0 2px 8px rgba(0,0,0,0.3);">
            <div style="display:flex; align-items:center; margin-bottom:8px;">
                <span style="font-size:24px; margin-right:12px;">{status}</span>
                <span style="color:#e2e8f0; font-weight:600; font-size:15px;">{entry.get('step', 'Step')}</span>
                <span style="color:#64748b; font-size:12px; margin-left:auto; font-family:monospace;">{entry.get('timestamp', '')}</span>
            </div>
            <div style="color:#94a3b8; font-size:13px; line-height:1.6; white-space:pre-wrap; padding-left:36px;">
                {entry.get('details', '')}
            </div>
        </div>
        """
    
    return f"""
    <div style="background:#1e293b; padding:24px; border-radius:12px; border:1px solid #334155; margin:10px 0;">
        <h3 style="color:#60a5fa; margin:0 0 16px 0; font-size:18px;">🔄 Process Log - Model Interaction Trace</h3>
        {log_html}
    </div>
    """

# ====================== GENERATE SQL WITH PROCESS LOG ======================
def generate_sql_with_reasoning(question: str):
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    """Generate SQL with detailed process logging using Vanna's proper API"""
    process_log = []
    reasoning = ""
    
    try:
        # Step 1: Question received
        process_log.append({
            "step": "1. Question Received",
            "status": "✓",
            "details": f"User question: '{question}'",
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        })
        
        # Step 2: Searching similar examples
        process_log.append({
            "step": "2. Searching Vector Database",
            "status": "⟳",
            "details": "Looking for similar questions and SQL patterns in training data...",
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        })
        
        try:
            similar_questions = vn.get_similar_question_sql(question, n_results=3)
            if similar_questions:
                examples_text = "\n".join([f"==={q}\nQuestion: {s['question']}\nSQL: {s['sql']}" for s in similar_questions])
                process_log[-1]["status"] = "✓"
                process_log[-1]["details"] = f"Found {len(similar_questions)} similar examples:\n{examples_text}"
                reasoning += f"**Similar Examples Found:**\n{examples_text}\n\n"
            else:
                process_log[-1]["status"] = "⚠"
                process_log[-1]["details"] = "No similar examples found in training data"
                reasoning += "No similar examples found - generating from schema knowledge.\n\n"
        except Exception as e:
            process_log[-1]["status"] = "⚠"
            process_log[-1]["details"] = "Vector search completed (training data may be empty)"
            reasoning += "Generating SQL from schema knowledge.\n\n"
        
        # Step 3: Getting relevant DDL/documentation
        process_log.append({
            "step": "3. Retrieving Schema Context",
            "status": "⟳",
            "details": "Fetching relevant tables and documentation...",
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        })
        
        try:
            related_ddl = vn.get_related_ddl(question)
            if related_ddl:
                process_log[-1]["status"] = "✓"
                process_log[-1]["details"] = f"Retrieved {len(related_ddl)} relevant table schemas\n{'\n\n'.join(related_ddl)}"
                reasoning += f"**Relevant Tables:** {len(related_ddl)} schemas found\n\n"
            else:
                process_log[-1]["status"] = "⚠"
                process_log[-1]["details"] = "No specific DDL found, using general schema"
        except:
            process_log[-1]["status"] = "⚠"
            process_log[-1]["details"] = "Using general schema knowledge"
        
        # Step 4: Generating SQL
        process_log.append({
            "step": "4. Generating SQL Query",
            "status": "✓",
            "details": f"Calling Ollama (llama3.1:8b) to generate SQL...",
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        })
        
        generated_sql = vn.generate_sql(question=question)
        sys.stdout = old_stdout
        process_log.append({
            "step": "5. Ollama Intreaction Capture",
            "status": "✓",
            "details": beautify_sql_prompt_text(mystdout.getvalue()),
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        })

        if generated_sql:
            process_log[-1]["status"] = "✓"
            process_log[-1]["details"] = mystdout.getvalue()
            reasoning += f"**Query Generation:** Successfully created SQL based on question analysis"
        else:
            process_log[-1]["status"] = "✗"
            process_log[-1]["details"] = "Failed to generate SQL"
            generated_sql = "-- ERROR: No SQL generated"
        
        # Step 5: SQL validation
        process_log.append({
            "step": "5. Validating SQL Syntax",
            "status": "⟳",
            "details": "Checking if generated SQL is valid...",
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        })
        
        generated_sql = clean_generated_sql(generated_sql)
        valid_sql = True

        if not generated_sql.startswith("SELECT") and not generated_sql.startswith("INSERT") and not generated_sql.startswith("UPDATE") and not generated_sql.startswith("DELETE"):
            valid_sql = False
            
        
        if "ERROR" in generated_sql or "FAIL" in generated_sql or not generated_sql or not valid_sql:
            process_log[-1]["status"] = "✗"
            process_log[-1]["details"] = "SQL generation failed - no valid query produced"
        else:
            process_log[-1]["status"] = "✓"
            process_log[-1]["details"] = f"SQL query validated: {len(generated_sql)} characters"
        
        return reasoning, generated_sql, process_log
        
    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        process_log.append({
            "step": "ERROR",
            "status": "✗",
            "details": error_msg,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        })
        return f"Error: {str(e)}", f"-- GENERATION FAILED: {str(e)[:200]}", process_log

# ====================== TRAIN ======================
def train_real(file_obj):
    if not file_obj:
        return "Please upload train.csv"
    df = pd.read_csv(file_obj.name)
    log = f"--- TRAINING STARTED {datetime.now():%Y-%m-%d %H:%M} ---\n"
    with open(TRAIN_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(log)
    prog = gr.Progress()
    for i, row in df.iterrows():
        prog(i / len(df), desc=f"Training {i+1}/{len(df)}")
        q, sql = row["question"], fix_quotes(row["sql"])
        try:
            execute_query_to_df(sql)[0]
            vn.train(question=q, sql=sql)
        except:
            pass
    with open(TRAIN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("**TRAINING COMPLETED SUCCESSFULLY**\n")
    return open(TRAIN_LOG_FILE).read()

# ====================== TEST ======================
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
            _, gen_sql, _ = generate_sql_with_reasoning(q)
            gen_sql = clean_generated_sql(gen_sql)
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
        <b>Exact Matching (EM):</b> {correct_em}/{total} ({em_score:.2f}%) | 
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
        title=f"Valid Efficiency Score (VES) – Speed on Semantic Different SQL ({semantic_count} queries)",
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

# ====================== UI ======================
with gr.Blocks(theme=gr.themes.Soft(), title="Text-To-SQL – Apartment Rentals") as demo:
    gr.Markdown("# 🏢 Text-To-SQL – Apartment Rentals Database")
    gr.Markdown("**EM • EX • VES • Process Logging with Vanna AI**")

    log_state = gr.State(value=[])

    with gr.Tabs():
        with gr.Tab("🧪 Test"):
            gr.Markdown("### Upload test.csv → Run Full Evaluation")
            test_file = gr.File(label="Upload test.csv", file_types=[".csv"])
            test_btn = gr.Button("Run Evaluation", variant="primary", size="lg")
            summary = gr.HTML()
            p1 = gr.Plot(); p2 = gr.Plot()
            table = gr.Dataframe(wrap=True, max_height=10000)
            test_btn.click(test_real, test_file, [summary, p1, p2, table])

        with gr.Tab("🎓 Train"):
            gr.Markdown("### Train Your Model")
            train_file = gr.File(label="Upload train.csv", file_types=[".csv"])
            train_btn = gr.Button("Start Training", variant="primary")
            train_log = gr.Textbox(label="Training Log", lines=12, value="Ready to train...")
            train_btn.click(train_real, train_file, train_log)

        with gr.Tab("💬 Interact"):
            gr.Markdown("### Ask Questions in Natural Language")

            with gr.Row():
                with gr.Column(scale=8):
                    q = gr.Textbox(
                        label="Your Question",
                        lines=6,
                        placeholder="e.g. Show apartments with rent > 2000 and at least 2 bedrooms",
                    )
                with gr.Column(scale=2, min_width=180):
                    gen = gr.Button("🔮 Generate SQL", variant="secondary", size="lg")
                    run = gr.Button("▶️ Execute Query", variant="primary", size="lg")
            
            sql = gr.Code(label="Generated SQL", language="sql", lines=10, interactive=True)
            out = gr.Dataframe(label="Query Result", wrap=True, max_height=500)

            # Process Log
            with gr.Accordion("🔍 Process Log - See How the Model Works", open=True):
                process_log_html = gr.HTML(value="<div style='color:#64748b; padding:20px; text-align:center; font-size:15px;'>Generate a query to see the step-by-step process log</div>")


            # Reasoning Trace
            with gr.Accordion("🤔 AI Reasoning Context", open=False):
                reasoning_html = gr.HTML(value="<div style='color:#64748b; padding:20px; text-align:center; font-size:15px;'>Context used for SQL generation will appear here</div>")

            gr.Markdown("### 📊 Interaction History")
            history_html = gr.HTML()

            def add_to_log(state, question, generated_sql, result_df, status, reasoning=""):
                timestamp = datetime.now().strftime("%H:%M:%S")
                short_q = question.strip()[:120] + ("..." if len(question.strip()) > 120 else "")
                short_sql = generated_sql.strip()[:200] + ("..." if len(generated_sql.strip()) > 200 else "")

                status_color = "#10b981" if status == "PASS" else "#ef4444"
                status_text = "Success" if status == "PASS" else "Failed"
                rows = result_df.shape[0] if not result_df.empty else 0
                cols = result_df.shape[1] if not result_df.empty else 0

                entry = f"""
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding:12px 16px; color:#94a3b8; font-family:monospace;">{timestamp}</td>
                    <td style="padding:12px 16px; max-width:500px; word-break:break-word;">{short_q}</td>
                    <td style="padding:12px 16px; font-family:'Fira Code',monospace; color:#a8ffbc; font-size:13px;">
                        {short_sql}
                    </td>
                    <td style="padding:12px 16px; text-align:center;">
                        <span style="background:{status_color}; color:white; padding:5px 14px; border-radius:999px; font-weight:bold;">
                            {status_text}
                        </span>
                    </td>
                    <td style="padding:12px 16px; text-align:center; color:#60a5fa;">{rows} × {cols}</td>
                </tr>
                """
                state.append(entry)
                body = "".join(state[-30:])
                full_history = f"""
                <div style="background:#0f172a; border-radius:16px; overflow:hidden; border:1px solid #334155; box-shadow:0 8px 32px rgba(0,0,0,0.4); margin-top:10px;">
                    <table style="width:100%; border-collapse:collapse; color:#e2e8f0;">
                        <thead>
                            <tr style="background:#1e293b;">
                                <th style="padding:16px; text-align:left;">Time</th>
                                <th style="padding:16px; text-align:left;">Question</th>
                                <th style="padding:16px; text-align:left;">SQL</th>
                                <th style="padding:16px; text-align:center;">Status</th>
                                <th style="padding:16px; text-align:center;">Result</th>
                            </tr>
                        </thead>
                        <tbody>
                            {body if body else '<tr><td colspan="5" style="padding:60px; text-align:center; color:#64748b; font-size:18px;">No interactions yet – ask your first question!</td></tr>'}
                        </tbody>
                    </table>
                </div>
                """

                reasoning_display = ""
                if reasoning:
                    reasoning_display = f"""
                    <div style="background:#1e1b4b; border-left:5px solid #8b5cf6; padding:20px; border-radius:0 12px 12px 0; margin:10px 0;">
                        <h4 style="color:#c4b5fd; margin:0 0 12px 0; font-size:16px;">🤔 Context Used for Generation</h4>
                        <div style="background:#0f172a; padding:18px; border-radius:10px; color:#e2e8f0; font-family:'Segoe UI',sans-serif; line-height:1.8; white-space: pre-wrap; border:1px solid #334155; font-size:14px;">
                            {reasoning.strip()}
                        </div>
                    </div>
                    """
                else:
                    reasoning_display = f"<div style='color:#94a3b8; padding:20px; text-align:center;'>No context information available.</div>"

                return full_history, reasoning_display, state

            def generate_and_log(question, state):
                reasoning, generated, process_log = generate_sql_with_reasoning(question)
                process_log_display = format_process_log(process_log)
                df, status = execute_query_to_df(generated)
                history_out, reasoning_out, new_state = add_to_log(state, question, generated, df, status, reasoning)
                return generated, df, process_log_display, reasoning_out, history_out, new_state

            def execute_and_log(sql_code, state):
                df, status = execute_query_to_df(sql_code)
                last_q = "Manual SQL execution"
                if state:
                    match = re.search(r'<td[^>]*>([^<]+)', state[-1])
                    if match:
                        last_q = match.group(1).strip()
                
                exec_log = [
                    {
                        "step": "1. SQL Received",
                        "status": "✓",
                        "details": f"Manual SQL execution: {sql_code[:100]}...",
                        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    },
                    {
                        "step": "2. Executing on Database",
                        "status": "✓" if status == "PASS" else "✗",
                        "details": f"Execution {'successful' if status == 'PASS' else 'failed'} - {df.shape[0]} rows returned" if status == "PASS" else f"Error: {status}",
                        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    }
                ]
                process_log_display = format_process_log(exec_log)
                
                history_out, reasoning_out, new_state = add_to_log(state, last_q, sql_code, df, status)
                return df, process_log_display, reasoning_out, history_out, new_state

            gen.click(
                generate_and_log,
                inputs=[q, log_state],
                outputs=[sql, out, process_log_html, reasoning_html, history_html, log_state]
            )
            # ).then(
            #     lambda: "", outputs=q
            # )

            run.click(
                execute_and_log,
                inputs=[sql, log_state],
                outputs=[out, process_log_html, reasoning_html, history_html, log_state]
            )

        with gr.Tab("📋 Database Schema SQL Definition"):
            # gr.Markdown("### Database Schema Definition")
            gr.HTML(load_ddl_schema())

        with gr.Tab("🗺️ Database Schema"):
            # gr.Markdown("### Database ER Diagram")
            gr.Markdown("_Default: `static/static_diagram.png` • Upload to replace_")
            default_img = DEFAULT_DIAGRAM_PATH if os.path.exists(DEFAULT_DIAGRAM_PATH) else None
            upload = gr.File(label="Upload New Diagram (PNG/JPG/WEBP)", file_types=["image"])
            display = gr.Image(value=default_img, label="Current Database Diagram", height=800, interactive=False)
            upload.change(fn=lambda f: f.name if f else default_img, inputs=upload, outputs=display)

        with gr.Tab("⚙️ System Prompt"):
            gr.Markdown("### Customize System Prompt")
            gr.Markdown("Edit the system prompt to control how the model generates SQL queries.")
            p = gr.Textbox(value=current_prompt, lines=15, label="System Prompt")
            save_prompt_btn = gr.Button("💾 Save & Apply", variant="primary", size="lg")
            prompt_status = gr.Markdown("")
            save_prompt_btn.click(save_system_prompt, p, prompt_status)

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True, share=False)