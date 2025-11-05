from vanna.chromadb import ChromaDB_VectorStore
from vanna.ollama import Ollama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import gradio as gr  # For the UI

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


local_embed = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
vn = MyVanna(config={'model': 'llama3.1:8b', 'embedding_function': local_embed})

# Connect to the Docker Postgres DB
vn.connect_to_postgres(
    host='localhost',
    dbname='bankdb',
    user='bankuser',
    password='bankpass',
    port='5432'
)

# Train Vanna on the schema
vn.train(ddl="""
CREATE TABLE Customer (
    CustomerID SERIAL PRIMARY KEY,
    Name VARCHAR(255),
    Address VARCHAR(255),
    Contact VARCHAR(255),
    Username VARCHAR(50) UNIQUE,
    Password VARCHAR(255)
);
""")

vn.train(ddl="""
CREATE TABLE Account (
    AccountID SERIAL PRIMARY KEY,
    CustomerID INT,
    Type VARCHAR(50),
    Balance DECIMAL(10, 2),
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);
""")

vn.train(ddl="""
CREATE TABLE Transaction (
    TransactionID SERIAL PRIMARY KEY,
    AccountID INT,
    Type TEXT CHECK (Type IN ('deposit', 'withdrawal')),
    Amount DECIMAL(10, 2),
    Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (AccountID) REFERENCES Account(AccountID)
);
""")

vn.train(ddl="""
CREATE TABLE Beneficiary (
    BeneficiaryID SERIAL PRIMARY KEY,
    CustomerID INT,
    Name VARCHAR(255),
    AccountNumber VARCHAR(50),
    BankDetails VARCHAR(255),
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);
""")

vn.train(documentation="The bank database has customers with accounts, transactions (deposits/withdrawals), and beneficiaries for transfers.")
vn.train(sql="SELECT * FROM Account WHERE Balance > 1000;", question="Find accounts with balance over 1000.")

# Gradio UI Functions
def generate_sql(question):
    if not question:
        return "Please enter a question."
    try:
        sql = vn.generate_sql(question)
        return sql
    except Exception as e:
        return f"Error generating SQL: {str(e)}"

def run_query(sql):
    if not sql:
        return "No SQL to run."
    try:
        df = vn.run_sql(sql)
        return df.to_markdown(index=False)  # Display as Markdown table
    except Exception as e:
        return f"Error running query: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Bank DB Text-to-SQL UI") as demo:
    gr.Markdown("# Bank Database Query Assistant")
    gr.Markdown("Enter a natural language question about the bank DB, generate SQL, and view results.")
    
    question_input = gr.Textbox(label="Your Question", placeholder="e.g., List customers with savings accounts over $1000 balance.")
    
    with gr.Row():
        generate_btn = gr.Button("Generate SQL")
        run_btn = gr.Button("Run Query")
    
    sql_output = gr.Code(label="Generated SQL", language="sql")
    results_output = gr.Markdown(label="Query Results")
    
    # Bind buttons
    generate_btn.click(fn=generate_sql, inputs=question_input, outputs=sql_output)
    run_btn.click(fn=run_query, inputs=sql_output, outputs=results_output)

if __name__ == "__main__":
    demo.launch(share=True)