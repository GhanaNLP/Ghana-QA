import pandas as pd
import os
import time
import random
from datetime import datetime
from dotenv import load_dotenv, set_key, find_dotenv
import requests
from tkinter import Tk, filedialog

# ------------------------
# Setup environment file
# ------------------------
env_path = find_dotenv()
if not env_path:
    env_path = ".env"
    open(env_path, 'a').close()  # create if not exists

load_dotenv(env_path)

# ------------------------
# Load or prompt for API key and input file
# ------------------------
API_KEY = os.getenv("NVIDIA_API_KEY")
INPUT_PATH = os.getenv("INPUT_PATH")

if not API_KEY:
    print("🔑 No NVIDIA API key found.")
    API_KEY = input("Please enter your NVIDIA Build API key: ").strip()
    set_key(env_path, "NVIDIA_API_KEY", API_KEY)

if not INPUT_PATH:
    print("\n📁 No input CSV file path found.")
    print("Opening file picker dialog...")
    
    # Create a hidden Tkinter root window
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    # Open file picker dialog
    INPUT_PATH = filedialog.askopenfilename(
        title="Select your input CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    root.destroy()  # Close the Tkinter instance
    
    if not INPUT_PATH:
        raise ValueError("No file was selected. Exiting.")
    
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"The file '{INPUT_PATH}' does not exist.")
    
    set_key(env_path, "INPUT_PATH", INPUT_PATH)

# ------------------------
# Configuration
# ------------------------
MODEL_NAME = "mistralai/mistral-medium-3-instruct"
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
OUTPUT_PATH = os.path.join(os.getcwd(), "output_qa.csv")
BATCH_SIZE = 40
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5
MAX_ARTICLE_LENGTH = 1200
RATE_LIMIT_DELAY = 1.5

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

print(f"\n✅ Using NVIDIA Build API with model: {MODEL_NAME}")
print(f"📄 Input file selected: {INPUT_PATH}")
print(f"⏳ Validating and loading input file...")

# ------------------------
# Load data
# ------------------------
bad_rows = []

df = pd.read_csv(
    INPUT_PATH,
    encoding="latin1",
    engine="python",
    on_bad_lines=lambda row: bad_rows.append(row)
)

print(f"Skipped {len(bad_rows)} bad rows")


total_rows = len(df)
print(f"✅ File validated successfully!")
print(f"📊 Total rows in input file: {total_rows:,}")
print(f"💾 Output will be saved in this folder: {os.getcwd()}")
print(f"ℹ️  Note: If interrupted, simply run the script again to resume from where it stopped")

# ------------------------
# Resume logic
# ------------------------
start_idx = 0
write_header = True

if os.path.exists(OUTPUT_PATH):
    try:
        existing_df = pd.read_csv(OUTPUT_PATH)
        start_idx = len(existing_df)
        if start_idx > 0:
            print(f"\n⚠️  RESUMING from row {start_idx:,} (output file exists with {start_idx:,} completed rows)")
            write_header = False
    except Exception as e:
        print(f"\n⚠️  Could not read existing output file ({e}). Starting fresh...")
        start_idx = 0
else:
    print("\n✅ Starting fresh (no previous output file found)")

# ------------------------
# Function to generate queries with retry
# ------------------------
def generate_queries_with_retry(text, row_idx, max_retries=MAX_RETRIES):
    """Generate queries with exponential backoff on rate limits."""
    if len(text) > MAX_ARTICLE_LENGTH:
        text = text[:MAX_ARTICLE_LENGTH] + "..."

    prompt = f"""Give me 10 mutually exclusive Google search queries that different users might type to find this article. Each query should come from someone with NO prior knowledge of this specific event - they should be asking general questions or searching for broader information that this article would answer. The queries should NOT reference specific details from the article (like dates, operation names, or assume the reader knows an event occurred). In addition to the questions provide a 160-character answer to each question in the format below.
Format:
Q1:
A1:
Q2:
A2:
Q3:
A3:
Article: {text}
"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 2)
                print(f"    ⏳ Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)

            response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']

            elif response.status_code == 429:
                continue  # Rate limited, try again

            else:
                print(f"    ✗ HTTP {response.status_code}: {response.text}")
                return ""

        except requests.exceptions.Timeout:
            print(f"    ⏱️  Timeout on row {row_idx}. Retrying...")
            continue
        except Exception as e:
            print(f"    ✗ Error on row {row_idx}: {e}")
            continue

    print(f"    ✗ Failed after {max_retries} retries for row {row_idx}")
    return ""

# ------------------------
# Processing loop
# ------------------------
progress_interval = 100
start_time = datetime.now()

for i in range(start_idx, total_rows, BATCH_SIZE):
    end_idx = min(i + BATCH_SIZE, total_rows)
    batch = df.iloc[i:end_idx].copy()

    print(f"\n📦 Batch {i//BATCH_SIZE + 1}: Processing rows {i:,}-{end_idx-1:,}")
    
    batch_results = []
    successful = 0
    batch_start_time = time.time()
    
    for batch_idx, (original_idx, row) in enumerate(batch.iterrows()):
        text = row['content']
        
        if pd.isna(text) or not text.strip():
            print(f"  ⚠️  Skipping row {original_idx}: Empty content")
            batch_results.append("")
            continue
        
        result = generate_queries_with_retry(text, original_idx)
        batch_results.append(result)
        
        if result:
            successful += 1
            print(f"  ✓ Row {original_idx} done ({successful}/{batch_idx + 1})")
        
        time.sleep(RATE_LIMIT_DELAY + random.uniform(0, 0.3))
    
    batch['generated_queries'] = batch_results
    batch.to_csv(OUTPUT_PATH, mode='a', header=write_header, index=False)
    write_header = False
    
    elapsed = datetime.now() - start_time
    print(f"✅ Saved batch {i//BATCH_SIZE + 1} | Success: {successful}/{len(batch)} | Time: {elapsed}")

print("\n" + "="*50)
print(f"🎉 Processing complete! Output saved to: {OUTPUT_PATH}")
print(f"📊 Total rows processed: {total_rows:,}")
print(f"⏱️  Total time: {datetime.now() - start_time}")
print("📬 Please share your output file with Ghana NLP (michsethowusu@gmail.com) to contribute to the Ghana-QA project")
print("="*50)
