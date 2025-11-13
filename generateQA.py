import pandas as pd
import os
import time
import random
from datetime import datetime
from dotenv import load_dotenv
import requests

# ------------------------
# Configuration
# ------------------------
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY", "ENTER-NVIDIA-API-KEY-HERE")  # Load from .env
MODEL_NAME = "mistralai/mistral-medium-3-instruct"
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
INPUT_PATH = "/media/owusus/Godstestimo/NLP-Projects/GhanaQA/data/news/myjoyonline.csv"
OUTPUT_PATH = "/media/owusus/Godstestimo/NLP-Projects/GhanaQA/output_qa.csv"
BATCH_SIZE = 40  # Adjusted for 40 requests per minute
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5  # Start with 5 seconds
MAX_ARTICLE_LENGTH = 1200
RATE_LIMIT_DELAY = 1.5  # 60 seconds / 40 requests = 1.5 seconds per request

# ------------------------
# Initialize headers
# ------------------------
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}
print(f"✅ Using NVIDIA Build API with model: {MODEL_NAME}")

# ------------------------
# Load data
# ------------------------
df = pd.read_csv(INPUT_PATH)
total_rows = len(df)
print(f"Total rows in input file: {total_rows:,}")

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
# Generate queries with retry logic
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
            # Add jitter to avoid thundering herd
            if attempt > 0:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 2)
                print(f"    ⏳ Rate limited. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            
            response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=60)
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            
            # Handle rate limiting
            elif response.status_code == 429:
                if attempt == max_retries - 1:
                    print(f"    ✗ Max retries exceeded. Skipping row {row_idx}.")
                    return ""
                # Continue to next retry attempt
                continue
            
            # Handle other HTTP errors
            else:
                print(f"    ✗ Error on row {row_idx}: HTTP {response.status_code} - {response.text}")
                return ""
            
        except requests.exceptions.Timeout:
            print(f"    ⏱️  Timeout on row {row_idx}. Retrying...")
            if attempt == max_retries - 1:
                print(f"    ✗ Max retries exceeded. Skipping row {row_idx}.")
                return ""
            continue
            
        except Exception as e:
            error_msg = str(e)
            print(f"    ✗ Error on row {row_idx}: {error_msg}")
            if attempt == max_retries - 1:
                return ""
            continue
    
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
        
        # Skip empty content
        if pd.isna(text) or not text.strip():
            print(f"  ⚠️  Skipping row {original_idx}: Empty content")
            batch_results.append("")
            continue
        
        result = generate_queries_with_retry(text, original_idx)
        batch_results.append(result)
        
        if result:
            successful += 1
            print(f"  ✓ Processed row {original_idx} (success {successful}/{batch_idx + 1})")
        
        # Rate limiting: 40 requests per minute
        # Add a small buffer to be safe (1.5 seconds minimum between requests)
        time.sleep(RATE_LIMIT_DELAY + random.uniform(0, 0.3))
    
    # Ensure we don't exceed rate limit for the batch
    batch_elapsed = time.time() - batch_start_time
    min_batch_time = len(batch) * RATE_LIMIT_DELAY
    if batch_elapsed < min_batch_time:
        sleep_time = min_batch_time - batch_elapsed
        print(f"  ⏸️  Pausing {sleep_time:.1f}s to respect rate limit...")
        time.sleep(sleep_time)
    
    # Save batch
    batch['generated_queries'] = batch_results
    batch.to_csv(OUTPUT_PATH, mode='a', header=write_header, index=False)
    write_header = False
    
    # Progress summary
    elapsed = datetime.now() - start_time
    rows_processed = min(i + len(batch), total_rows) - start_idx
    rate = rows_processed / elapsed.total_seconds() * 3600 if elapsed.total_seconds() > 0 else 0
    print(f"✅ Saved batch {i//BATCH_SIZE + 1} | Success: {successful}/{len(batch)} | Elapsed: {elapsed} | Rate: {rate:.0f} rows/hr")
    
    # Save checkpoint every 500 rows
    if (i + len(batch)) % 500 == 0:
        print(f"🔄 Checkpoint saved at row {i + len(batch):,}")

# Final summary
print("\n" + "="*50)
print(f"🎉 Processing complete! Output saved to: {OUTPUT_PATH}")
print(f"📊 Total rows processed: {total_rows:,}")
print(f"⏱️  Total time: {datetime.now() - start_time}")
