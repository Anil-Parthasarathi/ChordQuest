import json
from pathlib import Path
import string
import re
import math
import pickle

sheets = []
tokenized_sheets = []
universal_dictionary = {}
tfidf_retrieval_index = {}
lengths_dict = {}
num_docs = 0
avg_doc_length = 0

k1 = 1.2
b = 0.75

data_path = Path(__file__).parent / 'musescore_metadata' / 'musescore_song_metadata.jsonl'
cache_path = Path(__file__).parent / 'musescore_metadata' / 'bm25_index_cache.pkl'

def remove_punctuation_and_special_characters(text):
    return re.sub(r'[^a-z0-9\s]', ' ', text)

def has_letter(token):
    return any(char.isalpha() for char in token)

def process_token(input_string):
  tokenized = input_string.lower()
  tokenized = tokenized.replace("\n", " ")
  tokenized = remove_punctuation_and_special_characters(tokenized)
  tokenized = tokenized.split()
  tokenized = [token for token in tokenized if has_letter(token)]

  return tokenized

def build_index():
  global sheets, tokenized_sheets, universal_dictionary, tfidf_retrieval_index, lengths_dict, num_docs, avg_doc_length
  
  print(f"Loading data from {data_path}")
  
  all_sheets = []
  with open(data_path, "r", encoding="utf-8") as sheet_data_file:
    for line in sheet_data_file:
      all_sheets.append(json.loads(line))

  print("Tokenizing sheets and building indexes...")

  for sheet in all_sheets:
    if "id" not in sheet:
      continue
    
    sheets.append(sheet)
    
    title = sheet.get("title", "")
    description = sheet.get("description", "")
    combined_text = f"{title} {description}"
    tokenized_text = process_token(combined_text)
    
    doc_id = sheet["id"]
    tokenized_sheets.append({
        "id": doc_id,
        "tokens": tokenized_text
    })
    
    lengths_dict[doc_id] = len(tokenized_text)
    
    token_positions = {}
    for i, token in enumerate(tokenized_text):
      if token not in universal_dictionary:
        universal_dictionary[token] = 0
      universal_dictionary[token] += 1
      
      if token not in token_positions:
        token_positions[token] = []
      token_positions[token].append(i)
    
    for token, positions in token_positions.items():
      if token not in tfidf_retrieval_index:
        tfidf_retrieval_index[token] = []
      tfidf_retrieval_index[token].append((doc_id, positions))

  print("Finalizing index...")

  num_docs = len(lengths_dict)
  avg_doc_length = sum(lengths_dict.values()) / num_docs
  
  print("Saving index to cache...")
  cache_data = {
    'sheets': sheets,
    'tokenized_sheets': tokenized_sheets,
    'universal_dictionary': universal_dictionary,
    'tfidf_retrieval_index': tfidf_retrieval_index,
    'lengths_dict': lengths_dict,
    'num_docs': num_docs,
    'avg_doc_length': avg_doc_length
  }
  with open(cache_path, 'wb') as f:
    pickle.dump(cache_data, f)
  print("Index cached successfully!")

if cache_path.exists():
  print(f"Loading cached index from {cache_path}")
  with open(cache_path, 'rb') as f:
    cache_data = pickle.load(f)
    sheets = cache_data['sheets']
    tokenized_sheets = cache_data['tokenized_sheets']
    universal_dictionary = cache_data['universal_dictionary']
    tfidf_retrieval_index = cache_data['tfidf_retrieval_index']
    lengths_dict = cache_data['lengths_dict']
    num_docs = cache_data['num_docs']
    avg_doc_length = cache_data['avg_doc_length']
  print("Index loaded successfully!")
else:
  build_index()

def compute_bm25_idf(token):
  docs_containing_token = len(tfidf_retrieval_index[token])
  return math.log(((num_docs - docs_containing_token + 0.5) / (docs_containing_token + 0.5)) + 1)

def bm25_Search(query):

  print(f"Searching for: {query}")

  query_tokens = query.lower().split()

  idf_list = []

  for token in query_tokens:
    if token not in universal_dictionary:
      idf_list.append(0)
      continue
    idf_list.append(compute_bm25_idf(token))

  doc_count_dict = {}

  for index, token in enumerate(query_tokens):
    if token not in universal_dictionary:
      continue
    for sheet in tfidf_retrieval_index[token]:
      if sheet[0] not in doc_count_dict:
        doc_count_dict[sheet[0]] = [0] * len(query_tokens)
      doc_count_dict[sheet[0]][index] = len(sheet[1])

  scores = []

  for doc, doc_counts in doc_count_dict.items():

    cur_doc_length = lengths_dict[doc]

    doc_score = 0

    for index, count in enumerate(doc_counts):

      if count == 0:
        continue

      tf = math.log10(1 + count)

      idf = idf_list[index]
      numerator = tf * idf * (k1 + 1)
      denominator = tf + k1 * (1 - b + b * (cur_doc_length / avg_doc_length))

      doc_score += (numerator / denominator)

    scores.append((doc, doc_score))

  score_dict = {}

  for score in scores:
    score_dict[score[0]] = score[1]

  result = sorted(score_dict, key=lambda x: score_dict[x], reverse=True)

  sheet_id_to_data = {sheet["id"]: sheet for sheet in sheets}
  
  top_results = []
  for doc_id in result[:10]:
    sheet_data = sheet_id_to_data[doc_id]
    top_results.append({
      'id': sheet_data.get('id'),
      'title': sheet_data.get('title', 'Untitled'),
      'artist': sheet_data.get('authorUserId', 'Unknown'),
      'difficulty': 'N/A',
      'key': 'N/A',
      'url': sheet_data.get('url', ''),
      'score': score_dict[doc_id],
      'description': sheet_data.get('description', ''),
      'instrumentsNames': sheet_data.get('instrumentsNames', []),
      'pagesCount': sheet_data.get('pagesCount', 0),
      'partsCount': sheet_data.get('partsCount', 0)
    })

  return top_results