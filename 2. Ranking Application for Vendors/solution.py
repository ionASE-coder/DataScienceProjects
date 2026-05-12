import ast
import numpy as np
import pandas as pd
import pycountry
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re


def create_dictionary(key_value):
    if not key_value:
        return None
    else:
    
        parsed_query=key_value.group(1)
        parts = [p.strip("[]") for p in parsed_query.split(",")]
        # Convert parts to proper Python types
        parsed_parts = []
        for p in parts:
            try:
                parsed_parts.append(ast.literal_eval(p))
            except:
                parsed_parts.append(p.strip('"'))  # keep as string if literal_eval fails
        
        return(parsed_parts)


def query_parser(query):
    #defining the filters
    #defining the filters 
    dict_filters={
        "filter_founded":['<',2000],
        "filter_revenue":['>', 4801924000],
        "filter_employees":['=', 5287],
        "filter_countries":['Sweden', 'Norway', 'Finland'],
        "filter_public":True,
    }
    format_of_filters = json.dumps(dict_filters, indent=2)
    
    #loading LLM API
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    LLM=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    device=0 if torch.cuda.is_available() else -1
    prompt=f"""You are a query parsing assistant.
    
    Your task is to extract structured filters from a user query and return a valid dictionary.
    
    You MUST follow these rules strictly:
    
    1. Output ONLY a valid JSON dictionary. No explanations, no extra text.
    2. Use ONLY the keys provided in the filter schema below.
    3. If a value is not present in the query, set it to null.
    4. For numeric filters, return values as:
       ["operator", value]
       where operator is one of: ">", "<", "="
    
    5. Interpret natural language correctly:
       - "less than", "under", "below" → "<"
       - "more than", "over", "above" → ">"
       - "exactly", "equal to" → "="
    
    6. Country handling:
       - If a region is mentioned (e.g. "Scandinavian countries"), expand it:
         → ["Sweden", "Norway", "Finland"]
         If a continent is mentioned, expand it and provide all the countries in that region, such all countries in Europe
       - If one country is mentioned, return a single string (not a list)
    7. Make sure to output all the countries of that region. When asked about Europe output all countries in Europe.
    8. Financial and numeric normalization:
   - Convert all monetary values to absolute integers (no symbols, no words)
   - Examples:
     - "$50 million" → 50000000
     - "€2 billion" → 2000000000
     - "100k" → 100000
     - "1.5 million" → 1500000

   - Remove currency symbols like $, €, £
   - Always return numeric values as integers

   - Examples:
     Query: "companies with revenue over $50 million"
     Output:
     
       "filter_revenue": [">", 50000000]
    
     Query: "companies with revenue below 2 billion"
     Output:
       "filter_revenue": ["<", 2000000000]
    9. Interpret time and quantity expressions:

   - "after YEAR" → [">", YEAR]
   - "before YEAR" → ["<", YEAR]
   - "since YEAR" → [">", YEAR]
   - "prior to YEAR" → ["<", YEAR]

   - "fewer than X" → ["<", X]
   - "less than X" → ["<", X]
   - "more than X" → [">", X]
   - "at least X" → [">", X]
   - "at most X" → ["<", X]

   Examples:
   Query: "companies founded after 2018"
   Output:
   
     "filter_founded": [">", 2018]
   

   Query: "companies with fewer than 200 employees"
   Output:
   
     "filter_employees": ["<", 200]
   

    10. Always output the result after the word Output:
    
    ---
    
    Filter schema:
    {format_of_filters}
    
    ---
    
    Examples:
    
    Query: "Companies with less than 500 employees"
    Output:
    
      "filter_employees": ["<", 500]
    
    
    Query: "Tech companies in Scandinavian countries"
    Output:
    
      "filter_countries": ["Sweden", "Norway", "Finland"]
    
    
    ---
    
    Now process the following query:
    
    Query: {query}
    
    """
    inputs=tokenizer(prompt, return_tensors="pt").to(LLM.device)
    outputs=LLM.generate(**inputs, max_new_tokens=150, temperature=0.2, do_sample=False)
    AI_response=tokenizer.decode(outputs[0])
    
    #selecting the output
    output_only=AI_response.split("Output")[-1].strip()

        
    #creating query dictionary
    query_filters={}
    #copmany founded in?
    founded=re.search(r'"filter_founded"\s*:\s*\[\s*([^\]]+)\s*\]',output_only)
    query_filters["filter_founded"]=create_dictionary(founded)
    #revenue of the company?
    revenue=re.search(r'"filter_revenue"\s*:\s*\[\s*([^\]]+)\s*\]',output_only)
    query_filters["filter_revenue"]=create_dictionary(revenue)
    #company with how many employees?
    employees=re.search(r'"filter_employees"\s*:\s*\[\s*([^\]]+)\s*\]',output_only)
    query_filters["filter_employees"]=create_dictionary(employees)
    #in what country?
    countries=re.search(r'"filter_countries"\s*:\s*(?P<countries>\[.*?\]|"[^"\n]*")',output_only, re.S)
    query_filters["filter_countries"]=create_dictionary(countries)
    print(countries)
    #is the company public?
    public=re.search(r'"filter_public"\s*:\s*(true|false|False|True)',output_only)
    query_filters["filter_public"]=create_dictionary(public)

    return query_filters
    

    
    
        


#converting form string to dict
def convert_to_dict(x):
    if isinstance(x, dict):   # already parsed
        return x
    if pd.isna(x):            # null
        return None
    if isinstance(x, str):    # needs parsing
        try:
            return ast.literal_eval(x)
        except:
            return None
    return None


def extract_text_for_embeddings(row):
    dict_embed={"primary_naics":{"label":[]}, 'description':[], 'business_model':[], 'target_markets':[],  "core_offerings":[], "secondary_naics":{"label":[]}}
    text_for_embeddings=[]
    for key in dict_embed.keys():
        val=row.get(key)
        if isinstance(val, dict) and 'label' in val:
            text_for_embeddings.append(f"{key} {str(val['label'])}")
        elif isinstance(val, list):
            text_for_embeddings.append(f'{key} {" ".join(str(v) for v in val if v)}')
        elif val is not None:
            text_for_embeddings.append(val)
    return text_for_embeddings



#extracting the filters
def extract_address_fields(addr):
    if not isinstance(addr, dict):
        return pd.Series([None, None, None, None, None])
    
    return pd.Series([
        addr.get("country_code"),
        addr.get("region_name"),
        addr.get("town"),
        addr.get("latitude"),
        addr.get("longitude")
    ])



#creating additional column for country interrogation
def code_to_country(code):
    if not code:
        return None
    try:
        return pycountry.countries.get(alpha_2=code.upper()).name
    except:
        return None


def input_data_transform(df):
   
    #spliting data of the table into two separate parts 
    df_model=df.copy()
    df_model['text_for_embeddings']=None
    #converting string column to dictionary
    df_model['address'] = df_model["address"].apply(convert_to_dict)
    df_model['primary_naics'] = df_model["primary_naics"].apply(convert_to_dict)
    df_model['secondary_naics'] = df_model["secondary_naics"].apply(convert_to_dict)
    #defining how the embedding dictionry looks like
    dict_embed={"primary_naics":{"label":[]}, 'description':[], 'business_model':[], 'target_markets':[],  "core_offerings":[], "secondary_naics":{"label":[]}}
    #preparing the column text_for_embeddings
    df_model['text_for_embeddings']=df_model.apply(lambda row: " ".join(extract_text_for_embeddings(row)), axis=1)
    
    #preparing columns for filtering
    df_model[["filter_country_code", "filter_region", "filter_city", "filter_lat", "filter_lon"]] = df_model["address"].apply(extract_address_fields)

    #Creating Country column
    df_model['filter_country']=df_model['filter_country_code'].apply(code_to_country)

    #taking care of the other columns that will make a filter
    ##year founded
    df_model['filter_founded']=df_model["year_founded"].astype("Int64")
     ##revenue
    df_model["filter_revenue"]=df_model['revenue'].astype("Int64")
    ## is public
    df_model["filter_public"]=df_model['is_public'].astype(bool)

    return df_model


def embedder(df_model, query):
    
    #creating embeddings
    device="cuda" if torch.cuda.is_available() else "cpu"
    embedder=SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
    texts = df_model['text_for_embeddings'].to_list()
    embeddings_list=[]
    
    with torch.no_grad():
        for i in tqdm(range(len(texts))):
            embeddings=embedder.encode(texts[i], convert_to_numpy=True)
            embeddings_list.append(embeddings)


    #Insrting EMbeddings in DataFrame
    df_model["embedding"]=None
    df_model['embedding']=embeddings_list
    np_embeddings=np.array(embeddings_list)

    #generating embedding for querry
    query_embedding=embedder.encode(query, convert_to_numpy=True)
    
    #cosine similarity
    cos_scores = cosine_similarity(np_embeddings.reshape(len(embeddings_list),-1), query_embedding.reshape(1,-1))
    #writing scores to columns
    df_model['embed_score']=cos_scores

    
    results=df_model.sort_values('embed_score', ascending=False).head(5)
    #provizioriu return results
    return results
    
    
    
    
    
    


def check_if_empty_df(mask,df_filter,key):
    """ Avoiding giving empty dataframes"""
    if isinstance(mask, pd.Series):
        if mask.any():
            return df_filter[mask]
        else:
            print(f"Because for the filter {key} there weren't any matches it was discarded")
            return df_filter
            
    else:
        return df_filter


def apply_filters(query_filters, df_model):
    #renaming columns
    df_filter=df_model.copy()
    df_filter.rename(columns={"revenue":"filter_revenue", 'employee_count':'filter_employees', 'filter_country':'filter_countries'}, inplace=True )



    for key, value in query_filters.items():
        if isinstance(value, list) and len(value) == 2 and value[0] in [">", "<", "="]:
            op, v = value
            print(f"Filtering for {key} that has operator {op} and value {v}")
            if op == ">":
                mask = df_filter[key] > v
                df_filter=check_if_empty_df(mask, df_filter, key)
            elif op == "<":
                 mask = df_filter[key] < v
                 df_filter=check_if_empty_df(mask, df_filter, key)
            elif op == "=":
                mask = df_filter[key] == v
                df_filter=check_if_empty_df(mask, df_filter, key)
        elif value is not None:
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
                if isinstance(value, str) and value.lower() in ["true", "false", 'True', 'False']:
                    value = value.lower() == "true"
            if isinstance(value, bool):
                mask = df_filter[key] == value
                print(f"Filtering for {key} that has the following values {value}")
                df_filter=check_if_empty_df(mask, df_filter, key)
            elif isinstance(value, list):
                mask = df_filter[key].isin(value)
                df_filter=check_if_empty_df(mask, df_filter, key)
                print(f"Filtering for {key} that has the following values {(value)}")
            else:
                mask = df_filter[key] == value
                df_filter=check_if_empty_df(mask, df_filter, key)
                print(f"Filtering for {key} that has the following values {(value)}")
            
    return df_filter
    
    


def query_rephraser(query):
    
     #loading LLM API
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map='auto')
    LLM=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    device=0 if torch.cuda.is_available() else -1
    prompt=f"""You are an expert at query expansion for data matching. I will provide you with:
    A short user query.
    
    Your task is to:
    - Suggest 5 additional keywords or phrases that could help match this query more effectively to the dataset
    - Provide synonyms using an economic vocabulary 
    - Provide a pyhton list enclosed in square brackets with the 5 words after the word "Output"
    - DON'T write other confirmations or descriptions
    - Your output should be like Output: synonym1, synonym2, synonym3, synonym4, synonym5 
    User query: "{query}" 
   """
    
    inputs=tokenizer(prompt, return_tensors="pt").to(LLM.device)
    outputs=LLM.generate(**inputs, max_new_tokens=150, temperature=0.2, do_sample=False)
    AI_response=tokenizer.decode(outputs[0], skip_special_tokens=True)
    #selecting the output
    output_only=AI_response.split("Output")[-1].strip()
    
    return f"{query} {output_only}"


#main
#loading dataset
df = pd.read_json("companies.jsonl", lines=True)
query=input("Enter Query: ")
#query="Fast-growing fintech companies competing with traditional banks in Europe"
#optaining the preprocessed dataframe
df_model=input_data_transform(df)
#apply reasoning on the query
AI=(query_parser(query))
print(f"Identified filters by AI {AI}")
#if no filters identified rephrasing querty to better match dataset embeddings
if all(v is None for v in AI.values()):
    print("No filters extracted from AI")
    query=query_rephraser(query)
    print(f"Rephrasing query for better matching, new querry is {query}")
#filter according to the reasoning
df_filtered=apply_filters(AI, df_model)
#obtaining the embeddings out of the filtered results, reranking them in descending order
df_embedded_score=embedder(df_filtered,query)
#print top 5 results
print(df_embedded_score)




