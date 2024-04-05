import os
from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()

API_TOKEN_ONE = os.getenv("API_TOKEN_ONE")
API_TOKEN_TWO = os.getenv("API_TOKEN_TWO")
API_TOKEN_THREE = os.getenv("API_TOKEN_THREE")
API_TOKEN_FOUR = os.getenv("API_TOKEN_FOUR")
API_TOKEN_FIVE = os.getenv("API_TOKEN_FIVE")
API_TOKEN_SIX = os.getenv("API_TOKEN_SIX")
API_TOKEN_SEVEN = os.getenv("API_TOKEN_SEVEN")
API_TOKEN_EIGHT = os.getenv("API_TOKEN_EIGHT")

df = pd.read_csv('./dataset_creation.csv')

tokens_array = [
    API_TOKEN_ONE,
    API_TOKEN_TWO,
    API_TOKEN_THREE,
    API_TOKEN_FOUR,
    API_TOKEN_FIVE,
    API_TOKEN_SIX,
    API_TOKEN_SEVEN,
    API_TOKEN_EIGHT
]

global_index = 0

def construct_prompt(row):
    prompt = f"""
        You are a sales strategist tasked with improving sales for a specific product.
        The product is from the {row['department_name']} department and belongs to the {row['class_name']} class.
        Here's a review of the product: "{row['review_text']}".
        The sentiment class for this product is {row['sentiment_class']}.

        Generate a sales strategy consisting of 5 points to improve sales for this product.
        Include strategies that leverage the product's features, customer sentiment, and market trends.
        Make sure each point is actionable and provides a detailed explanation.
        """

    return prompt


def generate_rec(prompt,API_TOKEN):
  API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
  headers = {"Authorization": f"Bearer {API_TOKEN}"}

  payload = {
      "inputs" : prompt,
      "parameters" : { "max_new_tokens" : 1000 }
  }

  response = requests.post(API_URL, headers=headers, json=payload)
  return response.json()


req_count = 200
total_count = len(tokens_array) * req_count

idx = 0
API_TOKEN = tokens_array[idx]

curr_count = 0
count_sum = 0

for index, row in df.iterrows():

    if count_sum >= total_count:
      print(f'completed')
      break

    if curr_count == req_count:
        idx += 1
        curr_count = 0
        count_sum += req_count

        if idx >= len(tokens_array):
            print("All tokens used, exiting loop")
            break

        print(f'token change from {API_TOKEN} to {tokens_array[idx]}')
        API_TOKEN = tokens_array[idx]

    if pd.isna(row['recommendations']) or row['recommendations'].strip() == '':
        prompt = construct_prompt(row)
        recommendations = generate_rec(prompt,API_TOKEN)

        generated_text = recommendations[0]['generated_text']
        start_index = generated_text.find(prompt)
        clean_generated_text = generated_text[start_index + len(prompt):].strip()

        df.at[index, 'recommendations'] = clean_generated_text

        global_index = index
        curr_count += 1

        print(f'currently processing row {global_index} , current count {curr_count} , current token {API_TOKEN}')

    
df.to_csv('./dataset_creation.csv', index=False)