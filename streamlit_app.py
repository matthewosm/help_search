import streamlit as st
import requests
import pandas as pd
import json

# Constants
EMBEDDING_API_URL = "https://api.openai.com/v1/embeddings"
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
PINECONE_QUERY_URL = "https://eleco-qa-abd38d4.svc.northamerica-northeast1-gcp.pinecone.io/query"

OPENAI_HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + st.secrets["openai_key"]
}

PINECONE_HEADERS = {
    'Content-Type': 'application/json',
    'Api-Key': st.secrets["pinecone_key"]
}

# Functions
def get_embedding_payload(text):
    return json.dumps({
        "input": text,
        "model": "text-embedding-ada-002"
    })

def get_pinecone_payload(response_vector):
    return json.dumps({
        "vector": response_vector,
        "topK": 5,
        "includeValues": False,
        "include_metadata": True
    })

def get_chat_payload(pine_content, text):
    return json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpul machine learning assistant and tutor. Answer questions based on the context provided, or say I don't know."
            },
            {
                "role": "user",
                "content": pine_content +'\n\n' + text +'?'
            }
        ]
    })

def get_responses(url, headers, payload):
    response = requests.request("POST", url, headers=headers, data=payload)
    return json.loads(response.text)

def get_pine_results(pine_response):
    pine_results = []
    pine_content = ''

    for pine_object in pine_response['matches']:
        pine_meta = pine_object['metadata']
        pine_link =  f'<a target="_blank" href="'+pine_meta['link']+'">'+pine_meta['title']+'</a>'
        pine_update = {
            'score': pine_object['score'],
            'link': pine_link
        }
        pine_results.append(pine_update)
        pine_content += pine_meta['text']

    return pine_results, pine_content

# Main
def main():
    st.image("https://lh3.googleusercontent.com/drive-viewer/AITFw-ywtnyvjMDy51gQVBtjcLzqNqC-HDzAtWE3I_QF_dg-g7o9ox5o0t5noY5YlHyFmoFJ9B1sJP5Ha8jXX_Iczl5mQTMtAw=s1600")
    st.title('Help Search')

    query = st.text_input('Enter your query here', '')

    if st.button('Search') or query:
        # Get embeddings
        embedding_payload = get_embedding_payload(query)
        embedding_response = get_responses(EMBEDDING_API_URL, OPENAI_HEADERS, embedding_payload)

        # Query Pinecone
        pine_payload = get_pinecone_payload(embedding_response["data"][0]["embedding"])
        pine_response = get_responses(PINECONE_QUERY_URL, PINECONE_HEADERS, pine_payload)

        # Get results and content
        pine_results, pine_content = get_pine_results(pine_response)

        # Get chat completions
        chat_payload = get_chat_payload(pine_content, query)
        chat_response = get_responses(CHAT_API_URL, OPENAI_HEADERS, chat_payload)
        gpt_response = chat_response['choices'][0]['message']['content']

        # Display results
        st.title('GPT Response')
        st.markdown(gpt_response)
        st.title('Related Articles')

        if pine_results:
            df = pd.DataFrame(pine_results)
            df = df.to_html(escape=False)
            st.write(df, unsafe_allow_html=True)
        else:
            st.write('No results found')

if __name__ == "__main__":
    main()