
# author='raajs'

import os, streamlit as st
import time

os.environ['OPENAI_API_KEY']= ""

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext,load_index_from_storage,StorageContext
from langchain.llms.openai import OpenAI

# Define a simple Streamlit app
st.title("Ask BookBot")
query = st.text_input("What would you want to ask your book(s)?", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            if os.path.exists("stored_indices"):
                print('Cached index found, loading...')
                start = time.time()
                storage_context = StorageContext.from_defaults(persist_dir="stored_indices")
                index = load_index_from_storage(storage_context=storage_context)
                query_engine = index.as_query_engine(response_mode='compact')
                response = query_engine.query(query)
                response_section = query_engine.query("Chapters in the book for "+query)
                st.success(response)
                st.success(response_section)
                end = time.time()
                st.success(f"Time taken: {end-start:.2f} seconds")
            else:
                print('No cached index found, creating...')
        except Exception as e:
            st.error(f"An error occurred: {e}")


