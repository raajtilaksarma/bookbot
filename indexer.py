''''
    Script to run indexing of documents in the 'data' directory.
'''


import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI
import time

os.environ['OPENAI_API_KEY']=""


start = time.time()
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))

# Configure prompt parameters and initialise helper
max_input_size = 2048
num_output = 256
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap,)

# Load documents from the 'data' directory
documents = SimpleDirectoryReader('data').load_data()
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)


index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist(persist_dir="stored_indices")
end = time.time()
print(f"Time taken: {end-start:.2f} seconds")

