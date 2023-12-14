from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

import numpy as np
from sklearn.cluster import KMeans

def summarize_doc(model,llm,file_data):
    print("in medium func")
    num_tokens = llm.get_num_tokens(file_data)
    print(num_tokens)
    if num_tokens > 2000:
        
        docs = file_to_docs(file_data,separator=["\n\n","\n"])
        print(docs)
        # print per document tokens
        print(f"total docs {len(docs)}")
        for i in range(len(docs)):
            print(llm.get_num_tokens(docs[i].page_content))
        
        summary_chain = load_summarize_chain(llm=llm,chain_type='map_reduce')
        summary = summary_chain.run(docs)

    else:
        template = """
                    Please write concise summary of the following text in less than 100 words.
                    Make sure summary has context of text and Do not add anything extra.

                    {text} 
                """
        prompt = PromptTemplate(input_variables=["text"],template=template)
        
        summary_prompt = prompt.format(text=file_data)
        
        summary = llm(summary_prompt)

    return summary


def summarize_long_doc(model,llm,api_key,file_data):
    print("using Kmeans")
    docs = file_to_docs(file_data)
    if len(docs) >= 11:
        vectors = doc_to_vector(model,api_key,docs)
        print(vectors)
        print("hello")
        # performing clustering to get most important part of books that help summarize book in a best way
        num_clusters = 11
        print(num_clusters)
        kmeans = KMeans(n_clusters=num_clusters,random_state=12)
        print(kmeans)
        kmeans.fit(vectors)
        print(kmeans)
        # we can consider embedding that is nearest to any cluster centroid is part that explain that cluster most.
        closest_indices = [] 

        for i in range(num_clusters):

            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i],axis=1)
            print(distances)
            closest_index = np.argmin(distances)
            print(closest_index)
            closest_indices.append(closest_index)
        print(closest_indices)

        selected_indices = sorted(closest_indices)
        print(selected_indices)

        ## summarizing closest index docs
        map_prompt = """You need to summarize single passage of book. This section of book is enclosed in triple quotes.
        Your goal is to give concise summary of this section so that user can have full understanding of what happened.
        Your response should be atleast of 3 paragraphs long and fully encompasses what was said in the passage.

        ```{text}```
        """
        map_prompt_template = PromptTemplate(template=map_prompt,input_variables=["text"])

        combine_prompt = """
            You will be given a series of summaries of book. All the summaries are enclosed in triple backticks (```)
            Your goal is to summarize this summaries concisely atleast in 100 words, so user can understand what happened in the book.
            ```{text}```
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt,input_variables=["text"])

        summary_chain = load_summarize_chain(llm=llm,chain_type='map_reduce',map_prompt=map_prompt_template,combine_prompt=combine_prompt_template,verbose=True)
        selected_docs = [docs[i] for i in selected_indices]
        print("selected docs",len(selected_docs))

        summary = summary_chain.run(selected_docs)
    else:
        summary = 1
    return summary

def file_to_docs(file_data,separator=["\n\n","\n","\t"]):
    print("in file to docs")
    text_splitter = RecursiveCharacterTextSplitter(separators=separator,chunk_size=8000,chunk_overlap=500)
    docs = text_splitter.create_documents([file_data])
    print(len(docs))
    return docs


def doc_to_vector(model,api_key,docs):
    if model == 'ChatGPT':
        # making vectors using openai embedder
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = embeddings.embed_documents([i.page_content for i in docs])
    elif model == 'Gemini-Pro':
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001',google_api_key=api_key)
        vectors = embeddings.embed_documents([i.page_content for i in docs])
        # embed_model = 'models/embedding-001'
        # vectors = genai.embed_content(model=embed_model,content=client_prompt)['embedding']
    return vectors