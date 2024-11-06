from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

model = SentenceTransformer('sentence-transformers/LaBSE')

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/LaBSE')  


def find_similar_documents(prompt, k=4):

    vectorstore = FAISS.load_local(
    "faiss_index_LaBSE_numeric",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
    )

    embed_prompt = model.encode(prompt).reshape(1, -1) 

    distances, indices = vectorstore.index.search(embed_prompt, k)

    similar_documents = [vectorstore.docstore[str(i)] for i in indices[0]]
    
    return similar_documents






