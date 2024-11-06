import faiss
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

df = pd.read_excel("payment_laws_numeric.xlsx")

model = SentenceTransformer('sentence-transformers/LaBSE')

splitted_documents = [
    Document(
        page_content=row["data"], 
        metadata={"index": idx}
    )
    for idx, row in df.iterrows()
]

texts = [doc.page_content for doc in splitted_documents]
embeddings = model.encode(texts, convert_to_tensor=False)

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

index_to_docstore_id = {i: str(i) for i in range(len(splitted_documents))}
docstore = {str(i): doc for i, doc in enumerate(splitted_documents)}

vectorstore = FAISS(
    index=faiss_index,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=model.encode,
    docstore=docstore
)

vectorstore.save_local("faiss_index_LaBSE_numeric")