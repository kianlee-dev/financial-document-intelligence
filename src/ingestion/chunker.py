"""Text chunking with RecursiveCharacterTextSplitter."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents: list[Document]) -> list[Document]:
    #Initialise the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,      # Maximum characters per chunk, ~500 tokens (~2000 characters)
        chunk_overlap=400,    # Number of characters to overlap between chunks
    )  
    return text_splitter.split_documents(documents)
