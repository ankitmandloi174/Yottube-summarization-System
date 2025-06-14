from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from config import OPENAI_API_KEY

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def create_vector_from_yt_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url, language="en")
    transcript = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(transcript)
    return FAISS.from_documents(docs, embeddings)
