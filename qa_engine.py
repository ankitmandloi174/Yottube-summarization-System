from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from config import OPENAI_API_KEY

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    context = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "user",
            """You are an assistant that answers questions about YouTube videos based on their transcriptions.
Answer the following question: {question}
Searching in the following transcriptions: {docs}
Use only the information from the transcription to answer the question. If you don't know, respond with "I don't know".
Your answers should be detailed and verbose."""
        )
    ])

    chain = LLMChain(llm=llm, prompt=prompt, output_key="answer")
    response = chain({"question": query, "docs": context})
    return response["answer"], docs
