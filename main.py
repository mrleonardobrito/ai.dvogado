import os
import streamlit as st
from langchain_voyageai import VoyageAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
st.title("Me fale seu causo, meu guerreiro:")

template = """
    You're a lawyer. Given the context: {context} answer user questions. Giving the law article citation for your argument.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat = ChatOpenAI(temperature=0)


def query_llm(query):
    embeder = VoyageAIEmbeddings(
        model="voyage-2",
        show_progress_bar=True,
    )

    retriever = PineconeVectorStore(
        embedding=embeder,
        distance_strategy="cosine", 
        index_name="advogado-index"
    )

    trechos = retriever.similarity_search(query=query, k=10)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    result = chat.invoke(
        chat_prompt.format_prompt(
            context=trechos, text=query
        ).to_messages()
    )

    result = result.content
    st.session_state.messages.append((query, result))
    return result

def boot():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(query)
        st.chat_message("ai").write(response)

if __name__ == "__main__":
    boot()