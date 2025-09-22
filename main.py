from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Blog2Social", page_icon="✍️", layout="centered")
st.title("Blog2Social")
st.write("Turn your ideas into blog posts and viral tweets in seconds")

# User input
input_text = st.text_input("Enter a topic or idea:")

# Memory
memory1 = ConversationBufferMemory(
    memory_key="chat_history", input_key="topic", output_key="text1")
memory2 = ConversationBufferMemory(
    memory_key="chat_history", input_key="text1", output_key="text2")

# Prompt templates
blog_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed, engaging blog post about {topic}. Make it structured with headings and paragraphs."
)

tweet_prompt = PromptTemplate(
    input_variables=["text1"],
    template="Summarize this blog into one catchy tweet under 280 characters: {text1}"
)

# LLM
llm = OpenAI(temperature=0.7)

# Chains
blog_chain = LLMChain(
    prompt=blog_prompt,
    llm=llm,
    verbose=True,
    output_key="text1",
    memory=memory1
)

tweet_chain = LLMChain(
    prompt=tweet_prompt,
    llm=llm,
    verbose=True,
    output_key="text2",
    memory=memory2
)

# Sequential chain
parent_chain = SequentialChain(
    chains=[blog_chain, tweet_chain],
    verbose=True,
    input_variables=["topic"],
    output_variables=["text1", "text2"]
)

# Run
if input_text:
    output = parent_chain({"topic": input_text})

    # Show results
    st.subheader("Generated Blog Post")
    st.write(output["text1"])

    st.subheader("Suggested Tweet")
    st.success(output["text2"])

    # Expanders to see memory/chat history
    with st.expander("Blog History"):
        st.info(memory1.buffer)
    with st.expander("Tweet History"):
        st.info(memory2.buffer)

    # Download blog post
    st.download_button(
        label="Download Blog Post",
        data=output["text1"],
        file_name=f"{input_text}_blog.txt",
        mime="text/plain"
    )
