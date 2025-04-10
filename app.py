import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains import LLMChain
import random


# Streamlit Config
st.set_page_config(page_title="GenAI YouTube Quiz Generator", page_icon="🧠")
st.title("🧠 GenAI: Learn from YouTube with Quizzes")
st.subheader("Enter a YouTube URL to generate a quiz")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# URL input
generic_url = st.text_input("Paste your YouTube/Website URL here", label_visibility="collapsed")

# Initialize LLM
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt for summarization
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Provide a concise educational summary of the following content in 300 words:\n\nContent:\n{text}"
)

# Prompt for MCQ generation
mcq_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
Based on the following summary, generate 15 multiple-choice questions (each with 1 correct and 3 wrong answers). 
Format:
Q: Question?
A. Option 1
B. Option 2
C. Option 3
D. Option 4
Answer: A/B/C/D

Summary:
{summary}
"""
)

# Processing
if st.button("Generate Quiz"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please enter both the Groq API key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL. Please provide a YouTube or website link.")
    else:
        try:
            with st.spinner("Loading content and generating quiz..."):
                # Load content
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                docs = loader.load()

                # Summarize
                summarize_chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
                summary = summarize_chain.run(docs)
                st.success("✅ Summary Generated!")

                # Show summary
                with st.expander("🔍 Summary", expanded=False):
                    st.write(summary)

                # Generate Questions
                mcq_chain = LLMChain(llm=llm, prompt=mcq_prompt)
                mcq_output = mcq_chain.run(summary)
                questions = mcq_output.strip().split("Q: ")[1:]

                user_score = 0
                feedback = []

                st.subheader("📝 Take the Quiz")
                for i, q_block in enumerate(questions):
                    lines = q_block.strip().split("\n")
                    question_text = lines[0].strip()
                    options = [line[3:].strip() for line in lines[1:5]]
                    correct_option = lines[5].split(":")[-1].strip()

                    user_answer = st.radio(f"Q{i+1}: {question_text}", options, key=f"q{i}")
                    if user_answer == options[ord(correct_option) - ord('A')]:
                        user_score += 1
                    else:
                        feedback.append((question_text, correct_option, options))

                if st.button("Submit Quiz"):
                    st.success(f"Your Score: {user_score} / {len(questions)}")
                    if feedback:
                        st.info("📌 Suggested Revisions:")
                        for q_text, correct, opts in feedback:
                            st.markdown(f"**Q: {q_text}**")
                            st.markdown(f"✅ Correct Answer: {opts[ord(correct) - ord('A')]}")
        except Exception as e:
            st.exception(f"Error: {e}")
