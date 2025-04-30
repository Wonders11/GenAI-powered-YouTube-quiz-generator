import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains import LLMChain
import traceback
from langchain.schema import Document
import re

# Streamlit Config
st.set_page_config(page_title="GenAI YouTube Quiz Generator", page_icon="🧠")
st.title("🧠 GenAI: Learn from YouTube with Quizzes")
st.subheader("Enter a YouTube URL to generate a quiz")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# URL input
generic_url = st.text_input("Paste your YouTube/Website URL here", label_visibility="collapsed")

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
Format each question like this:
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

def extract_youtube_id(url):
    if "youtube.com/watch" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/embed/" in url:
        return url.split("embed/")[1].split("?")[0]
    elif "youtube.com/v/" in url:
        return url.split("v/")[1].split("?")[0]
    return None

# Initialize LLM if API key is provided
if groq_api_key:
    try:
        llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
        with st.sidebar:
            if st.button("Test API Connection"):
                with st.spinner("Testing connection..."):
                    try:
                        test_response = llm.invoke("Hello")
                        st.success("✅ API connection successful!")
                    except Exception as e:
                        st.error(f"❌ API connection failed: {str(e)}")
    except Exception as e:
        st.sidebar.error(f"Error initializing LLM: {str(e)}")

# Main Quiz Generation Trigger
if st.button("Generate Quiz"):
    if not groq_api_key.strip():
        st.error("Please enter your Groq API key in the sidebar.")
    elif not generic_url.strip():
        st.error("Please enter a URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL. Please provide a YouTube or website link.")
    else:
        try:
            with st.spinner("Loading content and generating quiz..."):
                docs = []
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = extract_youtube_id(generic_url)
                    if not video_id:
                        st.error("Could not extract YouTube video ID from the URL.")
                        st.stop()
                    from youtube_transcript_api import YouTubeTranscriptApi
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = " ".join([item['text'] for item in transcript_list])
                    try:
                        import pytube
                        yt = pytube.YouTube(generic_url)
                        video_title = yt.title
                    except:
                        video_title = "YouTube Video"
                    docs = [Document(page_content=transcript_text, metadata={"source": generic_url, "title": video_title})]
                    st.success("✅ YouTube transcript loaded successfully!")
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False)
                    docs = loader.load()
                    st.success("✅ Website content loaded successfully!")

                summarize_chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
                summary = summarize_chain.run(docs)
                st.success("✅ Summary Generated!")

                with st.expander("🔍 Summary", expanded=False):
                    st.write(summary)

                mcq_chain = LLMChain(llm=llm, prompt=mcq_prompt)
                mcq_output = mcq_chain.run(summary)
                st.code(mcq_output, language="text")  # Show raw output

                questions = []
                if re.search(r'Q\d+:', mcq_output):
                    q_parts = re.split(r'Q\d+:', mcq_output)
                    questions = [q.strip() for q in q_parts[1:] if q.strip()]
                elif "Q:" in mcq_output:
                    q_parts = mcq_output.split("Q:")
                    questions = [q.strip() for q in q_parts[1:] if q.strip()]

                st.write(f"🛠️ DEBUG: Parsed {len(questions)} questions")

                st.subheader("📝 Take the Quiz")
                with st.form(key="quiz_form"):
                    user_answers = []
                    for i, q_block in enumerate(questions):
                        lines = q_block.strip().split("\n")
                        if len(lines) < 5:
                            st.warning(f"Question {i+1} has an invalid format. Skipping.")
                            continue

                        question_text = lines[0].strip()
                        options = []
                        answer_line = ""

                        for line in lines:
                            if re.match(r"^[A-D][\.)\-:]", line.strip()):
                                options.append(line.strip()[2:].strip())
                            elif "Answer:" in line:
                                answer_line = line.strip()

                        if len(options) == 4 and answer_line:
                            user_answer = st.radio(f"Q{i+1}: {question_text}", options, key=f"q{i}")
                            answer_index = options.index(user_answer)
                            user_answers.append((i, question_text, options, user_answer, answer_index, answer_line))
                        else:
                            st.warning(f"Question {i+1} is missing options or answer. Skipping.")

                    submit_button = st.form_submit_button("Submit Quiz")

                if submit_button:
                    st.write("✅ Submit button clicked")
                    st.write(f"🛠️ DEBUG: user_answers collected: {len(user_answers)}")

                    user_score = 0
                    feedback = []
                    for i, (q_idx, q_text, options, user_answer, answer_idx, answer_line) in enumerate(user_answers):
                        try:
                            match = re.search(r"Answer:\s*([A-D])", answer_line)
                            if match:
                                correct_option = match.group(1)
                                if chr(ord('A') + answer_idx) == correct_option:
                                    user_score += 1
                                else:
                                    correct_answer = options[ord(correct_option) - ord('A')]
                                    feedback.append((q_text, correct_option, correct_answer))
                            else:
                                st.warning(f"Could not find valid answer in: {answer_line}")
                        except Exception as e:
                            st.warning(f"Error parsing answer for question {i+1}: {str(e)}")

                    st.success(f"Your Score: {user_score} / {len(user_answers)}")
                    percentage = (user_score / len(user_answers)) * 100 if user_answers else 0
                    st.progress(percentage / 100)

                    if feedback:
                        st.info("📌 Suggested Revisions:")
                        for q_text, correct_letter, correct_text in feedback:
                            st.markdown(f"**Q: {q_text}**")
                            st.markdown(f"✅ Correct Answer ({correct_letter}): {correct_text}")

                    if percentage >= 90:
                        st.balloons()
                        st.success("Excellent! You've mastered this content! 🏆")
                    elif percentage >= 70:
                        st.success("Good job! You have a solid understanding of the material. 👍")
                    elif percentage >= 50:
                        st.warning("You're on the right track. Review the topics you missed and try again. 📚")
                    else:
                        st.error("You might need to review the content more thoroughly. Don't give up! 💪")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.code(traceback.format_exc())