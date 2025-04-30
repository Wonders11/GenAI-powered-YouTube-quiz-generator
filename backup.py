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

# Initialize LLM if API key is provided
if groq_api_key:
    try:
        llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
        # Test connection
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
    """Extract YouTube video ID from different URL formats"""
    if "youtube.com/watch" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/embed/" in url:
        return url.split("embed/")[1].split("?")[0]
    elif "youtube.com/v/" in url:
        return url.split("v/")[1].split("?")[0]
    else:
        return None

# Processing
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
                # Load content
                try:
                    docs = []
                    
                    # Check if it's a YouTube URL
                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        video_id = extract_youtube_id(generic_url)
                        
                        if not video_id:
                            st.error("Could not extract YouTube video ID from the URL.")
                            st.stop()
                        
                        st.info("Fetching YouTube transcript...")
                        
                        # Try method 1: youtube-transcript-api
                        try:
                            from youtube_transcript_api import YouTubeTranscriptApi
                            
                            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                            transcript_text = " ".join([item['text'] for item in transcript_list])
                            
                            # Get video title using pytube (just for metadata)
                            try:
                                import pytube
                                yt = pytube.YouTube(generic_url)
                                video_title = yt.title
                            except:
                                video_title = "YouTube Video"
                            
                            docs = [Document(page_content=transcript_text, metadata={"source": generic_url, "title": video_title})]
                            st.success("✅ YouTube transcript loaded successfully!")
                            
                        except Exception as transcript_error:
                            st.warning(f"Could not fetch transcript directly: {str(transcript_error)}")
                            st.info("Trying alternative method...")
                            
                            # Try method 2: pytube captions
                            try:
                                import pytube
                                yt = pytube.YouTube(generic_url)
                                video_title = yt.title
                                
                                captions = yt.captions.get_by_language_code('en')
                                if not captions:
                                    captions = next(iter(yt.captions), None)
                                
                                if captions:
                                    transcript = captions.generate_srt_captions()
                                    # Clean up the transcript
                                    transcript_text = re.sub(r'\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+', '', transcript)
                                    transcript_text = re.sub(r'^\d+$', '', transcript_text, flags=re.MULTILINE)
                                    
                                    docs = [Document(page_content=transcript_text, metadata={"source": generic_url, "title": video_title})]
                                    st.success("✅ YouTube captions loaded with fallback method!")
                                else:
                                    st.error("No captions available for this video.")
                                    st.stop()
                            except Exception as e:
                                st.error(f"All YouTube loading methods failed: {str(e)}")
                                st.error("Please try a different video or a regular website URL.")
                                st.stop()
                    else:
                        # Regular webpage
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                                "Accept-Language": "en-US,en;q=0.5"
                            }
                        )
                        docs = loader.load()
                        st.success("✅ Website content loaded successfully!")
                    
                    if not docs:
                        st.error("Could not extract content from the provided URL.")
                        st.stop()
                        
                except Exception as e:
                    st.error(f"Error loading content: {str(e)}")
                    st.code(traceback.format_exc())
                    st.stop()

                # Display document info
                with st.expander("📄 Document Information"):
                    st.markdown(f"**Source:** {docs[0].metadata.get('source', 'Unknown')}")
                    st.markdown(f"**Title:** {docs[0].metadata.get('title', 'Unknown')}")
                    st.markdown(f"**Content Length:** {len(docs[0].page_content)} characters")

                # Summarize
                try:
                    llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
                    summarize_chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
                    summary = summarize_chain.run(docs)
                    st.success("✅ Summary Generated!")
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    st.code(traceback.format_exc())
                    st.stop()

                # Show summary
                with st.expander("🔍 Summary", expanded=False):
                    st.write(summary)

                # Generate Questions
                try:
                    mcq_chain = LLMChain(llm=llm, prompt=mcq_prompt)
                    mcq_output = mcq_chain.run(summary)
                    
                    # Parse questions - handle different question formats (Q1: vs Q:)
                    if "Q:" not in mcq_output and "Q1:" not in mcq_output and not re.search(r'Q\d+:', mcq_output):
                        st.error("The AI didn't generate questions in the expected format.")
                        st.write("Raw output:")
                        st.code(mcq_output)
                        st.stop()
                    
                    # Handle different question formats
                    questions = []
                    
                    if re.search(r'Q\d+:', mcq_output):  # Q1:, Q2:, etc.
                        q_parts = re.split(r'Q\d+:', mcq_output)
                        # First part is usually empty or intro text
                        if q_parts[0].strip():
                            st.info("Skipping intro text before questions")
                        
                        for part in q_parts[1:]:  # Skip the first element which is before Q1
                            questions.append(part.strip())
                    elif "Q:" in mcq_output:  # Original format (Q:)
                        parts = mcq_output.split("Q:")
                        for part in parts[1:]:  # Skip the first element
                            questions.append(part.strip())
                    
                    # Debug info
                    with st.expander("🔍 Debug Question Parsing"):
                        st.markdown(f"**Found {len(questions)} questions**")
                        if len(questions) == 0:
                            st.code(mcq_output)
                    
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
                    st.code(traceback.format_exc())
                    st.stop()

                # Display quiz
                st.subheader("📝 Take the Quiz")
                
                # Create a form for the quiz
                with st.form(key="quiz_form"):
                    user_answers = []
                    
                    for i, q_block in enumerate(questions):
                        lines = q_block.strip().split("\n")
                        
                        # Skip questions with invalid format
                        if len(lines) < 5:
                            st.warning(f"Question {i+1} has an invalid format. Skipping.")
                            continue
                        
                        question_text = lines[0].strip()
                        options = []
                        answer_line = ""
                        
                        # Find options and answer line
                        for line in lines:
                            if line.strip().startswith(("A.", "B.", "C.", "D.")):
                                options.append(line[3:].strip())
                            elif "Answer:" in line:
                                answer_line = line.strip()
                        
                        # Check if we have all 4 options and an answer
                        if len(options) == 4 and answer_line:
                            user_answer = st.radio(f"Q{i+1}: {question_text}", options, key=f"q{i}")
                            answer_index = options.index(user_answer)
                            user_answers.append((i, question_text, options, user_answer, answer_index, answer_line))
                        else:
                            st.warning(f"Question {i+1} is missing options or answer. Skipping.")
                    
                    submit_button = st.form_submit_button("Submit Quiz")
                
                if submit_button:
                    user_score = 0
                    feedback = []
                    
                    for i, (q_idx, q_text, options, user_answer, answer_idx, answer_line) in enumerate(user_answers):
                        # Parse the correct answer from the answer line (e.g., "Answer: B")
                        try:
                            # Extract the letter only (A, B, C, or D)
                            correct_option = answer_line.split("Answer:")[1].strip()[0]
                            
                            if chr(ord('A') + answer_idx) == correct_option:
                                user_score += 1
                            else:
                                # Find the correct answer text
                                correct_answer = options[ord(correct_option) - ord('A')]
                                feedback.append((q_text, correct_option, correct_answer))
                        except Exception as e:
                            st.warning(f"Error parsing answer for question {i+1}: {str(e)}")
                    
                    st.success(f"Your Score: {user_score} / {len(user_answers)}")
                    
                    if feedback:
                        st.info("📌 Suggested Revisions:")
                        for q_text, correct_letter, correct_text in feedback:
                            st.markdown(f"**Q: {q_text}**")
                            st.markdown(f"✅ Correct Answer ({correct_letter}): {correct_text}")
                    
                    # Calculate percentage
                    percentage = (user_score / len(user_answers)) * 100 if user_answers else 0
                    st.progress(percentage / 100)
                    
                    # Give feedback based on percentage
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