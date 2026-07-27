import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from pydantic import BaseModel
from utils import detect_language, get_gtts_lang_code

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

from database import init_db, get_db
from rag_tool import upload_syllabus_or_notes, get_rag_context
from quiz_tool import QuizTool
from voice_tool import VoiceTool
from memory_tool import MemoryTool

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize database
init_db()

# Initialize tools
quiz_tool_instance = QuizTool()
voice_tool_instance = VoiceTool()
memory_tool_instance = MemoryTool()

# Define LangChain tools
@tool
def rag_upload_tool(file_path: str):
    """
    Uploads a PDF file (syllabus or notes) to the vector store for RAG. 
    Use this tool when the user wants to upload study materials.
    """
    return upload_syllabus_or_notes(file_path)

@tool
def rag_context_tool(query: str):
    """
    Retrieves relevant context from the vector store based on a query. 
    Use this tool when the user asks a question that requires information from uploaded study materials.
    """
    return get_rag_context(query)

@tool
def quiz_generate_tool(context: str, num_questions: int = 1):
    """
    Generates multiple-choice questions from the provided context. 
    Use this tool when the user explicitly asks for a quiz or practice questions.
    """
    return quiz_tool_instance.generate_mcq(context, num_questions)

@tool
def quiz_evaluate_tool(question: str, user_answer: str, correct_answer: str, topic_name: str):
    """
    Evaluates a student's answer to a quiz question and provides feedback. 
    Use this tool when the user provides an answer to a quiz question.
    """
    return quiz_tool_instance.evaluate_answer(question, user_answer, correct_answer, topic_name)

@tool
def stt_tool(audio_file_path: str):
    """
    Converts speech from an audio file to text. 
    Use this tool when the user provides an audio input.
    """
    return voice_tool_instance.speech_to_text(audio_file_path)

@tool
def tts_tool(text: str, lang: str = "en"):
    """
    Converts text to speech and returns the path to the generated audio file. 
    Use this tool when the user asks for an audio response or if the original input was audio.
    """
    return voice_tool_instance.text_to_speech(text, lang)

@tool
def get_struggling_topics_tool(limit: int = 3):
    """
    Retrieves topics the student is struggling with, ordered by difficulty. 
    Use this tool when the user asks about their progress or what topics they need to review.
    """
    return memory_tool_instance.get_struggling_topics(limit)

@tool
def get_recent_quiz_attempts_tool(topic_name: str = None, limit: int = 5):
    """
    Retrieves recent quiz attempts, optionally filtered by topic. 
    Use this tool when the user asks to review past quiz attempts.
    """
    return memory_tool_instance.get_recent_quiz_attempts(topic_name, limit)

@tool
def update_topic_difficulty_tool(topic_name: str, is_correct: bool):
    """
    Updates the difficulty of a topic based on quiz performance or repeated questions. 
    This tool is called internally by the quiz evaluation, but can be used directly if needed for other forms of feedback.
    """
    return memory_tool_instance.update_topic_difficulty(topic_name, is_correct)

all_tools = [
    rag_upload_tool, rag_context_tool, 
    quiz_generate_tool, quiz_evaluate_tool, 
    stt_tool, tts_tool, 
    get_struggling_topics_tool, get_recent_quiz_attempts_tool, update_topic_difficulty_tool
]

# Initialize Groq LLM for the orchestrator
llm = ChatGroq(temperature=0.7, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")

# Orchestrator Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an AI study buddy. Your goal is to help students learn effectively. "
     "You have access to various tools to assist with RAG, quizzes, voice interactions, and tracking struggling topics. "
     "Detect the user's language and respond in the same language or code-mixed style if appropriate. "
     "Prioritize using the provided tools to fulfill user requests. "
     "If the user asks a question, first try to retrieve context using `rag_context_tool` if relevant. "
     "If the user wants to upload a document, use `rag_upload_tool`. "
     "If the user asks for a quiz, use `quiz_generate_tool`. "
     "If the user provides an answer to a quiz, use `quiz_evaluate_tool`. "
     "If the user provides audio, use `stt_tool` to convert it to text. "
     "If the user requests an audio response or if the original input was audio, use `tts_tool` to convert text to speech. "
     "If the user asks about their progress or struggling topics, use `get_struggling_topics_tool` or `get_recent_quiz_attempts_tool`. "
     "Always be helpful and encouraging."
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(llm, all_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)

class ChatRequest(BaseModel):
    text_input: Optional[str] = None
    audio_file_path: Optional[str] = None # Path to a temporary audio file if uploaded
    response_audio: bool = False

@app.post("/chat")
async def chat(request: ChatRequest, audio_file: Optional[UploadFile] = File(None)):
    user_input = request.text_input
    response_audio = request.response_audio
    
    if audio_file:
        # Save the uploaded audio file temporarily
        audio_filename = f"./audio_files/{audio_file.filename}"
        with open(audio_filename, "wb") as f:
            f.write(await audio_file.read())
        
        stt_result = voice_tool_instance.speech_to_text(audio_filename)
        if stt_result["status"] == "success":
            user_input = stt_result["text"]
            os.remove(audio_filename) # Clean up temporary audio file
        else:
            raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {stt_result['message']}")

    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided. Please provide text or audio.")

    try:
        # The agent will handle tool calling and response generation
        response = agent_executor.invoke({"input": user_input})
        agent_response_text = response["output"]

        if response_audio:
            detected_lang = detect_language(agent_response_text)
            lang = get_gtts_lang_code(detected_lang)

            tts_result = voice_tool_instance.text_to_speech(agent_response_text, lang=lang)
            if tts_result["status"] == "success":
                return {"response_text": agent_response_text, "response_audio_path": tts_result["audio_file_path"]}
            else:
                raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {tts_result['message']}")
        else:
            return {"response_text": agent_response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

