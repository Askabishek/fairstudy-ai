from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from database import get_db, Topic, QuizAttempt
from sqlalchemy.orm import Session
from datetime import datetime
import os

class QuizTool:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.7, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192") # Using a smaller model for quiz generation for speed

    def generate_mcq(self, context: str, num_questions: int = 1):
        """
        Generates multiple-choice questions from the provided context.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant specialized in generating multiple-choice questions. Generate {num_questions} multiple-choice question(s) based on the following context. Provide 4 options and indicate the correct answer. Format the output as a JSON array of objects, each with 'question', 'options' (an array of strings), and 'correct_answer' fields."),
            ("user", "Context: {context}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "num_questions": num_questions})
        return response.content

    def evaluate_answer(self, question: str, user_answer: str, correct_answer: str, topic_name: str):
        """
        Evaluates a student's answer and provides feedback. Stores the attempt in the database.
        """
        is_correct = (user_answer.strip().lower() == correct_answer.strip().lower())
        feedback = "Correct!" if is_correct else f"Incorrect. The correct answer was {correct_answer}."

        db: Session = next(get_db())
        topic = db.query(Topic).filter(Topic.topic_name == topic_name).first()
        if not topic:
            topic = Topic(topic_name=topic_name)
            db.add(topic)
            db.commit()
            db.refresh(topic)

        quiz_attempt = QuizAttempt(
            topic_id=topic.id,
            question=question,
            user_answer=user_answer,
            correct_answer=correct_answer,
            is_correct=is_correct,
            attempt_time=datetime.utcnow()
        )
        db.add(quiz_attempt)

        if not is_correct:
            topic.difficulty += 1 # Increase difficulty for struggling topics
        else:
            topic.difficulty = max(0, topic.difficulty - 1) # Decrease difficulty, but not below 0
        topic.last_accessed = datetime.utcnow()
        db.commit()
        db.refresh(topic)

        return {"is_correct": is_correct, "feedback": feedback}
