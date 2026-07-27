from database import get_db, Topic, QuizAttempt
from sqlalchemy.orm import Session
from sqlalchemy import desc

class MemoryTool:
    def __init__(self):
        pass

    def get_struggling_topics(self, limit: int = 3):
        """
        Retrieves topics the student is struggling with, ordered by difficulty.
        """
        db: Session = next(get_db())
        topics = db.query(Topic).filter(Topic.difficulty > 0).order_by(desc(Topic.difficulty)).limit(limit).all()
        return [{"topic_name": t.topic_name, "difficulty": t.difficulty} for t in topics]

    def get_recent_quiz_attempts(self, topic_name: str = None, limit: int = 5):
        """
        Retrieves recent quiz attempts, optionally filtered by topic.
        """
        db: Session = next(get_db())
        query = db.query(QuizAttempt)
        if topic_name:
            topic = db.query(Topic).filter(Topic.topic_name == topic_name).first()
            if topic:
                query = query.filter(QuizAttempt.topic_id == topic.id)
            else:
                return [] # Topic not found
        attempts = query.order_by(desc(QuizAttempt.attempt_time)).limit(limit).all()
        return [
            {
                "question": a.question,
                "user_answer": a.user_answer,
                "correct_answer": a.correct_answer,
                "is_correct": a.is_correct,
                "attempt_time": str(a.attempt_time)
            }
            for a in attempts
        ]

    def update_topic_difficulty(self, topic_name: str, is_correct: bool):
        """
        Updates the difficulty of a topic based on quiz performance or repeated questions.
        """
        db: Session = next(get_db())
        topic = db.query(Topic).filter(Topic.topic_name == topic_name).first()
        if not topic:
            topic = Topic(topic_name=topic_name)
            db.add(topic)
            db.commit()
            db.refresh(topic)

        if not is_correct:
            topic.difficulty += 1
        else:
            topic.difficulty = max(0, topic.difficulty - 1)
        topic.last_accessed = datetime.utcnow()
        db.commit()
        db.refresh(topic)
        return {"status": "success", "topic_name": topic.topic_name, "difficulty": topic.difficulty}
