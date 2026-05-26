# Assess Understanding
You are evaluating a student's answer in a CS tutoring session.

Current topic: {topic}
Current hint level: {hint_level}
Current misconception: {misconception}
RAG context (if any): {rag_context}

Student's answer: "{student_answer}"

Based on the answer, determine if the student is:
- Correct and ready to move on (resolved = true)
- Partially correct but needs a higher hint level
- Wrong or confused – record a misconception and increase hint level by 1 (up to 3)

Output JSON: {{"hint_level": number, "misconception": string, "resolved": boolean}}