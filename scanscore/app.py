import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
import openai

# ---------- Configure OpenAI ----------
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# ---------- Answer Key Handling ----------
ANSWER_KEY_FILE = "answer_key.json"

def load_answer_key():
    if os.path.exists(ANSWER_KEY_FILE):
        with open(ANSWER_KEY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_answer_key(answer_key):
    with open(ANSWER_KEY_FILE, "w") as f:
        json.dump(answer_key, f, indent=4)

# ---------- Detect and Evaluate Bubbles ----------
def detect_answers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 500 > area > 200 and 0.8 <= aspect_ratio <= 1.2:
            bubble_contours.append((x, y, w, h))

    bubble_contours = sorted(bubble_contours, key=lambda b: (b[1], b[0]))
    detected_answers = {}
    question_number = 1
    options = ["A", "B", "C", "D"]

    for i in range(0, len(bubble_contours), 4):
        group = bubble_contours[i:i+4]
        if len(group) < 4:
            continue
        group = sorted(group, key=lambda b: b[0])

        filled_count = 0
        filled_option = None
        for j, (x, y, w, h) in enumerate(group):
            roi = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(roi)
            fill_ratio = total / (w*h)
            if fill_ratio > 0.5:
                filled_count += 1
                filled_option = options[j]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            elif 0.2 < fill_ratio <= 0.5:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)

        if filled_count > 1:
            detected_answers[str(question_number)] = "Multiple ❌"
        else:
            detected_answers[str(question_number)] = filled_option if filled_option else None

        question_number += 1

    return detected_answers, img

# ---------- Grade Answers ----------
def grade_answers(detected, answer_key):
    score = 0
    for q, ans in answer_key.items():
        if detected.get(q) == ans:
            score += 1
    total = len(answer_key)
    return score, total

# ---------- Student Feedback ----------
def student_feedback(detected_answers, answer_key):
    feedback = []
    wrong_questions = []
    for q, correct_ans in answer_key.items():
        student_ans = detected_answers.get(q)
        if student_ans is None:
            feedback.append(f"Q{q}: Not answered")
            wrong_questions.append(q)
        elif student_ans == "Multiple ❌":
            feedback.append(f"Q{q}: Multiple answers selected ❌")
            wrong_questions.append(q)
        elif student_ans != correct_ans:
            feedback.append(f"Q{q}: Wrong answer ❌ (Your: {student_ans}, Correct: {correct_ans})")
            wrong_questions.append(q)
        else:
            feedback.append(f"Q{q}: Correct ✅")
    return feedback, wrong_questions

# ---------- Weak Questions Analysis ----------
def weak_questions_analysis(df, answer_key):
    wrong_counts = {}
    df.columns = df.columns.astype(str)  # Ensure column names are strings
    for q in answer_key.keys():
        if str(q) in df.columns:
            wrong_counts[q] = df[str(q)].apply(
                lambda ans: ans != answer_key[q] or ans is None or ans == "Multiple ❌"
            ).sum()
        else:
            wrong_counts[q] = 0
    return wrong_counts

# ---------- GenAI Class Summary ----------
def generate_summary(df):
    avg_score = df["Score"].mean()
    hardest_questions = df.drop(columns=["Student","Score","Total"]).apply(lambda x: x.value_counts().get(None,0), axis=0)
    hardest_qs = hardest_questions.sort_values(ascending=False).head(5).index.tolist()
    prompt = f"""
    A class took an exam. The average score was {avg_score:.2f}.
    The hardest questions were: {hardest_qs}.
    Generate a short, insightful summary for teachers and students about class performance.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        summary = response.choices[0].text.strip()
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"

# ---------- GenAI Per-Student Suggestion ----------
def generate_student_suggestion(student_name, score, total, wrong_questions):
    if not openai.api_key:
        return "GenAI suggestion unavailable. Please set OPENAI_API_KEY."
    prompt = f"""
    Student {student_name} scored {score}/{total}.
    They got these questions wrong: {wrong_questions}.
    Generate a short, encouraging and constructive feedback message for this student.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=60,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating suggestion: {e}"

# ---------- Streamlit UI ----------
st.title("📊 Smart OMR Evaluator + GenAI Summary")

# --- Answer Key Creator ---
st.subheader("📝 Create / Edit Answer Key")
answer_key = load_answer_key()
num_questions = st.number_input(
    "Enter number of questions", min_value=1, step=1, value=len(answer_key) or 1
)
for i in range(1, num_questions+1):
    default_value = answer_key.get(str(i), "")
    answer = st.text_input(f"Answer for Question {i}", value=default_value)
    answer_key[str(i)] = answer.upper().strip()

if st.button("💾 Save Answer Key"):
    save_answer_key(answer_key)
    st.success(f"Answer key saved ✅")

# --- OMR Sheet Evaluator ---
st.subheader("📌 Upload OMR Sheets")
uploaded_files = st.file_uploader(
    "Upload one or more OMR sheets (PNG/JPG)", 
    type=["png","jpg","jpeg"], 
    accept_multiple_files=True
)

all_results = []

if uploaded_files:
    progress = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files, start=1):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detected, overlay_img = detect_answers(img)
        score, total = grade_answers(detected, answer_key)
        feedback, wrong_questions = student_feedback(detected, answer_key)

        result_row = {"Student": f"Student_{i}", "Score": score, "Total": total}
        result_row.update(detected)
        all_results.append(result_row)

        st.write(f"### 🧑‍🎓 Student_{i} → Score: {score}/{total}")
        st.write("Detected Answers:", detected)
        st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), caption=f"Student_{i} - Bubbles Detected")

        st.subheader(f"📝 Feedback for Student_{i}")
        for f in feedback:
            st.write(f)

        suggestion = generate_student_suggestion(f"Student_{i}", score, total, wrong_questions)
        st.subheader(f"🤖 GenAI Suggestion for Student_{i}")
        st.write(suggestion)

        progress.progress(i / len(uploaded_files))

    df = pd.DataFrame(all_results)
    df.columns = df.columns.astype(str)  # <-- important fix

    st.subheader("📊 Combined Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download All Results (CSV)",
        data=csv,
        file_name="scanscore_batch_results.csv",
        mime="text/csv",
    )

    st.subheader("📈 Analytics")
    df_numeric = df.drop(columns=["Student"])
    st.write("Average Score:", df_numeric["Score"].mean())
    st.bar_chart(df_numeric["Score"])

    st.subheader("📉 Weak Questions / Most Wrongly Answered")
    wrong_counts = weak_questions_analysis(df, answer_key)
    weak_df = pd.DataFrame.from_dict(wrong_counts, orient='index', columns=['Wrong Count'])
    weak_df = weak_df.sort_values(by='Wrong Count', ascending=False)
    st.bar_chart(weak_df)

    st.subheader("🤖 GenAI Class Performance Summary")
    class_summary = generate_summary(df)
    st.write(class_summary)
