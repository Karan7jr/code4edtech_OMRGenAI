import cv2
import numpy as np
import json

# ---------- Load Answer Key ----------
def load_answer_key(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

# ---------- Detect Answers (Bubble Detection) ----------
def detect_answers(img):
    # Load image
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur + threshold (binary image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (bubbles are circular)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store bubbles
    bubble_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Filter by area + shape
        if 500 > area > 200 and 0.8 <= aspect_ratio <= 1.2:
            bubble_contours.append((x, y, w, h))

    # Sort bubbles top-to-bottom, then left-to-right
    bubble_contours = sorted(bubble_contours, key=lambda b: (b[1], b[0]))

    detected_answers = {}
    question_number = 1
    options = ["A", "B", "C", "D"]

    for i in range(0, len(bubble_contours), 4):  # Assuming 4 options per question
        group = bubble_contours[i:i+4]
        group = sorted(group, key=lambda b: b[0])  # Sort left to right

        filled = None
        for j, (x, y, w, h) in enumerate(group):
            roi = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(roi)
            if total > (0.5 * w * h):  # >50% filled
                filled = options[j]

        detected_answers[str(question_number)] = filled if filled else None
        question_number += 1

    return detected_answers

# ---------- Grade Answers ----------
def grade_answers(detected, answer_key):
    score = 0
    for q, ans in answer_key.items():
        if detected.get(q) == ans:
            score += 1
    total = len(answer_key)
    return score, total


# ---------- Main Runner ----------
if __name__ == "__main__":
    test_img = r"D:\Project\omr_filled_perfect.png"  # raw string to avoid escape errors
    answer_key = load_answer_key("answer_key.json")

    detected = detect_answers(test_img)
    print("Detected Answers:", detected)

    score = grade_answers(detected, answer_key)
    print(f"Final Score: {score}/{len(answer_key)}")
