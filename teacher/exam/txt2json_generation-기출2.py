import json
import re
import os

def parse_txt_questions(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    questions = []
    subject = ""
    q_pattern = re.compile(r"\[\s*(\d+)번\]\s*\d+\.\s*(.+)")
    subject_pattern = re.compile(r"^=+\s*(.+?)\s*=+$")
    opt_split_pattern = re.compile(r"(①|②|③|④)\s*([^①②③④]+)")

    i = 0
    total_lines = len(lines)
    while i < total_lines:
        line = lines[i]
        subj_match = subject_pattern.match(line)
        if subj_match:
            subject = subj_match.group(1).strip()
            i += 1
            continue

        q_match = q_pattern.match(line)
        if q_match:
            q_num = int(q_match.group(1))
            q_text_full = q_match.group(2).strip()
            question_text = ""
            options = []

            # 문제+보기가 한 줄에 모두 있는 경우
            if "①" in q_text_full:
                first_opt_idx = q_text_full.find("①")
                question_text = q_text_full[:first_opt_idx].strip()
                opts = opt_split_pattern.findall(q_text_full[first_opt_idx:])
                options = [text.strip() for _, text in opts]
            else:
                # 여러 줄에 걸쳐 문제+보기인 경우
                question_lines = [q_text_full]
                j = i + 1
                while j < total_lines:
                    next_line = lines[j]
                    if "①" in next_line:
                        opts = opt_split_pattern.findall(next_line)
                        options = [text.strip() for _, text in opts]
                        j += 1
                        break
                    elif any(next_line.startswith(x) for x in ["②", "③", "④"]):
                        opts = opt_split_pattern.findall(next_line)
                        options.extend([text.strip() for _, text in opts])
                        j += 1
                        break
                    else:
                        question_lines.append(next_line)
                        j += 1
                question_text = " ".join(question_lines).strip()
                i = j - 1  # 다음 문제로 이동

            while len(options) < 4:
                options.append("")

            questions.append({
                "number": q_num,
                "question": question_text,
                "options": options,
                "answer": "",
                "explanation": "",
                "subject": subject
            })
        i += 1
    return questions

def load_answers(answer_txt_path):
    answers = {}
    with open(answer_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if ':' not in line:
                continue
            num, ans = line.split(':', 1)
            num = num.strip()
            ans = ans.strip()
            if num.isdigit():
                answers[int(num)] = ans
    return answers

if __name__ == "__main__":
    txt_folder = "./teacher/exam/parsed_exam_txt"
    answer_folder = "./teacher/exam/parsed_exam_answer"
    json_folder = "./teacher/exam/parsed_exam_json"

    os.makedirs(json_folder, exist_ok=True)

    target_files = ["2025년2회_기사필기_전체문제.txt"]  # 원하는 파일 이름 리스트로 수정

    for txt_file in os.listdir(txt_folder):
        if not txt_file.endswith(".txt") or txt_file not in target_files:
            continue
        base_name = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(txt_folder, txt_file)
        answer_path = os.path.join(answer_folder, base_name + ".txt")
        json_path = os.path.join(json_folder, base_name + ".json")

        if not os.path.exists(answer_path):
            print(f"답 파일 없음: {answer_path} - 건너뜀")
            continue

        questions = parse_txt_questions(txt_path)
        answers = load_answers(answer_path)

        for q in questions:
            num = q.get("number")
            q["answer"] = answers.get(num, "")

        for q in questions:
            q.pop("number", None)

        data = {
            "exam_title": base_name,
            "total_questions": len(questions),
            "questions": questions
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{json_path} 저장 완료 ({len(questions)}문제)")