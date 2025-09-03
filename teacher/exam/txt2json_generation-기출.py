import json
import re
import os

def parse_txt_questions(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    questions = []
    subject = ""
    q_pattern = re.compile(r"\[\s*(\d+)번\]\s*\d+\.\s*(.+)")
    opt_pattern = re.compile(r"^\s*[①-④]\s*(.+)")
    subject_pattern = re.compile(r"^=+\s*(.+?)\s*=+$")
    opt_split_pattern = re.compile(r"(①|②|③|④)\s*([^①②③④]+)")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 과목명 추출
        subj_match = subject_pattern.match(line)
        if subj_match:
            subject = subj_match.group(1).strip()
            i += 1
            continue

        # 문제 추출
        q_match = q_pattern.match(line)
        if q_match:
            q_num = int(q_match.group(1))
            q_text = q_match.group(2).strip()
            options = []
            # 다음 줄부터 보기 추출
            j = i + 1
            while j < len(lines):
                opt_line = lines[j].strip()
                # 인라인 보기 처리 (한 줄에 ①~④가 모두 있는 경우)
                if "①" in opt_line and "②" in opt_line and "③" in opt_line and "④" in opt_line:
                    opt_items = opt_split_pattern.findall(opt_line)
                    for _, text in opt_items:
                        options.append(text.strip())
                    j += 1
                    break
                # 각 보기별 줄 처리
                opt_match = opt_pattern.match(opt_line)
                if opt_match:
                    options.append(opt_match.group(1).strip())
                    j += 1
                else:
                    break
            # 보기 개수 보정 (무조건 4개)
            while len(options) < 4:
                options.append("")
            questions.append({
                "number": q_num,
                "question": q_text,
                "options": options,
                "answer": "",  # 답은 나중에 추가
                "explanation": "",
                "subject": subject
            })
            i = j
        else:
            i += 1
    return questions

def load_answers(answer_txt_path):
    answers = {}
    with open(answer_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
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

    for txt_file in os.listdir(txt_folder):
        if not txt_file.endswith(".txt"):
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