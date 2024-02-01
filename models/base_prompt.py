COT_STARTER = {
    "ar": "لنفكر خطوة بخطوة.",
    "bg": "Нека мислим стъпка по стъпка.",
    "de": "Lassen Sie uns Schritt für Schritt denken.",
    "el": "Ας σκεφτούμε βήμα βήμα.",
    "en": "Let's think step by step.",
    "es": "Pensemos paso a paso.",
    "fr": "Réfléchissons étape par étape.",
    "hi": "आइए कदम दर कदम सोचते हैं।",
    "ru": "Давайте думать шаг за шагом.",
    "sw": "Hebu tufikirie hatua kwa hatua.",
    "th": "มาคิดกันทีละขั้นตอน.",
    "tr": "Haydi adım adım düşünelim.",
    "ur": "آئیے قدم بہ قدم سوچتے ہیں۔",
    "vi": "Hãy suy nghĩ từng bước một.",
    "zh-CN": "让我们一步一步地思考。",
}

ANSWER_EXTRACTOR = {
    "en": "Therefore, the answer (English alphabet) is",
    "ar": "لذلك، الإجابة (أبجدية إنجليزية) هي",
    "bg": "Следователно, отговорът (английската азбука) е",
    "de": "Daher ist die Antwort (Englisches Alphabet) ist",
    "el": "Επομένως, η απάντηση (Αγγλικό αλφάβητο) είναι",
    "es": "Por lo tanto, la respuesta (alfabeto inglés) es",
    "fr": "Par conséquent, la réponse (alphabet anglais) est",
    "hi": "इसलिए, उत्तर (अंग्रेजी वर्णमाला) है",
    "ru": "Таким образом, ответ (английский алфавит) есть",
    "sw": "Hivyo, jibu (alfabeti ya Kiingereza) ni",
    "th": "ดังนั้นคำตอบ (อักษรภาษาอังกฤษ) คือ",
    "tr": "Bu nedenle, cevap (İngiliz alfabesi) şudur",
    "ur": "لہذا، جواب (انگریزی حروف تہجی) ہے",
    "vi": "Do đó, câu trả lời (bảng chữ cái tiếng Anh) là",
    "zh-CN": "因此，答案（英文字母）是",
}

def get_question_text(problem):
    # Compatability with the translated dataset
    if "translated_question" in problem:
        question = problem["translated_question"]
    else:
        question = problem["question"]
    return question


def get_context_text(problem):
    txt_context = problem["image"]
    return txt_context


def get_choice_text(problem, options):
    # Compatability with the translated dataset
    if "translated_choices" in problem:
        choices = problem["translated_choices"]
    else:
        choices = problem["choices"]
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    # print(choice_txt)
    return choice_txt


def get_options_text(problem, options):
    if "translated_choices" in problem:
        choices = problem["translated_choices"]
    else:
        choices = problem["choices"]
    options = options[: len(choices)]
    return "(" + ", ".join(options) + ")"


def get_answer(problem, options):
    if "translated_answer" in problem:
        return options[int(problem["translated_answer"])]
    return options[problem["answer"]]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem["lecture"].replace("\n", "\\n")
    return lecture


def get_solution_text(problem, translated=False):
    # \\n: GPT-3 can generate the solution with more tokens

    if translated:
        solution = problem["translated_solution"].replace("\n", "\\n")
    else:
        solution = problem["solution"].replace("\n", "\\n")
    return solution


def create_one_example(
    format,
    language,
    question,
    context,
    choice,
    answer,
    options,
    solution,
    test_example=True,
):
    cot_format, stages = format.split("-")

    input = f"Q: {question} [[IMG: {context}]]\n Choice: {choice}\n"

    if cot_format == "MCoT":
        cot_starter = COT_STARTER[language]
        answer_extractor = ANSWER_EXTRACTOR[language]
    elif cot_format == "EnCoT":
        cot_starter = COT_STARTER["en"]
        answer_extractor = ANSWER_EXTRACTOR["en"]
    elif cot_format == "Direct":
        cot_starter = ""
        answer_extractor = "The answer {Options} is"
    else:
        raise NotImplementedError

    answer_extractor = answer_extractor.replace("{Options}", options)

    # use Two
    if test_example:
        if stages == "Two":
            output = f"A: {cot_starter} [[{answer_extractor}]]"
        elif stages == "One":
            if cot_format == "Direct":
                output = f"A: {answer_extractor}"
            else:
                output = f"A: {cot_starter}"
        else:
            raise NotImplementedError
    else:
        if cot_format == "Direct":
            output = f"A: {answer_extractor} {answer}."
        else:
            output = f"A: {cot_starter} {solution} {answer_extractor} {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()
    return text


def build_prompt(problems, shot_qids, test_qid, args):
    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid])
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        options = get_options_text(problems[qid], args.options)
        translated = "M" in args.prompt_format
        solution = get_solution_text(problems[qid], translated)

        train_example = create_one_example(
            args.prompt_format,
            args.language,
            question,
            context,
            choice,
            answer,
            options,
            solution,
            test_example=False,
        )
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid])
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    options = get_options_text(problems[test_qid], args.options)
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(
        args.prompt_format,
        args.language,
        question,
        context,
        choice,
        answer,
        options,
        solution,
        test_example=True,
    )
    examples.append(test_example)

    # create the prompt input
    prompt_input = "\n\n".join(examples)

    return prompt_input
