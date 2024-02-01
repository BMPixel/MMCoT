import os
import re
import json
import argparse
import random
from models.base_prompt import *
from models.model_engine import load_model, model_predict, extract_prediction

def load_data(args):
    language = args.language
    problems = json.load(
        open(os.path.join(args.data_root, f"problems_{language}.json"))
    )
    pid_splits = json.load(open(os.path.join(args.data_root, "pid_splits.json")))
    
    # Add image path
    pid_not_exist = []
    pid_with_no_image = []
    for split in ['train', 'test', 'val']:
        for pid in pid_splits[split]:
            if pid not in problems:
                pid_not_exist.append(pid)
                continue
            if problems[pid]["image"]:
                problems[pid]["image"] = os.path.join(
                    args.data_root, "images", split, pid, problems[pid]["image"]
                )
                if not os.path.exists(problems[pid]["image"]):
                    print(f"Image not found: {problems[pid]['image']}")
            else:
                pid_with_no_image.append(pid)
    
    # Remove problems not exist
    print(f"Number of problems not exist: {len(pid_not_exist)}, total: {len(problems)}")
    for split in ['train', 'test', 'val']:
        pid_splits[split] = [pid for pid in pid_splits[split] if pid not in pid_not_exist]

    # Remove problems with no image
    print(f"Number of problems with no image: {len(pid_with_no_image)}, total: {len(problems) - len(pid_not_exist)}")
    for split in ['train', 'test', 'val']:
        pid_splits[split] = [pid for pid in pid_splits[split] if pid not in pid_with_no_image]

    qids = pid_splits["%s" % (args.test_split)]
    qids = qids[: args.test_number] if args.test_number > 0 else qids
    
    print(f"number of test problems: {len(qids)}\n")

    # pick up shot examples from the training set
    shot_qids = args.shot_qids
    train_qids = pid_splits["train"]
    if shot_qids == None:
        assert args.shot_number >= 0 and args.shot_number <= 32
        shot_qids = random.sample(train_qids, args.shot_number)  # random sample
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            assert qid in train_qids  # check shot_qids
    print("training question ids for prompting: ", shot_qids, "\n")

    return problems, qids, shot_qids


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[: len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


def get_result_file(args):
    result_file = "{}/{}/{}_{}_{}_{}_seed_{}.json".format(
        args.output_root,
        args.model,
        args.label,
        args.test_split,
        args.prompt_format,
        args.shot_number,
        args.seed,
    )

    return result_file


def save_results(result_file, acc, correct, count, shot_qids, args, results, outputs):
    data = {}
    data["acc"] = acc
    data["correct"] = correct
    data["count"] = count
    data["shot_qids"] = shot_qids
    data["args"] = vars(args)
    data["results"] = results
    data["outputs"] = outputs

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2, separators=(",", ": "))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        type=str,
        default="zh-CN",
        choices=[
            "ar",
            "bg",
            "de",
            "el",
            "en",
            "es",
            "fr",
            "hi",
            "ru",
            "sw",
            "th",
            "tr",
            "ur",
            "vi",
            "zh-CN",
        ],
    )
    parser.add_argument("--data_root", type=str, default="../data/multi_lingual")
    parser.add_argument("--output_root", type=str, default="../results")
    parser.add_argument("--options", type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument("--label", type=str, default="exp0")
    parser.add_argument(
        "--test_split", type=str, default="val", choices=["test", "val", "minival"]
    )
    parser.add_argument("--test_number", type=int, default=-1)
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save the result with every n examples.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="EnCoT-One",
        choices=["MCoT-One", "EnCoT-One", "MCoT-Two", "EnCoT-Two" "CodeSwitch-One", "CodeSwitch-Two", "Direct-One"],
        help="prompt format template",
    )
    parser.add_argument(
        "--shot_number", type=int, default=3, help="Number of n-shot training examples."
    )
    parser.add_argument(
        "--shot_qids", type=list, default=None, help="Question indexes of shot examples"
    )
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    # VLLM Settings
    parser.add_argument(
        "--model",
        type=str,
        default="mblip-bloomz-7b",
        help="huggingface compatible model name or path",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="The maximum number of tokens allowed for the generated answer.",
    )
    parser.add_argument("--device-map", type=str, default="cuda:0", help="device map")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    problems, qids, shot_qids = load_data(
        args
    )  # probelms, test question ids, shot example ids
    
    model = load_model(args.model, args.device_map, args.max_tokens)

    result_file = get_result_file(args)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(result_file))
        acc = check_point["acc"]
        correct = check_point["correct"]
        results = check_point["results"]
        outputs = check_point["outputs"]
        print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%")
    else:
        correct = 0
        results = {}
        outputs = {}

    # for qid in tqdm(qids):
    for i, qid in enumerate(qids):
        if qid in results:
            continue

        choices = problems[qid]["choices"]
        answer = problems[qid]["answer"]  # 0, 1, ..., 4
        label = args.options[answer]  # 'A', ..., 'E'

        # generate prompt
        prompt = build_prompt(problems, shot_qids, qid, args)

        # generate prediction
        # prediction, output = get_gpt3_result(prompt, args)  # 'A', ..., 'E'
        # some mock data for test
        output, prediction = model_predict(model, prompt, args.options)
        pred_idx = get_pred_idx(prediction, choices, args.options)  # 0, 1, ..., 4

        results[qid] = pred_idx
        outputs[qid] = output
        if pred_idx == answer:
            correct += 1

        acc = correct / len(results) * 100

        if args.debug or i < 10:
            print("##################################")
            print(prompt, "\n")
            print("# labeled answer:", label)
            print("# predicted answer:", prediction)
            print("# predicted index:", pred_idx)
            print("# predicted output:", output)

        if (i + 1) % args.save_every == 0 or (i + 1) == len(qids):
            print(
                f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}"
            )
            save_results(
                result_file, acc, correct, i + 1, shot_qids, args, results, outputs
            )
