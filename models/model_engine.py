from vlmeval.config import supported_VLM
from vlmeval.vlm import *
from vlmeval.smp.vlm import isimg
import re
import os


def cut_seq(seq, by=["\n\n", "Q:"]):
    min_idx = -1
    for b in by:
        idx = seq.find(b)
        if idx != -1:
            min_idx = min(min_idx, idx)
    if min_idx == -1:
        return seq
    return seq[:min_idx]


def auto_choose_model(model_name_or_path):
    model_name_or_path = model_name_or_path.lower()
    if "llava" in model_name_or_path:
        return LLaVA
    if "qwen" in model_name_or_path:
        return QwenVL
    if "blip" in model_name_or_path:
        return InstructBLIP
    raise ValueError(f"Unsupported model name: {model_name_or_path}")


def load_model(model_name_or_path, device_map="cuda", max_new_tokens=512):
    if device_map != "cuda":
        # get id:
        device_id = int(device_map.split(":")[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    if model_name_or_path in supported_VLM:
        return supported_VLM[model_name_or_path](max_new_tokens=max_new_tokens)
    else:
        return auto_choose_model(model_name_or_path)(
            model_path=model_name_or_path, max_new_tokens=max_new_tokens
        )


def model_predict(model, prompt):
    # Extract image
    pattern = re.compile(r"\[\[IMG:.*?\]\]")
    all_imgs = pattern.findall(prompt)

    input_list = []
    for img in all_imgs:
        idx = prompt.find(img)
        if not isimg(img[6:-2].strip()):
            print(f"Warning: {img} is not a valid image path")
            print(f"Current dir: {os.getcwd()}")
    
        if idx > 0:
            input_list.append(prompt[:idx].strip())
            input_list.append(img[6:-2].strip())
            prompt = prompt[idx + len(img) :]
    input_list.append(prompt)

    # Extract stage two prompt (extractor)
    extractor = None
    if "[[" in input_list[-1]:
        last_input = input_list[-1]
        pattern = re.compile(r"\[\[.*?\]\]")
        extractor = pattern.findall(last_input)[0][2:-2]
        last_input = pattern.sub("", last_input)
        input_list[-1] = last_input

    output = model.interleave_generate(input_list)
    output = cut_seq(output)
    seq_ret = output
    if extractor:
        input_list.append(output)
        input_list.append(extractor)
        output = model.interleave_generate(input_list)
        output = cut_seq(output)
        seq_ret += "\n" + extractor + output
    return seq_ret


if __name__ == "__main__":
    prompt = """Q: 后代与羊毛羊毛的预期比率是用毛茸茸的羊毛的预期比例？选择最可能的比率。 [[IMG: ../data/multi_lingual/images/test/42/image.png]]
    Choice: (A) 0：4 (B) 4：0 (C) 2：2 (D) 1：3 (E) 3：1
    A: 让我们根据图片一步一步地思考。 [[因此，答案 (字母) 是]]"""

    model = load_model(model_name_or_path="/cephfs/panwenbo/work/mmcot_assets/models/Qwen-VL", device_map="cuda:4", max_new_tokens=256)

    print(model_predict(model, prompt))