from folder_paths import get_filename_list
import random
import torch

LORA_COUNT = 8

class EasyLoraWeightRandomizer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        args = {
            "total_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 10.00, "step": 0.01}),
            "min_single_strength": ("FLOAT", {"default": -1.00, "min": -10.00, "max": 0.00, "step": 0.01}),
            "max_single_strength": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01}),
            "randomize_total_strength": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }
        arg_lora_name = ([""] + get_filename_list("loras"),)
        for i in range(LORA_COUNT):
            args["{}:lora".format(i)] = arg_lora_name
        return {"required": args}

    def apply(self, total_strength, min_single_strength, max_single_strength, randomize_total_strength, seed, **kwargs):
        # 選択されたLoRA名をリストアップ
        selected_loras = []
        for i in range(LORA_COUNT):
            lora_name = kwargs["{}:lora".format(i)]
            if lora_name != "":
                selected_loras.append(lora_name)

        if not selected_loras:
            return ([],)  # 選択されたLoRAがない場合は空のリストを返す

        torch.manual_seed(seed)
        random.seed(seed)

        if randomize_total_strength:
            total_strength = round(random.uniform(0, total_strength), 2)

        num_selected = len(selected_loras)
        # シフトを用いて、各割り当てを [0, adjusted_max] の範囲で扱う
        shift = min_single_strength
        adjusted_total = total_strength - num_selected * shift
        adjusted_max = max_single_strength - min_single_strength

        # adjusted_totalが0未満または最大値を超える場合は、無理のない値に補正
        if adjusted_total < 0:
            adjusted_total = 0
        if adjusted_total > num_selected * adjusted_max:
            adjusted_total = num_selected * adjusted_max

        adjusted_strengths = [0.00] * num_selected
        remaining = adjusted_total
        allocation_order = list(range(num_selected))
        random.shuffle(allocation_order)

        # 最後以外の項目に対して、0～adjusted_maxの範囲内でランダムに割り当て
        for i in allocation_order[:-1]:
            allowed = min(remaining, adjusted_max)
            value = round(random.uniform(0, allowed), 2)
            adjusted_strengths[i] = value
            remaining -= value

        adjusted_strengths[allocation_order[-1]] = round(min(remaining, adjusted_max), 2)

        # 割り当て合計がadjusted_totalに満たない場合、余剰を各項目に分配
        allocated_total = sum(adjusted_strengths)
        if allocated_total < adjusted_total:
            diff = adjusted_total - allocated_total
            num_under_max = sum(1 for v in adjusted_strengths if v < adjusted_max)
            if num_under_max > 0:
                increment = diff / num_under_max
                for i in range(num_selected):
                    if adjusted_strengths[i] < adjusted_max:
                        add = min(increment, adjusted_max - adjusted_strengths[i])
                        adjusted_strengths[i] = round(adjusted_strengths[i] + add, 2)

        # シフト分を元に戻す
        strengths = [round(val + shift, 2) for val in adjusted_strengths]

        # 各LoRAの設定情報を [lora_name, model_strength, clip_strength] としてリスト化
        lora_stack = []
        for lora_name, strength in zip(selected_loras, strengths):
            lora_stack.append([lora_name, strength, strength])

        return (lora_stack,)

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("optional_lora_stack",)
    FUNCTION = "apply"
    CATEGORY = "tksw_node"
