import re
from typing import List, Tuple


def extract_mt_token_ids_v1(text: str) -> List[int]:
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]


def extract_mt_token_ids_v2(text: str) -> List[int]:
    pattern = re.compile(r"<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>")
    matches = pattern.findall(text)
    ret_list = []
    for num1, num2 in matches:
        ret_list.append(int(num1))
        ret_list.append(int(num2))
    return ret_list


def fix_mt_format_comprehensive(text: str) -> str:
    pattern_too_many = r"(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>"
    replacement_too_many = r"\1\2\3<|mt_end|>"
    text = re.sub(pattern_too_many, replacement_too_many, text)

    pattern_too_few_with_end = r"(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)"
    replacement_too_few = r"\1\2<|mt_9999|><|mt_end|>"
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)

    pattern_too_few_no_end = r"(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)"
    replacement_too_few_no_end = r"\1\2<|mt_9999|><|mt_end|>"
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text


def parse_quant_codes(
    text: str, codebook_size: int, codebook_depth: int
) -> List[List[int]]:
    quant_ids = extract_mt_token_ids_v1(text)
    if len(quant_ids) % codebook_depth != 0:
        fixed = fix_mt_format_comprehensive(text)
        quant_ids = extract_mt_token_ids_v2(fixed)

    if not quant_ids:
        return []

    batch_size = len(quant_ids) // codebook_depth
    remapped = []
    for bs_id in range(batch_size):
        chunk = quant_ids[bs_id * codebook_depth : (bs_id + 1) * codebook_depth]
        remap_chunk = []
        for book_id, quant_id in enumerate(chunk):
            remap_chunk.append(quant_id - book_id * codebook_size)
        remapped.append(remap_chunk)
    return remapped
