import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def unpack(archive, destination):
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if len(members) > 100 or sum(item.size for item in members) > 256 * 1024 * 1024:
            raise ValueError("Artifact exceeds extraction limits")
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError("Artifact contains an invalid path")
        source.extractall(destination, members=members, filter="data")


def pack(directory, archive):
    with tarfile.open(archive, "w:gz") as target:
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file():
                target.add(
                    path, arcname=str(path.relative_to(directory)), recursive=False
                )


def example_tokens(tokenizer, example, limit):
    messages = example["prompt"]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("SFT requires a final user message")
    completion = example["completion"]
    if len(completion) != 1 or completion[0].get("role") != "assistant":
        raise ValueError("This SFT recipe requires one assistant completion")
    content = completion[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("SFT target is empty")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    target = tokenizer.encode(content + tokenizer.eos_token, add_special_tokens=False)
    if len(prefix) + len(target) > limit:
        return None
    return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


def read_examples(path):
    examples = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line
    ]
    if not 1 <= len(examples) <= 10000:
        raise ValueError("Dataset size is invalid")
    return examples
