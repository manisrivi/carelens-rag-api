import io

from pypdf import PdfReader


def extract_text(content: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(content))
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return text, len(reader.pages)

def clean_text(text:str) -> str:
    lines = text.split("\n")
    cleaned = [" ".join(line.split()) for line in lines]
    return "\n".join([line for line in cleaned if line])

def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    lines = text.split("\n")
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        if current and current_len + len(line) > max_chars:
            chunks.append("\n".join(current))
            current = [current[-1]]
            current_len = len(current[0])

        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks
