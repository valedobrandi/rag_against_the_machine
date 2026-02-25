from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .models import MinimalSource

class RepositoryChunker:
    def __init__(self, chunk_size: int = 1800):
        self.chunk_size = chunk_size

    def chunk_file(self, file_path: str, content: str) -> List[dict]:
        """Chunks a single file and returns list of dicts with content and metadata."""

        if self.chunk_size <= 0 or self.chunk_size > 2000:
            raise ValueError("chunk_size must be between 1-2000.")
        
        # Select separators based on file type
        if file_path.endswith(".py"):
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_size // 4  # 40% overlap for code files
            )
        elif file_path.endswith(".md"):
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_size // 4  # 20% overlap for markdown files
            )
        else:
            # Fallback for other text files
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )

        # We need the character offsets to satisfy the MinimalSource model
        # LangChain doesn't give offsets by default, so we calculate them manually
        chunks = []
        raw_chunks = splitter.split_text(content)
        
        last_index = 0
        for chunk_text in raw_chunks:
            # Find the actual start index in the original content
            start_index = content.find(chunk_text, last_index)
            end_index = start_index + len(chunk_text)
            
            chunks.append({
                "content": chunk_text,
                "metadata": MinimalSource(
                    file_path=file_path,
                    first_character_index=start_index,
                    last_character_index=end_index
                )
            })
            last_index = start_index + 1 # Move forward for next search
            
        return chunks