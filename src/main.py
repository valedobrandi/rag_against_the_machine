import fire
import csv
import json
from pathlib import Path
from tqdm import tqdm
from .chunker import RepositoryChunker
from .retriever import BM25Retriever
from .generator import AnswerGenerator
from .models import StudentSearchResults, MinimalSearchResults, MinimalSource, DatasetRecallAtK, MinimalAnswer, StudentSearchResultsAndAnswer
from concurrent.futures import ThreadPoolExecutor

class RagCLI:
    def __init__(self):
        self.chunker = RepositoryChunker()
        self.retriever = BM25Retriever()
        self.index_path = Path("data/processed")

    def __get_text_from_answer(self, retrieved_sources: list[MinimalSource]) -> list[str]:
        """
        Helper function to extract text from retrieved sources.
        """
        texts = []
        for source in retrieved_sources:
            try:
                with open(source.file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    start = source.first_character_index
                    end = source.last_character_index
                    
                    text_chunk = content[start:end]
                    formatted = f"Source File: {source.file_path}\nContent:\n{text_chunk}\n"
                    texts.append(formatted)
            except Exception as e:
                print(f"Error reading {source.file_path}: {e}")
                continue
        return texts

    def index(self, repo_path: str = "data/raw/vllm-0.10.1", max_chunk_size: int = 2000):
        """
        Ingest and index the repository files.
        """
        print(f"Indexing repository at {repo_path}...")
        all_chunks = []
        repo_dir = Path(repo_path)
        
        files = list(repo_dir.rglob("*"))
        files = [f for f in files if f.suffix in [".py", ".md"]]
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                file_chunks = self.chunker.chunk_file(str(file_path), content)
                all_chunks.extend(file_chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        self.retriever.build_index(all_chunks)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(self.index_path))
        
        print(f"Ingestion complete! Indices saved under {self.index_path}")

    def search(self, query: str, k: int = 10):
        """
        Search the indexed repository for a single query.
        """

        self.retriever.load(str(self.index_path))

        results = self.retriever.search(query, k=k)

        print("f\nTop-{k} results for: '{query}'")
        print("="*50)

        for i, source in enumerate(results, 1):
            print(f"{i}. File: {source.file_path}")
            print(f"   Indices: {source.first_character_index} -> {source.last_character_index}")
            print("-" * 20)

    def search_dataset(self, dataset_path: str, k: int = 10, save_directory: str = "data/output/search_results"):
        """
        Process multiple questions and output StudentSearchResults JSON.
        """
        questions = Path(dataset_path)
        if not questions.exists() or not questions.is_file():   
            raise FileNotFoundError(f"File not found: {dataset_path}")

        self.retriever.load(str(self.index_path))
        questions_output = []
        try:
            with open(questions, "r", encoding="utf-8") as f:
                dataset = json.load(f)

                required_fields = {"question_id", "question"}
                for question in dataset.get("rag_questions", []):
                    if not required_fields.issubset(question.keys()):
                        raise ValueError(f"Must contain the following fields: {required_fields - set(question.keys())}")
                
                for question in dataset.get("rag_questions", []):
                    question_id = question["question_id"]
                    question_text = question["question"]

                    if not question_id or not question_text:
                        print(f"Skipping invalid entry: {question}")
                        continue

                    sources = self.retriever.search(question_text, k=k)

                    top_k_sources = []
                    for s in sources:
                        top_k_sources.append(MinimalSource(
                            file_path=s.file_path,
                            first_character_index=s.first_character_index,
                            last_character_index=s.last_character_index
                        ))
                    questions_output.append(
                            {
                                "question_id": question_id,
                                "question": question_text,
                                "retrieved_sources": [source.dict() for source in top_k_sources]
                            }
                        )
        except UnicodeDecodeError:
            raise ValueError(f"Error reading: {dataset_path}. Please ensure it is UTF-8 encoded.")

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        formatted_output = {
            "search_results": questions_output,
            "k": k
            }

        output_path = save_dir / Path(dataset_path).name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted_output, f, indent=2)
        
        print(f"Saved student_search_results to {output_path}")
    
    def answer_dataset(self, student_search_results_path: str, save_directory: str = "data/output/recall@k"):
        """
        Generate answers from search results for an entire dataset.        
        """
        with open(student_search_results_path, "r") as f:
            search_data = json.load(f)
        
        search_results_obj = StudentSearchResults(**search_data)

        generator = AnswerGenerator()
        answers = []

        print(f"Loaded {len(search_results_obj.search_results)} questions.")

        for result in tqdm(search_results_obj.search_results, desc="Generating answers"):
            retrieve_texts = self.__get_text_from_answer(result.retrieved_sources[:2])
            
            generate_text = generator.generate_answer(
                question=result.question,
                retrieved_sources=retrieve_texts
            )

            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=generate_text
            ))

        output_dataset = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results_obj.k
        )

        save_path = Path(save_directory) / Path(student_search_results_path).name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            f.write(output_dataset.model_dump_json(indent=4))
        
        print(f"Saved results to {save_path}")
