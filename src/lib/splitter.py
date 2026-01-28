# uv add langchain-text-splitters


class Splitter:
    def __init__(self):
        pass



class RecursiveCharacterText_Splitter(Splitter):

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """Grok: What is a good chunk size and overlap for a RAG system?"""
        super().__init__()
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ".", "?", "!", ",", ";", ":", "---", "----", "-----"]
        )


    def get_chunks(self, text):
        chunks = self.text_splitter.split_text(text)
        return chunks   


if __name__ == "__main__":
    splitter = RecursiveCharacterText_Splitter(chunk_size=1500, chunk_overlap=250)
    chunks = splitter.get_chunks("Hello, world! This is a test of the splitter. It should split the text into chunks of 20 characters with 5 characters of overlap.")
    print(chunks)

