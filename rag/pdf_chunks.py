import PyPDF2

def extract_chunks(pdf_path, chunk_size=100):
    reader = PyPDF2.PdfReader(pdf_path)
    text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

if __name__ == "__main__":
    chunks = extract_chunks("data/pdfs/sample.pdf")
    print(chunks[:2])
