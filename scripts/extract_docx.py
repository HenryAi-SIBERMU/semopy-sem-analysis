
from docx import Document

files = [
    "Latar Belakang Penelitian Investasi IT_ROA_Efisiensi Operasional.docx",
    "Latar Belakang Penelitian Objek Inovasi Produk_Market value.docx"
]


with open('extracted_content.txt', 'w', encoding='utf-8') as f_out:
    for filename in files:
        f_out.write(f"--- START OF {filename} ---\n")
        try:
            doc = Document(filename)
            text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text)
            full_text = "\n".join(text)
            f_out.write(full_text + "\n")
        except Exception as e:
            f_out.write(f"Error reading {filename}: {e}\n")
        f_out.write(f"--- END OF {filename} ---\n")

