import docx

doc = docx.Document('Project 4 TECR_Net_revised.docx')

with open('scratch/full_revised_text.txt', 'w', encoding='utf-8') as f:
    for i, p in enumerate(doc.paragraphs):
        f.write(f"[{i}] {p.text}\n")

print("Wrote full text to scratch/full_revised_text.txt")
