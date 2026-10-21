import docx

doc = docx.Document('Project 4 TECR_Net_revised.docx')

rels = doc.part.rels
images = []
for rel in rels.values():
    if "image" in rel.target_ref:
        images.append((rel.target_ref, len(rel.target_part.blob)))

print(f"Total embedded images: {len(images)}")
for ref, size in images:
    print(f"Image {ref}: {size} bytes")
