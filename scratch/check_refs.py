with open('scratch/full_revised_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

import re
# find references section
pos = text.find('References')
if pos != -1:
    ref_part = text[pos:]
    print("References section length:", len(ref_part))
    # print all lines in ref_part
    with open('scratch/references_extracted.txt', 'w', encoding='utf-8') as out:
        out.write(ref_part)
    print("Wrote references to scratch/references_extracted.txt")
else:
    print("References header not found directly, searching...")
