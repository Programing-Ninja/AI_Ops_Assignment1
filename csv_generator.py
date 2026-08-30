import csv
from pathlib import Path

rows = []
for split in ["train", "validation"]:
    for label in ["cats", "dogs"]:
        for f in sorted((Path("data") / split / label).iterdir()):
            if f.is_file():
                rows.append([f.name, f.as_posix(), split, label])

with open("filenames.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["filename", "filepath", "split", "label"])
    w.writerows(rows)

print(f"{len(rows)} rows written")
