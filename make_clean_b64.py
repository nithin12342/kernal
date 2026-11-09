import base64, io, zipfile

b64 = open(r"d:\kernal\repo_core_b64.txt").read().strip()
lines = [b64[i:i+80] for i in range(0, len(b64), 80)]
formatted = "repo_b64 = (\n" + "\n".join([f'    "{l}"' for l in lines]) + "\n)\n"

loc = {}
exec(formatted, loc)
d = base64.b64decode(loc["repo_b64"])
z = zipfile.ZipFile(io.BytesIO(d))
print("Valid zip! Total files:", len(z.namelist()), "Total lines of b64:", len(lines))

with open(r"d:\kernal\repo_core_b64_clean_lines.txt", "w") as f:
    f.write(formatted)
