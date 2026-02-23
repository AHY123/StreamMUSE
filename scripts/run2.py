from scripts.dir_stats import RESULT, get_path
from scripts.json_processors import parse_by_type
import json

key = "interval1_gen3_prompt_128_gen_576"
base = "result/results-experiments2-local"

print(f"Testing key={key!r}, base_dir={base!r}\n")
for t in sorted(RESULT):
    p = get_path(key, t, base)
    print(f"TYPE: {t}")
    print(f"  PATH: {p}")
    try:
        parsed = parse_by_type(key, t, p)
        print(f"  parsed_count: {len(parsed)}")
        if len(parsed) > 0:
            # print a compact sample (first item) with limited length
            s = json.dumps(parsed[0], ensure_ascii=False)
            print(f"  sample: {s[:400]}{('...' if len(s) > 400 else '')}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
print("DONE")
