import re
import sys
sys.path.append("src")
from highway.kernels.compute_kernels import AggregationKernel

kernel = AggregationKernel()
line = "Project PHOENIX is managed by Jean Dupont."
manager = "Jean Dupont"
project = "PHOENIX"

res = kernel._confirms_management_in_line(line, manager, project)
print("Confirms management in line:", res)

# Let's inspect each step:
line_folded = kernel._fold_accents(line).lower()
manager_folded = kernel._fold_accents(manager).lower()
project_folded = kernel._fold_accents(project).lower()

print("line_folded:", line_folded)
print("manager_folded:", manager_folded)
print("project_folded:", project_folded)

pattern_p = r'(?<![a-zA-Z0-9_\-])' + re.escape(project_folded) + r'(?![a-zA-Z0-9_\-])'
print("pattern_p matches:", bool(re.search(pattern_p, line_folded)))

for pattern_template in kernel.MANAGEMENT_PATTERNS:
    pattern = pattern_template.replace("{manager}", re.escape(manager_folded))
    match = bool(re.search(pattern, line_folded, re.IGNORECASE))
    print(f"Pattern '{pattern}':", match)



