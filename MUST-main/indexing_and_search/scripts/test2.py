import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# 数据来自你的实验结果
# Set A: Target Only Correct (16531)
# Set B: Fusion Correct (17484)
# Intersection: Both Correct (16468) -> 16531 - 63 = 16468
# Only A (Broken by Fusion): 63
# Only B (Fixed by Fusion): 1016

plt.figure(figsize=(8, 6))
v = venn2(subsets=(63, 1016, 16468), set_labels=(
    'Target-Only Strategy', 'Fused Strategy'))

# 设置颜色和标签
v.get_label_by_id('10').set_text('63\n(Broken by Fusion)\nEvidence of M1')
v.get_label_by_id('01').set_text('1016\n(Gained by Fusion)')
v.get_label_by_id('11').set_text('16468\n(Robust Queries)')

# 突出显示 M1 的证据部分 (红色)
v.get_patch_by_id('10').set_color('red')
v.get_patch_by_id('10').set_alpha(0.5)

plt.title("Venn Diagram: Trade-off between Target-Only & Fused Search")
plt.savefig("M1_Verification_Venn.png")
plt.show()
