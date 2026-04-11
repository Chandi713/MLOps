import matplotlib.pyplot as plt

from config import CATEGORIES


def analyze_metadata(df, split_name):
    print(f"\n{'='*60}\n{split_name.upper()} SPLIT\n{'='*60}")
    print(f"Total annotations: {len(df):,}")
    if "path" in df.columns:
        u = df["path"].nunique()
        print(f"Unique images: {u:,} | Avg annot/img: {len(df)/u:.1f}")
    cat_counts = df["category_id"].value_counts().sort_index()
    for cid, cnt in cat_counts.items():
        print(f"  {cid:2d}. {CATEGORIES.get(cid,'?'):25s}: {cnt:>7,} ({cnt/len(df)*100:5.1f}%)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(
        [CATEGORIES.get(i, "?") for i in cat_counts.index],
        cat_counts.values, color="steelblue"
    )
    axes[0].set_title(f"{split_name} — Categories")
    if "scale" in df.columns:
        sc = df["scale"].value_counts().sort_index()
        axes[1].bar(
            [{1: "Small", 2: "Med", 3: "Large"}.get(s, str(s)) for s in sc.index],
            sc.values, color="coral"
        )
        axes[1].set_title(f"{split_name} — Scale")
    plt.tight_layout()
    plt.show()
