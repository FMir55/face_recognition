def match_info(id, label, best_similarity, id2cnt):
    print(
        f"Object id: {id}\n"\
        f"Identity: {label}\n"\
        f"Best Similarity: {best_similarity}\n"\
        f"Identity record:"
    )
    print(id2cnt[id].most_common())
