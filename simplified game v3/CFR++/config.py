
def compute_hand_from_labels(hand):
    # Map label prefixes to normalized names
    type_map = {
        "gold": "Gold",
        "fire": "Fire",
        "empty": "Empty"
    }

    # Extract type from each string (e.g., 'fire_2' → 'Fire')
    result = []
    for item in hand:
        prefix = item.split('_')[0].lower()
        if prefix in type_map:
            result.append(type_map[prefix])

    # Define priority: Gold > Fire > Empty
    priority = {"Gold": 0, "Fire": 1, "Empty": 2}
    result.sort(key=lambda x: priority[x])

    return "".join(result)